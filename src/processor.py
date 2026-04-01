"""Worker Model processing: distillation, gating, and probabilistic voting."""

import json
from pathlib import Path
from typing import Optional

from src.client import LLMClient
from src.models import (
    ClusterData,
    DiscardedContextLog,
    ProcessingResult,
    RejectedCluster,
)
from src.prompts import (
    Persona,
    get_distillation_prompt,
    get_gating_prompt,
    get_voting_prompt,
    PERSONA_PROMPTS,
)


class WorkerProcessor:
    """
    Worker Model processor for distillation, gating, and voting.

    Implements the core HR-RAG filtering mechanism using a small model
    to distill and evaluate text clusters.
    """

    def __init__(
        self,
        client: LLMClient,
        worker_model_id: str,
        summary_size: int = 500,
        voting_iterations: int = 3,
        voting_threshold: float = 0.7,
        data_dir: str = "./data",
    ):
        """
        Initialize the Worker Processor.

        Args:
            client: LLM client for API communication
            worker_model_id: Model ID for the Worker Model
            summary_size: Target size for distilled summaries (tokens)
            voting_iterations: Number of iterations for probabilistic voting
            voting_threshold: Threshold for promoting discarded items (0.0-1.0)
            data_dir: Directory for storing discarded context logs
        """
        self.client = client
        self.worker_model_id = worker_model_id
        self.summary_size = summary_size
        self.voting_iterations = voting_iterations
        self.voting_threshold = voting_threshold
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_log_path = self.data_dir / "discarded_context.json"

    def distill(
        self,
        cluster: ClusterData,
        persona: Persona,
    ) -> str:
        """
        Distill a cluster into a high-density summary.

        Args:
            cluster: The cluster data to distill
            persona: The persona to use for distillation

        Returns:
            The distilled summary text
        """
        prompt = get_distillation_prompt(
            persona, cluster.content, self.summary_size
        )

        # Get persona prompt - handle both Persona enum and string
        if isinstance(persona, Persona):
            persona_prompt = PERSONA_PROMPTS[persona]
        else:
            persona_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS[Persona.GENERAL])

        summary = self.client.generate(
            model_id=self.worker_model_id,
            system_msg=persona_prompt,
            user_msg=prompt,
            temperature=0.3,
        )

        return summary.strip()

    def gate(
        self,
        summary: str,
        core_question: str,
    ) -> bool:
        """
        Perform binary gating check on a summary.

        Args:
            summary: The distilled summary
            core_question: The core question to evaluate against

        Returns:
            True if relevant (YES), False if not relevant (NO)
        """
        prompt = get_gating_prompt(core_question, summary)

        response = self.client.generate(
            model_id=self.worker_model_id,
            system_msg="Answer only 'YES' or 'NO'.",
            user_msg=prompt,
            temperature=0.0,
        )

        return "YES" in response.upper()

    def vote(
        self,
        summary: str,
        core_question: str,
    ) -> tuple[int, int]:
        """
        Perform a single voting iteration.

        Args:
            summary: The distilled summary
            core_question: The core question to evaluate against

        Returns:
            Tuple of (votes_yes, votes_no) for this iteration (1,0 or 0,1)
        """
        prompt = get_voting_prompt(core_question, summary)

        response = self.client.generate(
            model_id=self.worker_model_id,
            system_msg="Persona: Voter. Answer only 'YES' or 'NO'.",
            user_msg=prompt,
            temperature=0.3,
        )

        if "YES" in response.upper():
            return 1, 0
        else:
            return 0, 1

    def probabilistic_voting(
        self,
        summary: str,
        core_question: str,
    ) -> tuple[float, int, int]:
        """
        Perform probabilistic voting with multiple iterations.

        Calculates score S = (sum of YES votes) / (total iterations)

        Args:
            summary: The distilled summary
            core_question: The core question to evaluate against

        Returns:
            Tuple of (relevance_score, votes_yes, votes_no)
        """
        votes_yes = 0
        votes_no = 0

        for _ in range(self.voting_iterations):
            yes, no = self.vote(summary, core_question)
            votes_yes += yes
            votes_no += no

        total_votes = votes_yes + votes_no
        score = votes_yes / total_votes if total_votes > 0 else 0.0

        return score, votes_yes, votes_no

    def process_cluster(
        self,
        cluster: ClusterData,
        core_question: str,
        persona: Persona,
    ) -> ProcessingResult:
        """
        Process a single cluster through distillation and gating.

        Args:
            cluster: The cluster data to process
            core_question: The core question for relevance evaluation
            persona: The persona to use for distillation

        Returns:
            ProcessingResult with summary and relevance information
        """
        # Step 1: Distill
        summary = self.distill(cluster, persona)

        # Step 2: Gate (1-pass check)
        is_relevant = self.gate(summary, core_question)

        return ProcessingResult(
            cluster_id=cluster.cluster_id,
            source_pages=cluster.source_pages,
            summary=summary,
            is_relevant=is_relevant,
        )

    def process_cluster_with_voting(
        self,
        cluster: ClusterData,
        core_question: str,
        persona: Persona,
    ) -> ProcessingResult:
        """
        Process a cluster with full probabilistic voting.

        Used for discarded items during recovery phase.

        Args:
            cluster: The cluster data to process
            core_question: The core question for relevance evaluation
            persona: The persona to use for distillation

        Returns:
            ProcessingResult with full voting information
        """
        # Step 1: Distill
        summary = self.distill(cluster, persona)

        # Step 2: Probabilistic voting
        score, votes_yes, votes_no = self.probabilistic_voting(
            summary, core_question
        )

        is_relevant = score >= self.voting_threshold

        return ProcessingResult(
            cluster_id=cluster.cluster_id,
            source_pages=cluster.source_pages,
            summary=summary,
            is_relevant=is_relevant,
            relevance_score=score,
            voting_iterations=self.voting_iterations,
            votes_yes=votes_yes,
            votes_no=votes_no,
        )

    def _load_discarded_log(self) -> DiscardedContextLog:
        """Load the discarded context log from disk."""
        if not self.discarded_log_path.exists():
            return DiscardedContextLog(root_query="")

        with open(self.discarded_log_path, "r") as f:
            data = json.load(f)
            return DiscardedContextLog(**data)

    def _save_discarded_log(self, log: DiscardedContextLog) -> None:
        """Save the discarded context log to disk."""
        with open(self.discarded_log_path, "w") as f:
            json.dump(log.model_dump(), f, indent=2)

    def log_discarded(
        self,
        root_query: str,
        results: list[ProcessingResult],
    ) -> None:
        """
        Log discarded (non-relevant) results to the JSON file.

        Args:
            root_query: The original user query
            results: List of ProcessingResult objects (non-relevant ones)
        """
        log = self._load_discarded_log()

        # Update root query if this is a new query
        if not log.root_query or log.root_query != root_query:
            log = DiscardedContextLog(root_query=root_query)

        for result in results:
            if not result.is_relevant:
                rejected = RejectedCluster(
                    cluster_id=result.cluster_id,
                    source_pages=result.source_pages,
                    persona_used="",
                    summary=result.summary,
                    relevance_score=result.relevance_score or 0.0,
                    iteration_count=result.voting_iterations,
                    votes_yes=result.votes_yes,
                    votes_no=result.votes_no,
                )
                log.rejected_clusters.append(rejected)

        self._save_discarded_log(log)

    def recover_from_discarded(
        self,
        root_query: str,
        core_question: str,
        persona: Persona,
    ) -> list[tuple[str, float]]:
        """
        Recover relevant items from the discarded context log.

        Runs probabilistic voting on all discarded items and promotes
        those meeting the threshold.

        Args:
            root_query: The original user query
            core_question: The core question for evaluation
            persona: The persona to use for re-evaluation

        Returns:
            List of (summary, score) tuples for promoted items
        """
        log = self._load_discarded_log()

        if log.root_query != root_query:
            # Different query, don't recover from old log
            return []

        promoted = []

        for rejected in log.rejected_clusters:
            # Re-run voting on the existing summary
            score, votes_yes, votes_no = self.probabilistic_voting(
                rejected.summary, core_question
            )

            if score >= self.voting_threshold:
                promoted.append((rejected.summary, score))

        return promoted

    def clear_discarded_log(self) -> None:
        """Clear the discarded context log."""
        if self.discarded_log_path.exists():
            self.discarded_log_path.unlink()

    def get_discarded_stats(self) -> dict:
        """
        Get statistics about the discarded context log.

        Returns:
            Dictionary with discarded log statistics
        """
        if not self.discarded_log_path.exists():
            return {"total_discarded": 0, "entries": []}

        log = self._load_discarded_log()
        return {
            "total_discarded": len(log.rejected_clusters),
            "root_query": log.root_query,
            "timestamp": log.timestamp,
            "entries": [
                {
                    "cluster_id": r.cluster_id,
                    "source_pages": r.source_pages,
                    "relevance_score": r.relevance_score,
                }
                for r in log.rejected_clusters
            ],
        }
