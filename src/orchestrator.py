"""Orchestrator: The brain coordinating the HR-RAG pipeline flow."""

from src.client import LLMClient
from src.models import ClusterData, RAGResponse, ProcessingResult
from src.prompts import (
    Persona,
    get_root_decompose_prompt,
    get_root_select_persona_prompt,
    get_root_synthesis_prompt,
    PERSONA_PROMPTS,
)
from src.rag_engine import RAGEngine
from src.processor import WorkerProcessor


class HR_Orchestrator:
    """
    Orchestrator for the Hierarchical Recursive RAG pipeline.

    Coordinates the full flow:
    1. Query decomposition (Root Model)
    2. Persona selection (Root Model)
    3. Document retrieval (RAG Engine)
    4. Distillation & Gating (Worker Model)
    5. Synthesis (Root Model)
    6. Backtracking/Recovery if needed
    """

    def __init__(
        self,
        root_client: LLMClient,
        worker_client: LLMClient,
        rag_engine: RAGEngine,
        root_model_id: str,
        worker_model_id: str,
        voting_threshold: float = 0.7,
        voting_iterations: int = 3,
        summary_size: int = 500,
        data_dir: str = "./data",
    ):
        """
        Initialize the Orchestrator.

        Args:
            root_client: LLM client for the Root Model
            worker_client: LLM client for the Worker Model
            rag_engine: RAG engine for document retrieval
            root_model_id: Model ID for the Root Model
            worker_model_id: Model ID for the Worker Model
            voting_threshold: Threshold for promoting discarded items
            voting_iterations: Number of voting iterations
            summary_size: Target size for distilled summaries
            data_dir: Directory for storing discarded context logs
        """
        self.root_client = root_client
        self.worker_client = worker_client
        self.rag_engine = rag_engine
        self.root_model_id = root_model_id
        self.worker_model_id = worker_model_id

        self.worker_processor = WorkerProcessor(
            client=worker_client,
            worker_model_id=worker_model_id,
            summary_size=summary_size,
            voting_iterations=voting_iterations,
            voting_threshold=voting_threshold,
            data_dir=data_dir,
        )

        self.voting_threshold = voting_threshold
        self.voting_iterations = voting_iterations

    def decompose_query(self, query: str) -> str:
        """
        Use Root Model to extract the core question from a query.

        Args:
            query: The user's query

        Returns:
            The core question string
        """
        prompt = get_root_decompose_prompt(query)

        core_question = self.root_client.generate(
            model_id=self.root_model_id,
            system_msg="You are the Orchestrator. Extract the core question from the user's query.",
            user_msg=prompt,
            temperature=0.0,
        )

        return core_question.strip()

    def select_persona(
        self,
        query: str,
        metadata: str = "General documents",
    ) -> Persona:
        """
        Use Root Model to select the appropriate Worker persona.

        Args:
            query: The user's query
            metadata: Document metadata context

        Returns:
            The selected Persona enum value
        """
        prompt = get_root_select_persona_prompt(query, metadata)

        response = self.root_client.generate(
            model_id=self.root_model_id,
            system_msg="Select the most appropriate persona for processing.",
            user_msg=prompt,
            temperature=0.0,
        )

        response = response.strip().upper()

        # Map response to Persona enum
        persona_map = {
            "TECHNICAL": Persona.TECHNICAL,
            "LEGAL": Persona.LEGAL,
            "NARRATIVE": Persona.NARRATIVE,
            "MEDICAL": Persona.MEDICAL,
            "FINANCIAL": Persona.FINANCIAL,
            "GENERAL": Persona.GENERAL,
        }

        return persona_map.get(response, Persona.GENERAL)

    def synthesize_answer(
        self,
        core_question: str,
        context: str,
    ) -> tuple[str, str]:
        """
        Use Root Model to synthesize the final answer.

        Args:
            core_question: The core question to answer
            context: The context from Worker Model

        Returns:
            Tuple of (answer, confidence_level)
        """
        prompt = get_root_synthesis_prompt(core_question, context)

        answer = self.root_client.generate(
            model_id=self.root_model_id,
            system_msg="You are the final reasoning engine. Provide comprehensive answers based on the context.",
            user_msg=prompt,
            temperature=0.3,
        )

        # Check confidence
        confidence_prompt = (
            "Evaluate your confidence in the answer above. "
            "If confident, respond with 'CONFIDENT'. "
            "If uncertain or needing more information, respond with 'LOW_CONFIDENCE'."
        )

        confidence_response = self.root_client.generate(
            model_id=self.root_model_id,
            system_msg="Evaluate your confidence level.",
            user_msg=confidence_prompt,
            temperature=0.0,
        )

        confidence = (
            "LOW_CONFIDENCE"
            if "LOW_CONFIDENCE" in confidence_response.upper()
            else "CONFIDENT"
        )

        return answer.strip(), confidence

    def execute(self, query: str, n_retrieve: int = 10) -> RAGResponse:
        """
        Execute the full HR-RAG pipeline.

        Args:
            query: The user's query
            n_retrieve: Number of chunks to retrieve initially

        Returns:
            RAGResponse with the answer and metadata
        """
        # Step 1: Decompose query to core question
        core_question = self.decompose_query(query)

        # Step 2: Select persona based on query
        persona = self.select_persona(query)

        # Step 3: Retrieve relevant chunks
        clusters = self.rag_engine.retrieve(query, n_results=n_retrieve)

        if not clusters:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                core_question=core_question,
                confidence="NO_CONTEXT",
            )

        # Step 4: Process clusters through Worker Model
        relevant_context = []
        discarded_results = []

        for cluster in clusters:
            result = self.worker_processor.process_cluster(
                cluster, core_question, persona
            )

            if result.is_relevant:
                relevant_context.append(result.summary)
            else:
                discarded_results.append(result)

        # Log discarded items
        if discarded_results:
            self.worker_processor.log_discarded(query, discarded_results)

        # Step 5: Synthesize answer with relevant context
        if relevant_context:
            context_text = "\n\n---\n\n".join(relevant_context)
            answer, confidence = self.synthesize_answer(
                core_question, context_text
            )

            # Check if backtracking is needed
            if confidence == "LOW_CONFIDENCE" or "INSUFFICIENT_CONTEXT" in answer.upper():
                # Step 6: Backtracking - recover from discarded log
                recovered = self.worker_processor.recover_from_discarded(
                    query, core_question, persona
                )

                if recovered:
                    # Add recovered context and re-synthesize
                    for summary, score in recovered:
                        relevant_context.append(
                            f"[Recovered (score: {score:.2f})] {summary}"
                        )

                    context_text = "\n\n---\n\n".join(relevant_context)
                    answer, confidence = self.synthesize_answer(
                        core_question, context_text
                    )

                    return RAGResponse(
                        answer=answer,
                        core_question=core_question,
                        context_used=relevant_context,
                        discarded_count=len(discarded_results),
                        recovered_count=len(recovered),
                        confidence=confidence,
                        used_discarded_log=True,
                    )

            return RAGResponse(
                answer=answer,
                core_question=core_question,
                context_used=relevant_context,
                discarded_count=len(discarded_results),
                confidence=confidence,
                used_discarded_log=False,
            )
        else:
            # No relevant context from initial pass - try recovery
            recovered = self.worker_processor.recover_from_discarded(
                query, core_question, persona
            )

            if recovered:
                recovered_context = [
                    f"[Recovered (score: {score:.2f})] {summary}"
                    for summary, score in recovered
                ]
                context_text = "\n\n---\n\n".join(recovered_context)
                answer, confidence = self.synthesize_answer(
                    core_question, context_text
                )

                return RAGResponse(
                    answer=answer,
                    core_question=core_question,
                    context_used=recovered_context,
                    discarded_count=len(discarded_results),
                    recovered_count=len(recovered),
                    confidence=confidence,
                    used_discarded_log=True,
                )

            return RAGResponse(
                answer="No relevant information found to answer the question.",
                core_question=core_question,
                context_used=[],
                discarded_count=len(discarded_results),
                confidence="NO_CONTEXT",
                used_discarded_log=False,
            )

    def execute_with_clusters(
        self,
        query: str,
        clusters: list[ClusterData],
        persona: Persona,
    ) -> RAGResponse:
        """
        Execute the pipeline with pre-retrieved clusters.

        Useful for custom retrieval strategies.

        Args:
            query: The user's query
            clusters: Pre-retrieved cluster data
            persona: The persona to use

        Returns:
            RAGResponse with the answer and metadata
        """
        core_question = self.decompose_query(query)

        relevant_context = []
        discarded_results = []

        for cluster in clusters:
            result = self.worker_processor.process_cluster(
                cluster, core_question, persona
            )

            if result.is_relevant:
                relevant_context.append(result.summary)
            else:
                discarded_results.append(result)

        if discarded_results:
            self.worker_processor.log_discarded(query, discarded_results)

        if relevant_context:
            context_text = "\n\n---\n\n".join(relevant_context)
            answer, confidence = self.synthesize_answer(core_question, context_text)

            return RAGResponse(
                answer=answer,
                core_question=core_question,
                context_used=relevant_context,
                discarded_count=len(discarded_results),
                confidence=confidence,
            )

        return RAGResponse(
            answer="No relevant information found.",
            core_question=core_question,
            confidence="NO_CONTEXT",
        )
