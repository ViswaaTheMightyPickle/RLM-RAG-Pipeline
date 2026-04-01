"""HR-RAG CLI: Hierarchical Recursive RAG Pipeline Command Line Interface."""

import os
from pathlib import Path
from dotenv import load_dotenv
import click

from src.client import LLMClient
from src.rag_engine import RAGEngine
from src.orchestrator import HR_Orchestrator
from src.prompts import Persona
from src.models import RAGResponse


# Load environment variables
load_dotenv()


def get_config() -> dict:
    """Get configuration from environment variables."""
    return {
        "lm_studio_base_url": os.getenv(
            "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
        ),
        "worker_model_id": os.getenv("WORKER_MODEL_ID", "qwen2.5-0.5b"),
        "root_model_id": os.getenv("ROOT_MODEL_ID", "llama-3.1-8b"),
        "embedding_model_id": os.getenv(
            "EMBEDDING_MODEL_ID", "text-embedding-nomic-embed-text-v1.5"
        ),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "4000")),
        "summary_size": int(os.getenv("SUMMARY_SIZE", "500")),
        "voting_iterations": int(os.getenv("VOTING_ITERATIONS", "3")),
        "voting_threshold": float(os.getenv("VOTING_THRESHOLD", "0.7")),
        "data_dir": os.getenv("DATA_DIR", "./data"),
    }


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """HR-RAG: Hierarchical Recursive RAG Pipeline.

    A Worker-Orchestrator pipeline that uses a small model as a semantic
    filter to maximize the reasoning efficiency of a larger root model.
    """
    pass


@cli.command()
@click.argument("query")
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Voting threshold for discarded logs (0.0-1.0)",
)
@click.option(
    "--persona",
    default=None,
    type=click.Choice([p.value for p in Persona], case_sensitive=False),
    help="Force a specific worker persona",
)
@click.option(
    "--retrieve",
    default=10,
    type=int,
    help="Number of chunks to retrieve initially",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show verbose output including context used",
)
def ask(query: str, threshold: float, persona: str, retrieve: int, verbose: bool):
    """Ask a question using the HR-RAG pipeline.

    QUERY is the question you want to answer using the knowledge base.

    Examples:

        python main.py ask "What is the architecture of the system?"

        python main.py ask "Explain the gating mechanism" --persona TECHNICAL

        python main.py ask "What are the legal requirements?" --threshold 0.5
    """
    config = get_config()

    # Override config with command-line options
    if threshold is not None:
        config["voting_threshold"] = threshold

    # Initialize components
    click.echo("Initializing HR-RAG pipeline...")

    root_client = LLMClient(config["lm_studio_base_url"])
    worker_client = LLMClient(config["lm_studio_base_url"])
    rag_engine = RAGEngine(
        persist_directory=Path(config["data_dir"]) / "chroma_db",
        lm_studio_base_url=config["lm_studio_base_url"],
        embedding_model_id=config["embedding_model_id"],
    )

    orchestrator = HR_Orchestrator(
        root_client=root_client,
        worker_client=worker_client,
        rag_engine=rag_engine,
        root_model_id=config["root_model_id"],
        worker_model_id=config["worker_model_id"],
        voting_threshold=config["voting_threshold"],
        voting_iterations=config["voting_iterations"],
        summary_size=config["summary_size"],
        data_dir=config["data_dir"],
    )

    # Check if LM Studio is accessible
    if not root_client.check_health():
        click.secho(
            "Error: Cannot connect to LM Studio API. "
            "Please ensure LM Studio is running with the local server enabled.",
            fg="red",
        )
        raise click.Abort()

    click.echo(f"Processing query: {query}")
    click.echo(f"Retrieving {retrieve} chunks...")

    # Execute the pipeline
    response: RAGResponse = orchestrator.execute(query, n_retrieve=retrieve)

    # Display the result
    click.secho("\n" + "=" * 60, fg="cyan")
    click.secho("Final Answer:", fg="green", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo(response.answer)

    if verbose:
        click.secho("\n" + "-" * 60, fg="cyan")
        click.secho("Pipeline Details:", fg="yellow", bold=True)
        click.secho("-" * 60, fg="cyan")
        click.echo(f"Core Question: {response.core_question}")
        click.echo(f"Confidence: {response.confidence}")
        click.echo(f"Context Chunks Used: {len(response.context_used)}")
        click.echo(f"Discarded Items: {response.discarded_count}")
        click.echo(f"Recovered from Log: {response.recovered_count}")
        click.echo(f"Used Discarded Log: {response.used_discarded_log}")

        if response.context_used:
            click.secho("\nContext Summaries:", fg="yellow")
            for i, ctx in enumerate(response.context_used, 1):
                preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
                click.echo(f"  [{i}] {preview}")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--source",
    default=None,
    help="Source identifier (defaults to filename)",
)
@click.option(
    "--chunk-size",
    default=None,
    type=int,
    help="Chunk size in characters",
)
@click.option(
    "--chunk-overlap",
    default=200,
    type=int,
    help="Overlap between chunks in characters",
)
def ingest(
    file_path: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Ingest a document into the knowledge base.

    FILE_PATH is the path to the text file to ingest.

    Examples:

        python main.py ingest document.txt

        python main.py ingest manual.pdf --source "Product Manual v2.0"

        python main.py ingest report.txt --chunk-size 2000
    """
    config = get_config()

    # Override config with command-line options
    if chunk_size is None:
        chunk_size = config["chunk_size"]

    if source is None:
        source = Path(file_path).name

    click.echo(f"Ingesting: {file_path}")
    click.echo(f"Source: {source}")
    click.echo(f"Chunk size: {chunk_size} characters")
    click.echo(f"Chunk overlap: {chunk_overlap} characters")

    # Read the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        click.secho(
            f"Error: Could not read {file_path}. Please ensure it's a text file.",
            fg="red",
        )
        raise click.Abort()

    # Initialize RAG engine
    rag_engine = RAGEngine(
        persist_directory=Path(config["data_dir"]) / "chroma_db",
        lm_studio_base_url=config["lm_studio_base_url"],
        embedding_model_id=config["embedding_model_id"],
    )

    # Ingest the document
    num_chunks = rag_engine.ingest_document(
        text=content,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    click.secho(
        f"\nSuccessfully ingested '{source}': {num_chunks} chunks created.",
        fg="green",
    )


@cli.command()
@click.option(
    "--source",
    default=None,
    help="Delete only chunks from this source",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
def clear(source: str, yes: bool):
    """Clear the knowledge base or a specific source.

    Examples:

        python main.py clear  # Clear everything

        python main.py clear --source "document.txt"  # Clear specific source
    """
    config = get_config()
    rag_engine = RAGEngine(
        persist_directory=Path(config["data_dir"]) / "chroma_db",
        lm_studio_base_url=config["lm_studio_base_url"],
        embedding_model_id=config["embedding_model_id"],
    )

    stats = rag_engine.get_collection_stats()

    if source:
        if not yes:
            click.confirm(
                f"Delete all chunks from source '{source}'?", abort=True
            )

        deleted = rag_engine.delete_source(source)
        click.secho(f"Deleted {deleted} chunks from '{source}'.", fg="green")
    else:
        if not yes:
            click.confirm(
                f"Clear entire knowledge base ({stats['total_chunks']} chunks)?",
                abort=True,
            )

        rag_engine.clear_collection()
        click.secho("Knowledge base cleared.", fg="green")


@cli.command()
def stats():
    """Show knowledge base statistics."""
    config = get_config()
    rag_engine = RAGEngine(
        persist_directory=Path(config["data_dir"]) / "chroma_db",
        lm_studio_base_url=config["lm_studio_base_url"],
        embedding_model_id=config["embedding_model_id"],
    )

    stats = rag_engine.get_collection_stats()

    click.secho("Knowledge Base Statistics:", fg="cyan", bold=True)
    click.echo(f"Total chunks: {stats['total_chunks']}")
    click.echo(f"Unique sources: {stats['unique_sources']}")

    if stats["sources"]:
        click.secho("\nSources:", fg="yellow")
        for source in stats["sources"]:
            click.echo(f"  - {source}")


@cli.command()
def discarded():
    """Show discarded context log statistics."""
    config = get_config()

    from src.processor import WorkerProcessor
    from src.client import LLMClient

    worker_client = LLMClient(config["lm_studio_base_url"])
    processor = WorkerProcessor(
        client=worker_client,
        worker_model_id=config["worker_model_id"],
        data_dir=config["data_dir"],
    )

    stats = processor.get_discarded_stats()

    if stats["total_discarded"] == 0:
        click.echo("No discarded items in the log.")
        return

    click.secho("Discarded Context Log:", fg="cyan", bold=True)
    click.echo(f"Total discarded: {stats['total_discarded']}")
    click.echo(f"Root query: {stats['root_query']}")
    click.echo(f"Timestamp: {stats['timestamp']}")

    if stats["entries"]:
        click.secho("\nEntries:", fg="yellow")
        for entry in stats["entries"]:
            click.echo(
                f"  [{entry['cluster_id']}] {entry['source_pages']} "
                f"(score: {entry['relevance_score']:.2f})"
            )


@cli.command()
def clear_discarded():
    """Clear the discarded context log."""
    config = get_config()

    from src.processor import WorkerProcessor
    from src.client import LLMClient

    worker_client = LLMClient(config["lm_studio_base_url"])
    processor = WorkerProcessor(
        client=worker_client,
        worker_model_id=config["worker_model_id"],
        data_dir=config["data_dir"],
    )

    processor.clear_discarded_log()
    click.secho("Discarded context log cleared.", fg="green")


@cli.command()
def health():
    """Check the health of the LM Studio connection."""
    config = get_config()
    client = LLMClient(config["lm_studio_base_url"])

    click.echo(f"Checking LM Studio at {config['lm_studio_base_url']}...")

    if client.check_health():
        click.secho("✓ LM Studio is accessible!", fg="green")

        # Try to get models
        try:
            import requests

            response = requests.get(
                f"{config['lm_studio_base_url']}/models", timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                click.secho(f"\nAvailable models ({len(models)}):", fg="cyan")
                for model in models:
                    click.echo(f"  - {model.get('id', 'unknown')}")
        except Exception:
            pass
    else:
        click.secho(
            "✗ Cannot connect to LM Studio. "
            "Please ensure:\n"
            "  1. LM Studio is running\n"
            "  2. Local server is enabled (click the <-> icon)\n"
            "  3. The base URL is correct in .env",
            fg="red",
        )


if __name__ == "__main__":
    cli()
