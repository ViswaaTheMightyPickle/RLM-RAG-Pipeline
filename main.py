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
    include_patterns = os.getenv("INCLUDE_PATTERNS")
    exclude_patterns = os.getenv("EXCLUDE_PATTERNS")

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
        "include_patterns": include_patterns.split(",") if include_patterns else None,
        "exclude_patterns": exclude_patterns.split(",") if exclude_patterns else None,
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
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--chunk-size",
    default=None,
    type=int,
    help="Chunk size in characters (default: from .env)",
)
@click.option(
    "--chunk-overlap",
    default=None,
    type=int,
    help="Overlap between chunks in characters (default: from .env)",
)
def ingest(path: str, chunk_size: int, chunk_overlap: int):
    """Ingest documents from a file or directory into the knowledge base.

    PATH can be a single file or a directory (recursive by default).

    Supported formats:
      Documents: PDF, EPUB, TXT, TEXT
      Markdown: MD, MDX, Markdown, MDown, MKD, MKDN
      Web: HTM, HTML, XHTML

    Include/exclude patterns are configured in .env file.

    Examples:

        python main.py ingest document.pdf

        python main.py ingest ./docs/  # Ingest entire directory

        python main.py ingest ./knowledge --chunk-size 2000
    """
    config = get_config()

    # Override config with command-line options
    if chunk_size is None:
        chunk_size = config["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = 200

    path = Path(path)

    # Initialize RAG engine
    rag_engine = RAGEngine(
        persist_directory=Path(config["data_dir"]) / "chroma_db",
        lm_studio_base_url=config["lm_studio_base_url"],
        embedding_model_id=config["embedding_model_id"],
    )

    click.echo(f"Ingesting from: {path}")
    click.echo(f"Chunk size: {chunk_size} characters")
    click.echo(f"Chunk overlap: {chunk_overlap} characters")

    try:
        if path.is_file():
            # Single file ingestion
            from src.document_loader import is_supported_file, get_supported_extensions

            if not is_supported_file(path):
                supported = ", ".join(get_supported_extensions())
                click.secho(
                    f"Error: Unsupported file format '{path.suffix}'.\n"
                    f"Supported formats: {supported}",
                    fg="red",
                )
                raise click.Abort()

            num_chunks = rag_engine.ingest_file(
                file_path=path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            click.secho(
                f"\nSuccessfully ingested '{path.name}': {num_chunks} chunks created.",
                fg="green",
            )
        else:
            # Directory ingestion
            click.echo(f"Scanning directory: {path}")
            click.echo(f"Include patterns: {', '.join(config['include_patterns'] or ['*'])}")
            click.echo(f"Exclude patterns: {', '.join(config['exclude_patterns'] or ['none'])}")

            total_chunks, files_found, success_files, failed_files = rag_engine.ingest_directory(
                dir_path=path,
                include_patterns=config["include_patterns"],
                exclude_patterns=config["exclude_patterns"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            click.secho("\n" + "=" * 60, fg="cyan")
            click.secho("Ingestion Summary", fg="cyan", bold=True)
            click.secho("=" * 60, fg="cyan")
            click.echo(f"Files found:     {files_found}")
            click.echo(f"Files processed: {len(success_files)}")
            click.echo(f"Files failed:    {len(failed_files)}")
            click.echo(f"Total chunks:    {total_chunks}")

            if failed_files:
                click.secho("\nFailed files:", fg="red")
                for failed in failed_files:
                    click.echo(f"  - {failed}")

            if success_files:
                click.secho("\nSuccessfully processed:", fg="green")
                for success in success_files:
                    click.echo(f"  ✓ {success}")

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")
        raise click.Abort()


@cli.command()
@click.option(
    "--source",
    default=None,
    help="Delete only chunks from this source",
)
@click.option(
    "--all",
    "clear_all",
    is_flag=True,
    help="Clear everything (knowledge base + discarded log)",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
def clear(source: str, clear_all: bool, yes: bool):
    """Clear the knowledge base and/or discarded context log.

    Examples:

        python main.py clear  # Clear everything with confirmation

        python main.py clear --all  # Clear everything (no confirmation)

        python main.py clear --source "document.txt"  # Clear specific source

        python main.py clear --yes  # Skip confirmation
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
    elif clear_all:
        # Clear knowledge base
        if stats["total_chunks"] > 0 and not yes:
            click.confirm(
                f"Clear knowledge base ({stats['total_chunks']} chunks)?",
                abort=True,
            )

        if stats["total_chunks"] > 0:
            rag_engine.clear_collection()
            click.secho("Knowledge base cleared.", fg="green")
        else:
            click.echo("Knowledge base is already empty.")

        # Clear discarded log
        from src.processor import WorkerProcessor

        worker_client = LLMClient(config["lm_studio_base_url"])
        processor = WorkerProcessor(
            client=worker_client,
            worker_model_id=config["worker_model_id"],
            data_dir=config["data_dir"],
        )
        processor.clear_discarded_log()
        click.secho("Discarded context log cleared.", fg="green")
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
        # Group sources by file extension
        from collections import defaultdict
        by_extension = defaultdict(list)

        for source in stats["sources"]:
            ext = Path(source).suffix.lower() or "(no extension)"
            by_extension[ext].append(source)

        click.secho("\nSources by type:", fg="yellow")
        for ext in sorted(by_extension.keys()):
            sources = by_extension[ext]
            click.secho(f"\n  {ext.upper()} ({len(sources)} files):", fg="cyan")
            for source in sorted(sources):
                click.echo(f"    - {source}")


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
def init():
    """Initialize and verify the HR-RAG setup.

    This command performs a comprehensive health check:
    - Verifies LM Studio connection
    - Validates configured models are available
    - Checks .env configuration
    - Tests embedding API
    - Verifies ChromaDB setup

    Examples:

        python main.py init  # Run full setup verification
    """
    config = get_config()

    click.secho("=" * 60, fg="cyan", bold=True)
    click.secho("HR-RAG Setup Verification", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")

    all_ok = True

    # 1. Check .env configuration
    click.secho("\n[1/5] Checking configuration...", fg="yellow", bold=True)
    missing_vars = []
    required_vars = [
        "LM_STUDIO_BASE_URL",
        "WORKER_MODEL_ID",
        "ROOT_MODEL_ID",
        "EMBEDDING_MODEL_ID",
    ]

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            click.echo(f"  ✗ {var} not set")
        else:
            click.echo(f"  ✓ {var}={os.getenv(var)}")

    if missing_vars:
        click.secho(
            f"\nMissing required environment variables: {', '.join(missing_vars)}",
            fg="red",
        )
        all_ok = False
    else:
        click.secho("  Configuration OK!", fg="green")

    # 2. Check LM Studio connection
    click.secho("\n[2/5] Checking LM Studio connection...", fg="yellow", bold=True)
    client = LLMClient(config["lm_studio_base_url"])

    if not client.check_health():
        click.secho(
            "  ✗ Cannot connect to LM Studio.\n"
            "  Please ensure:\n"
            "    1. LM Studio is running\n"
            "    2. Local server is enabled (click the <-> icon)",
            fg="red",
        )
        all_ok = False
    else:
        click.secho("  ✓ LM Studio is accessible!", fg="green")

    # 3. Check available models
    click.secho("\n[3/5] Checking configured models...", fg="yellow", bold=True)

    try:
        import requests

        response = requests.get(
            f"{config['lm_studio_base_url']}/models", timeout=5
        )
        if response.status_code == 200:
            available_models = [
                m.get("id", "") for m in response.json().get("data", [])
            ]

            # Check worker model
            if config["worker_model_id"] in available_models:
                click.echo(f"  ✓ Worker model: {config['worker_model_id']}")
            else:
                click.secho(
                    f"  ✗ Worker model not found: {config['worker_model_id']}",
                    fg="red",
                )
                all_ok = False

            # Check root model
            if config["root_model_id"] in available_models:
                click.echo(f"  ✓ Root model: {config['root_model_id']}")
            else:
                click.secho(
                    f"  ✗ Root model not found: {config['root_model_id']}",
                    fg="red",
                )
                all_ok = False

            # Check embedding model
            if config["embedding_model_id"] in available_models:
                click.echo(f"  ✓ Embedding model: {config['embedding_model_id']}")
            else:
                click.secho(
                    f"  ✗ Embedding model not found: {config['embedding_model_id']}",
                    fg="red",
                )
                all_ok = False

            click.secho(f"\n  Available models ({len(available_models)}):", fg="cyan")
            for model in available_models:
                click.echo(f"    - {model}")
    except Exception as e:
        click.secho(f"  Error fetching models: {e}", fg="red")
        all_ok = False

    # 4. Test embedding API
    click.secho("\n[4/5] Testing embedding API...", fg="yellow", bold=True)

    try:
        embeddings_url = f"{config['lm_studio_base_url']}/embeddings"
        test_payload = {
            "model": config["embedding_model_id"],
            "input": "Test embedding",
        }
        response = requests.post(embeddings_url, json=test_payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get("data") and len(result["data"]) > 0:
                embedding_dim = len(result["data"][0].get("embedding", []))
                click.secho(
                    f"  ✓ Embedding API working (dimension: {embedding_dim})",
                    fg="green",
                )
            else:
                click.secho("  ✗ Embedding API returned invalid response", fg="red")
                all_ok = False
        else:
            click.secho(f"  ✗ Embedding API error: {response.status_code}", fg="red")
            all_ok = False
    except Exception as e:
        click.secho(f"  ✗ Embedding API test failed: {e}", fg="red")
        all_ok = False

    # 5. Check ChromaDB setup
    click.secho("\n[5/5] Checking ChromaDB setup...", fg="yellow", bold=True)

    try:
        from src.rag_engine import RAGEngine

        rag_engine = RAGEngine(
            persist_directory=Path(config["data_dir"]) / "chroma_db",
            lm_studio_base_url=config["lm_studio_base_url"],
            embedding_model_id=config["embedding_model_id"],
        )

        stats = rag_engine.get_collection_stats()
        click.secho(
            f"  ✓ ChromaDB initialized (chunks: {stats['total_chunks']})",
            fg="green",
        )
    except Exception as e:
        click.secho(f"  ✗ ChromaDB error: {e}", fg="red")
        all_ok = False

    # Final summary
    click.secho("\n" + "=" * 60, fg="cyan", bold=True)

    if all_ok:
        click.secho("✓ All checks passed! HR-RAG is ready to use.", fg="green", bold=True)
        click.secho("=" * 60, fg="cyan")
        click.echo("\nQuick start:")
        click.echo("  1. Ingest documents: python main.py ingest <path>")
        click.echo("  2. Ask questions:    python main.py ask \"<question>\"")
    else:
        click.secho(
            "✗ Some checks failed. Please fix the issues above.",
            fg="red",
            bold=True,
        )
        click.secho("=" * 60, fg="cyan")
        raise click.Abort()


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
