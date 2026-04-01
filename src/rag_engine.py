"""Vector search and document chunking using ChromaDB."""

import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
import requests

from src.models import ClusterData
from src.document_loader import load_document, is_supported_file, get_supported_extensions


class RAGEngine:
    """
    RAG Engine for document ingestion, chunking, and vector retrieval.

    Uses ChromaDB for persistent vector storage and similarity search.
    Uses LM Studio API for embeddings (OpenAI-compatible).
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        lm_studio_base_url: str = "http://localhost:1234/v1",
        embedding_model_id: str = "text-embedding-nomic-embed-text-v1.5",
    ):
        """
        Initialize the RAG Engine.

        Args:
            persist_directory: Directory for ChromaDB persistence
            lm_studio_base_url: Base URL for LM Studio API
            embedding_model_id: Model ID for embeddings (from LM Studio)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.lm_studio_base_url = lm_studio_base_url
        self.embedding_model_id = embedding_model_id
        self.embeddings_url = f"{lm_studio_base_url}/embeddings"

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Default collection for document chunks
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Document chunks for RAG retrieval"},
        )

    def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector from LM Studio API.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        payload = {
            "model": self.embedding_model_id,
            "input": text,
        }
        
        response = requests.post(self.embeddings_url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["data"][0]["embedding"]

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        payload = {
            "model": self.embedding_model_id,
            "input": texts,
        }
        
        response = requests.post(self.embeddings_url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]

    def _generate_id(self, content: str, source: str) -> str:
        """Generate a unique ID for a chunk based on content and source."""
        combined = f"{source}:{content[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _chunk_text(
        self, text: str, chunk_size: int = 4000, overlap: int = 200
    ) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size // 2:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap if end < len(text) else len(text)

        return chunks

    def ingest_document(
        self,
        text: str,
        source: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest a document into the vector store.

        Args:
            text: The document text
            source: Source identifier (filename, URL, etc.)
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks created
        """
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        metadata = metadata or {}

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_id(chunk, source)
            chunk_metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata,
            }

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)

        if ids:
            # Get embeddings from LM Studio API
            embeddings = self._get_embeddings(documents)
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        return len(chunks)

    def ingest_documents(
        self,
        documents: list[tuple[str, str, Optional[dict]]],
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Ingest multiple documents.

        Args:
            documents: List of (text, source, metadata) tuples
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Total number of chunks created
        """
        total_chunks = 0
        for text, source, metadata in documents:
            chunks = self.ingest_document(
                text, source, chunk_size, chunk_overlap, metadata
            )
            total_chunks += chunks
        return total_chunks

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[ClusterData]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of ClusterData objects
        """
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        
        query_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        clusters = []
        if query_result["ids"] and query_result["ids"][0]:
            for i, doc_id in enumerate(query_result["ids"][0]):
                doc = query_result["documents"][0][i]
                meta = query_result["metadatas"][0][i]
                distance = query_result["distances"][0][i] if query_result["distances"] else None

                clusters.append(
                    ClusterData(
                        cluster_id=i,
                        content=doc,
                        source_pages=meta.get("source", "unknown"),
                        metadata={
                            **meta,
                            "retrieval_distance": distance,
                        },
                    )
                )

        return clusters

    def retrieve_by_source(
        self,
        query: str,
        source: str,
        n_results: int = 5,
    ) -> list[ClusterData]:
        """
        Retrieve chunks from a specific source.

        Args:
            query: The search query
            source: The source to filter by
            n_results: Number of results to return

        Returns:
            List of ClusterData objects
        """
        return self.retrieve(
            query,
            n_results,
            filter_metadata={"source": source},
        )

    def get_all_sources(self) -> list[str]:
        """
        Get all unique sources in the vector store.

        Returns:
            List of unique source identifiers
        """
        # Get a sample to extract sources
        sample = self.collection.get(include=["metadatas"], limit=1000)
        sources = set()

        if sample["metadatas"]:
            for meta in sample["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])

        return sorted(list(sources))

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        sources = self.get_all_sources()

        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "sources": sources,
        }

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection("document_chunks")
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Document chunks for RAG retrieval"},
        )

    def delete_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Args:
            source: The source identifier to delete

        Returns:
            Number of chunks deleted
        """
        # Get all IDs for this source
        existing = self.collection.get(
            where={"source": source},
            include=[],
        )

        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            return len(existing["ids"])

        return 0

    def ingest_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file to ingest
            source: Source identifier (defaults to file path)
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks created

        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not is_supported_file(file_path):
            supported = ", ".join(get_supported_extensions())
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported: {supported}"
            )

        # Load document content
        content = load_document(file_path)

        # Use file path as source if not provided
        if source is None:
            source = str(file_path)

        return self.ingest_document(
            text=content,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_directory(
        self,
        dir_path: Path,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        show_progress: bool = True,
    ) -> tuple[int, int, list[str], list[str]]:
        """
        Ingest all supported files from a directory recursively.

        Args:
            dir_path: Path to the directory to ingest
            include_patterns: Glob patterns for files to include (e.g., ["*.pdf", "*.md"])
            exclude_patterns: Patterns to exclude (e.g., ["node_modules", "*.min.js"])
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            show_progress: Whether to show progress indicators

        Returns:
            Tuple of (total_chunks, files_processed, success_files, failed_files)
        """
        import fnmatch

        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        # Default include patterns - all supported extensions
        if include_patterns is None:
            include_patterns = [f"*.{ext}" for ext in get_supported_extensions()]

        # Default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = [
                "node_modules", ".git", "__pycache__",
                "*.pyc", "*.pyo", "venv", ".venv"
            ]

        all_files = []
        failed_files = []
        success_files = []
        total_chunks = 0

        # Recursively collect files
        for root, dirs, files in dir_path.walk() if hasattr(dir_path, 'walk') else self._walk_directory(dir_path):
            # Filter out excluded directories
            dirs[:] = [
                d for d in dirs
                if not any(
                    fnmatch.fnmatch(d, pattern.rstrip('*')) or
                    fnmatch.fnmatch(str(root / d), f"*{pattern}*")
                    for pattern in exclude_patterns
                )
            ]

            # Process files
            for file in files:
                file_path = root / file

                # Check if file matches include patterns
                include_match = any(
                    fnmatch.fnmatch(file, pattern)
                    for pattern in include_patterns
                )

                if not include_match:
                    continue

                # Check if file matches exclude patterns
                exclude_match = any(
                    fnmatch.fnmatch(file, pattern) or
                    fnmatch.fnmatch(str(file_path), f"*{pattern}*")
                    for pattern in exclude_patterns
                )

                if exclude_match:
                    continue

                # Check if file is supported
                if not is_supported_file(file_path):
                    continue

                all_files.append(file_path)

        # Ingest all collected files with progress
        total_files = len(all_files)
        for i, file_path in enumerate(all_files, 1):
            if show_progress:
                # Show progress: [1/10] filename
                rel_path = file_path.relative_to(dir_path) if file_path.is_relative_to(dir_path) else file_path
                print(f"[{i}/{total_files}] Processing: {rel_path}", end="", flush=True)
            
            try:
                chunks = self.ingest_file(
                    file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                total_chunks += chunks
                success_files.append(str(file_path))
                
                if show_progress:
                    print(f" ✓ ({chunks} chunks)")
            except Exception as e:
                failed_files.append(f"{file_path}: {str(e)}")
                if show_progress:
                    print(f" ✗ (Error: {str(e)})")

        if show_progress:
            print()  # New line after completion

        return total_chunks, len(all_files), success_files, failed_files

    def _walk_directory(self, dir_path: Path):
        """
        Generator to walk through directory tree.

        Yields:
            Tuple of (root_path, directories, files)
        """
        dirs = [d for d in dir_path.iterdir() if d.is_dir()]
        files = [f for f in dir_path.iterdir() if f.is_file()]
        yield dir_path, [d.name for d in dirs], [f.name for f in files]

        for d in dirs:
            yield from self._walk_directory(d)
