"""Vector search and document chunking using ChromaDB."""

import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
import requests

from src.models import ClusterData


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
