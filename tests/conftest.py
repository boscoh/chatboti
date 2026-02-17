"""Shared test utilities and fixtures for integration testing."""

from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np
import pytest
from chatboti.llm import SimpleLLMClient, get_llm_client

from chatboti.document import ChunkRef, Document, DocumentChunk
from chatboti.faiss_rag import FaissRAGService
from chatboti.hdf5_rag import HDF5RAGService


class DeterministicEmbedClient:
    """Deterministic embed client for integration testing.

    Returns consistent embeddings based on text hash to avoid external dependencies
    while maintaining realistic behavior.
    """

    def __init__(self, embedding_dim: int = 384):
        self.model = "deterministic-test-embed"
        self.embedding_dim = embedding_dim
        self.call_count = 0
        self.embedded_texts = []

    async def connect(self) -> None:
        """Initialize connection."""
        pass

    async def close(self) -> None:
        """Close connection."""
        pass

    async def embed(self, text: str) -> List[float]:
        """Generate deterministic embedding based on text hash."""
        self.call_count += 1
        self.embedded_texts.append(text)

        # Create deterministic embedding from text hash
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]


@pytest.fixture
def embed_client():
    """Fixture providing a deterministic embed client."""
    return DeterministicEmbedClient()


@pytest.fixture
def embed_client_768():
    """Fixture providing a deterministic embed client with 768 dimensions."""
    return DeterministicEmbedClient(embedding_dim=768)


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        Document(
            id="doc1",
            content={"title": "First Document", "body": "Content of first document"},
            full_text="First Document\nContent of first document",
            chunks={
                "title": DocumentChunk(faiss_id=-1),
                "body": DocumentChunk(faiss_id=-1),
            },
        ),
        Document(
            id="doc2",
            content={"title": "Second Document", "summary": "Brief summary"},
            chunks={
                "title": DocumentChunk(faiss_id=-1),
                "summary": DocumentChunk(faiss_id=-1),
            },
        ),
        Document(
            id="doc3",
            content={"name": "Alice", "bio": "Software engineer with AI expertise"},
            chunks={
                "name": DocumentChunk(faiss_id=-1),
                "bio": DocumentChunk(faiss_id=-1),
            },
        ),
    ]


@pytest.fixture
async def faiss_service(tmp_path, embed_client_768):
    """Fixture providing a FaissRAGService with sample data."""
    service = FaissRAGService(
        index_path=tmp_path / "test.index",
        metadata_path=tmp_path / "test_meta.json",
        embed_client=embed_client_768,
    )

    # Initialize the service
    await service.__aenter__()

    # Add sample documents
    doc1 = Document(
        id="doc1",
        content={"title": "AI Research", "field": "machine learning"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc1)

    doc2 = Document(
        id="doc2",
        content={"title": "Software Engineering", "field": "distributed systems"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc2)

    yield service

    # Cleanup
    service.save()
    await service.__aexit__(None, None, None)


@pytest.fixture
async def hdf5_service(tmp_path, embed_client_768):
    """Fixture providing an HDF5RAGService with sample data."""
    service = HDF5RAGService(
        hdf5_path=tmp_path / "test.h5", embed_client=embed_client_768
    )

    # Initialize the service
    await service.__aenter__()

    # Add sample documents
    doc1 = Document(
        id="doc1",
        content={"title": "AI Research", "field": "machine learning"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc1)

    doc2 = Document(
        id="doc2",
        content={"title": "Software Engineering", "field": "distributed systems"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc2)

    yield service

    # Cleanup
    service.save()
    await service.__aexit__(None, None, None)


@pytest.fixture(params=["faiss", "hdf5"])
async def rag_service(request, tmp_path, embed_client_768):
    """Parametrized fixture that yields both FaissRAGService and HDF5RAGService."""
    backend = request.param

    if backend == "faiss":
        service = FaissRAGService(
            index_path=tmp_path / "test.index",
            metadata_path=tmp_path / "test_meta.json",
            embed_client=embed_client_768,
        )
    else:  # hdf5
        service = HDF5RAGService(
            hdf5_path=tmp_path / "test.h5", embed_client=embed_client_768
        )

    # Initialize the service
    await service.__aenter__()

    # Add sample documents
    doc1 = Document(
        id="doc1",
        content={"title": "AI Research", "field": "machine learning"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc1)

    doc2 = Document(
        id="doc2",
        content={"title": "Software Engineering", "field": "distributed systems"},
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "field": DocumentChunk(faiss_id=-1),
        },
    )
    await service.add_document(doc2)

    yield service

    # Cleanup
    service.save()
    await service.__aexit__(None, None, None)


@pytest.fixture
def csv_file(tmp_path):
    """Fixture creating a sample CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    csv_content = """name,title,bio,abstract
Alice,Professor,Expert in quantum computing,Research on quantum algorithms
Bob,Researcher,Machine learning specialist,Deep learning applications
Charlie,Engineer,Software architecture expert,Cloud computing solutions
"""
    csv_path.write_text(csv_content)
    return csv_path


def assert_search_results_valid(results, expected_count=None):
    """Helper to validate search results."""
    assert isinstance(results, list)

    if expected_count is not None:
        assert len(results) == expected_count

    for result in results:
        assert hasattr(result, "document_id")
        assert hasattr(result, "chunk_key")
        assert hasattr(result, "text")
        assert result.document_id
        assert result.chunk_key
        assert result.text


def assert_documents_persisted(service, expected_doc_count):
    """Helper to verify documents are properly persisted."""
    assert len(service.documents) == expected_doc_count
    assert len(service.chunk_refs) >= expected_doc_count  # At least one chunk per doc

    # Verify all documents can be retrieved
    for doc_id in service.documents:
        doc = service.documents[doc_id]
        assert doc.id == doc_id
        assert hasattr(doc, "content")
        assert hasattr(doc, "chunks")
