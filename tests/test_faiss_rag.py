"""Tests for FaissRAGService.

Covers: add_document + search, i_vector_start/i_vector_end, DocumentChunk.id,
include_documents, multi-document search, and save/reload round-trip.
"""

import pytest

from chatboti.document import ChunkResult, Document, DocumentChunk
from chatboti.faiss_rag import FaissRAGService
from tests.conftest import DeterministicEmbedClient


def _make_doc(doc_id: str, fields: dict) -> Document:
    """Build a field-chunked Document from a dict of field_name -> text.

    :param doc_id: Document identifier
    :param fields: Mapping of field name to text content
    :return: Document with one DocumentChunk per field
    """
    return Document(
        id=doc_id,
        content=fields,
        chunks={key: DocumentChunk(id=-1) for key in fields},
    )


class TestFaissRAGServiceAddAndSearch:
    """End-to-end add_document + search tests."""

    @pytest.mark.asyncio
    async def test_add_document_and_search_returns_results(self, tmp_path):
        """add_document followed by search returns at least one ChunkResult."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc = _make_doc("doc1", {"bio": "machine learning researcher"})
            await service.add_document(doc)

            results = await service.search("machine learning", k=1)

        assert len(results) == 1
        assert isinstance(results[0], ChunkResult)
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "bio"
        assert results[0].text == "machine learning researcher"

    @pytest.mark.asyncio
    async def test_i_vector_start_and_end_set_correctly(self, tmp_path):
        """add_document sets i_vector_start / i_vector_end on the Document."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc1 = _make_doc("doc1", {"a": "alpha", "b": "beta"})
            doc2 = _make_doc("doc2", {"c": "gamma"})
            await service.add_document(doc1)
            await service.add_document(doc2)

            assert doc1.i_vector_start == 0
            assert doc2.i_vector_start == 2

    @pytest.mark.asyncio
    async def test_chunk_id_assigned_on_add(self, tmp_path):
        """DocumentChunk.id is updated to its vector index after add_document."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc = _make_doc("doc1", {"x": "text x", "y": "text y"})
            await service.add_document(doc)

            assert doc.chunks["x"].id == 0
            assert doc.chunks["y"].id == 1

    @pytest.mark.asyncio
    async def test_search_include_documents_returns_content(self, tmp_path):
        """search() with include_documents=True populates result.content."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc = _make_doc("doc1", {"title": "AI talk", "abstract": "Deep learning"})
            await service.add_document(doc)

            results = await service.search("deep learning", k=1, include_documents=True)

        assert len(results) >= 1
        assert results[0].content is not None
        assert "title" in results[0].content or "abstract" in results[0].content

    @pytest.mark.asyncio
    async def test_search_multiple_documents(self, tmp_path):
        """search returns results across multiple documents."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            for i in range(3):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)

            results = await service.search("content", k=3)

        assert len(results) == 3
        doc_ids = {r.document_id for r in results}
        assert doc_ids == {"doc0", "doc1", "doc2"}


class TestFaissRAGServiceSaveReload:
    """Tests for save/reload round-trip persistence."""

    @pytest.mark.asyncio
    async def test_save_creates_files(self, tmp_path):
        """save() writes index and metadata files to disk."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc = _make_doc("doc1", {"field": "hello world"})
            await service.add_document(doc)
            service.save()

        assert index_path.exists()
        assert metadata_path.exists()

    @pytest.mark.asyncio
    async def test_reload_recovers_search_results(self, tmp_path):
        """Data added before save() is searchable in a new service instance."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            doc = _make_doc("doc1", {"field": "test content"})
            await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with FaissRAGService(
            embed_client=embed_client2,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service2:
            results = await service2.search("test query", k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "field"
        assert results[0].text == "test content"

    @pytest.mark.asyncio
    async def test_reload_preserves_document_count(self, tmp_path):
        """Multiple documents survive a save/reload cycle."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            for i in range(5):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with FaissRAGService(
            embed_client=embed_client2,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service2:
            results = await service2.search("content", k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_reload_preserves_model_name(self, tmp_path):
        """model_name stored via save() is recovered on reload."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "my-custom-model"

        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service:
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with FaissRAGService(
            embed_client=embed_client2,
            index_path=index_path,
            metadata_path=metadata_path,
        ) as service2:
            assert service2.model_name == "my-custom-model"
