"""Async tests for SQLiteRAGService."""

import pytest

from chatboti.document import ChunkResult, Document, DocumentChunk
from chatboti.sqlite_rag import SQLiteRAGService
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


class TestSQLiteRAGServiceAddAndSearch:
    """End-to-end add_document + search tests."""

    @pytest.mark.asyncio
    async def test_add_document_and_search_returns_results(self, tmp_path):
        """add_document followed by search returns at least one ChunkResult."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
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
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc1 = _make_doc("doc1", {"a": "alpha", "b": "beta"})
            doc2 = _make_doc("doc2", {"c": "gamma"})
            await service.add_document(doc1)
            await service.add_document(doc2)

            assert doc1.i_vector_start == 0
            assert doc1.i_vector_end == 2
            assert doc2.i_vector_start == 2
            assert doc2.i_vector_end == 3

    @pytest.mark.asyncio
    async def test_chunk_id_assigned_on_add(self, tmp_path):
        """DocumentChunk.id is updated to its vector rowid after add_document."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc = _make_doc("doc1", {"x": "text x", "y": "text y"})
            await service.add_document(doc)

            assert doc.chunks["x"].id == 0
            assert doc.chunks["y"].id == 1

    @pytest.mark.asyncio
    async def test_search_include_documents_returns_content(self, tmp_path):
        """search() with include_documents=True populates result.content."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
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
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            for i in range(3):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)

            results = await service.search("content", k=3)

        assert len(results) == 3
        doc_ids = {r.document_id for r in results}
        assert doc_ids == {"doc0", "doc1", "doc2"}


class TestSQLiteRAGServiceDocIdsFilter:
    """Tests for doc_ids-restricted search."""

    @pytest.mark.asyncio
    async def test_doc_ids_restricts_results_to_selected_docs(self, tmp_path):
        """search with doc_ids only returns chunks from those documents."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc1 = _make_doc("doc1", {"field": "alpha content"})
            doc2 = _make_doc("doc2", {"field": "beta content"})
            await service.add_document(doc1)
            await service.add_document(doc2)

            results = await service.search("query", k=5, doc_ids=["doc1"])

        assert len(results) >= 1
        assert all(r.document_id == "doc1" for r in results)

    @pytest.mark.asyncio
    async def test_doc_ids_excludes_unrequested_docs(self, tmp_path):
        """search with doc_ids does not return chunks from excluded documents."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            for i in range(4):
                doc = _make_doc(f"doc{i}", {"field": f"text {i}"})
                await service.add_document(doc)

            results = await service.search("text", k=10, doc_ids=["doc0", "doc3"])

        result_doc_ids = {r.document_id for r in results}
        assert result_doc_ids == {"doc0", "doc3"}
        assert "doc1" not in result_doc_ids
        assert "doc2" not in result_doc_ids

    @pytest.mark.asyncio
    async def test_doc_ids_empty_list_returns_no_results(self, tmp_path):
        """search with empty doc_ids returns an empty list."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc = _make_doc("doc1", {"field": "some content"})
            await service.add_document(doc)

            results = await service.search("query", k=5, doc_ids=[])

        assert results == []

    @pytest.mark.asyncio
    async def test_doc_ids_unknown_id_returns_no_results(self, tmp_path):
        """search with a non-existent doc_id returns an empty list."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc = _make_doc("doc1", {"field": "content"})
            await service.add_document(doc)

            results = await service.search("query", k=5, doc_ids=["nonexistent"])

        assert results == []


class TestSQLiteRAGServiceSaveReload:
    """Tests for save/reload round-trip persistence."""

    @pytest.mark.asyncio
    async def test_save_persists_metadata(self, tmp_path):
        """save() writes model_name and embedding_dim to rag_metadata."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc = _make_doc("doc1", {"field": "hello world"})
            await service.add_document(doc)
            service.save()

        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_reload_recovers_search_results(self, tmp_path):
        """Data added before save() is searchable in a new service instance."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            doc = _make_doc("doc1", {"field": "test content"})
            await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with SQLiteRAGService(
            embed_client=embed_client2, db_path=db_path
        ) as service2:
            results = await service2.search("test query", k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "field"
        assert results[0].text == "test content"

    @pytest.mark.asyncio
    async def test_reload_preserves_document_count(self, tmp_path):
        """Multiple documents survive a save/reload cycle."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            for i in range(5):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with SQLiteRAGService(
            embed_client=embed_client2, db_path=db_path
        ) as service2:
            results = await service2.search("content", k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_reload_preserves_model_name(self, tmp_path):
        """model_name stored via save() is recovered on reload."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "my-custom-model"

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with SQLiteRAGService(
            embed_client=embed_client2, db_path=db_path
        ) as service2:
            assert service2.model_name == "my-custom-model"

    @pytest.mark.asyncio
    async def test_auto_db_path_persists_to_file(self, tmp_path):
        """When db_path is omitted, data_dir is used to resolve the path and data persists."""
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "test-model"

        async with SQLiteRAGService(
            embed_client=embed_client, data_dir=tmp_path
        ) as service:
            doc = _make_doc("doc1", {"field": "auto path content"})
            await service.add_document(doc)
            service.save()
            db_path = service.db_path

        assert db_path.exists(), "DB file must be created on disk, not in :memory:"

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        embed_client2.model = "test-model"
        async with SQLiteRAGService(
            embed_client=embed_client2, data_dir=tmp_path
        ) as service2:
            results = await service2.search("auto path", k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_save_reload_doc_ids_filter_still_works(self, tmp_path):
        """doc_ids filter is correct after a save/reload round-trip."""
        db_path = tmp_path / "test.db"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with SQLiteRAGService(
            embed_client=embed_client, db_path=db_path
        ) as service:
            for i in range(3):
                doc = _make_doc(f"doc{i}", {"field": f"value {i}"})
                await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with SQLiteRAGService(
            embed_client=embed_client2, db_path=db_path
        ) as service2:
            results = await service2.search("value", k=5, doc_ids=["doc0", "doc2"])

        result_doc_ids = {r.document_id for r in results}
        assert result_doc_ids == {"doc0", "doc2"}
        assert "doc1" not in result_doc_ids
