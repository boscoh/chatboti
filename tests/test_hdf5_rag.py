"""Tests for HDF5RAGService.

Covers: add_document + search, i_vector_start/i_vector_end, DocumentChunk.id,
include_documents, multi-document search, doc_ids filtered search, and
save/reload round-trip.
"""

import pytest

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from chatboti.document import ChunkResult, Document, DocumentChunk
from chatboti.hdf5_rag import HDF5RAGService
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


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceAddAndSearch:
    """End-to-end add_document + search tests."""

    @pytest.mark.asyncio
    async def test_add_document_and_search_returns_results(self, tmp_path):
        """add_document followed by search returns at least one ChunkResult."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
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
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
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
        """DocumentChunk.id is updated to its vector index after add_document."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            doc = _make_doc("doc1", {"x": "text x", "y": "text y"})
            await service.add_document(doc)

            assert doc.chunks["x"].id == 0
            assert doc.chunks["y"].id == 1

    @pytest.mark.asyncio
    async def test_search_include_documents_returns_content(self, tmp_path):
        """search() with include_documents=True populates result.content."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
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
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            for i in range(3):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)

            results = await service.search("content", k=3)

        assert len(results) == 3
        doc_ids = {r.document_id for r in results}
        assert doc_ids == {"doc0", "doc1", "doc2"}


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceDocIdsFilter:
    """Tests for doc_ids-restricted search."""

    @pytest.mark.asyncio
    async def test_doc_ids_restricts_results_to_selected_docs(self, tmp_path):
        """search with doc_ids only returns chunks from those documents."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
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
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
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
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            doc = _make_doc("doc1", {"field": "some content"})
            await service.add_document(doc)

            results = await service.search("query", k=5, doc_ids=[])

        assert results == []

    @pytest.mark.asyncio
    async def test_doc_ids_unknown_id_returns_no_results(self, tmp_path):
        """search with a non-existent doc_id returns an empty list."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            doc = _make_doc("doc1", {"field": "content"})
            await service.add_document(doc)

            results = await service.search("query", k=5, doc_ids=["nonexistent"])

        assert results == []


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceSaveReload:
    """Tests for save/reload round-trip persistence."""

    @pytest.mark.asyncio
    async def test_save_creates_file(self, tmp_path):
        """save() writes HDF5 file to disk."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            doc = _make_doc("doc1", {"field": "hello world"})
            await service.add_document(doc)
            service.save()

        assert hdf5_path.exists()

    @pytest.mark.asyncio
    async def test_reload_recovers_search_results(self, tmp_path):
        """Data added before save() is searchable in a new service instance."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            doc = _make_doc("doc1", {"field": "test content"})
            await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with HDF5RAGService(
            embed_client=embed_client2, hdf5_path=hdf5_path
        ) as service2:
            results = await service2.search("test query", k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "field"
        assert results[0].text == "test content"

    @pytest.mark.asyncio
    async def test_reload_preserves_document_count(self, tmp_path):
        """Multiple documents survive a save/reload cycle."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            for i in range(5):
                doc = _make_doc(f"doc{i}", {"field": f"content {i}"})
                await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with HDF5RAGService(
            embed_client=embed_client2, hdf5_path=hdf5_path
        ) as service2:
            results = await service2.search("content", k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_reload_preserves_model_name(self, tmp_path):
        """model_name stored via save() is recovered on reload."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "my-custom-model"

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with HDF5RAGService(
            embed_client=embed_client2, hdf5_path=hdf5_path
        ) as service2:
            assert service2.model_name == "my-custom-model"

    @pytest.mark.asyncio
    async def test_save_reload_doc_ids_filter_still_works(self, tmp_path):
        """doc_ids filter is correct after a save/reload round-trip."""
        hdf5_path = tmp_path / "test.h5"
        embed_client = DeterministicEmbedClient(embedding_dim=64)

        async with HDF5RAGService(
            embed_client=embed_client, hdf5_path=hdf5_path
        ) as service:
            for i in range(3):
                doc = _make_doc(f"doc{i}", {"field": f"value {i}"})
                await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        async with HDF5RAGService(
            embed_client=embed_client2, hdf5_path=hdf5_path
        ) as service2:
            results = await service2.search("value", k=5, doc_ids=["doc0", "doc2"])

        result_doc_ids = {r.document_id for r in results}
        assert result_doc_ids == {"doc0", "doc2"}
        assert "doc1" not in result_doc_ids


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceDataDir:
    """Tests for data_dir-based path resolution (mirrors CLI usage)."""

    @pytest.mark.asyncio
    async def test_data_dir_creates_expected_file(self, tmp_path):
        """Using data_dir creates a model-named .h5 file in that directory."""
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "test-model"

        async with HDF5RAGService(
            embed_client=embed_client, data_dir=tmp_path
        ) as service:
            doc = _make_doc("doc1", {"field": "hello world"})
            await service.add_document(doc)
            service.save()

        h5_files = list(tmp_path.glob("*.h5"))
        assert len(h5_files) == 1, "Expected exactly one .h5 file in data_dir"
        assert h5_files[0].name == "embeddings-test-model.h5"

    @pytest.mark.asyncio
    async def test_data_dir_persists_across_open_close(self, tmp_path):
        """Data written via data_dir is searchable in a second instance using the same data_dir."""
        embed_client = DeterministicEmbedClient(embedding_dim=64)
        embed_client.model = "test-model"

        async with HDF5RAGService(
            embed_client=embed_client, data_dir=tmp_path
        ) as service:
            doc = _make_doc("doc1", {"field": "data dir content"})
            await service.add_document(doc)
            service.save()

        embed_client2 = DeterministicEmbedClient(embedding_dim=64)
        embed_client2.model = "test-model"
        async with HDF5RAGService(
            embed_client=embed_client2, data_dir=tmp_path
        ) as service2:
            results = await service2.search("data dir", k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "field"
        assert results[0].text == "data dir content"
