"""Comprehensive tests for FaissRAGService."""

import json
import pytest
import numpy as np
from pathlib import Path

from chatboti.faiss_rag import FaissRAGService
from chatboti.document import Document, DocumentChunk, ChunkRef, ChunkResult


class TestFaissRAGServiceInitialization:
    """Test FaissRAGService initialization."""

    @pytest.mark.asyncio
    async def test_new_service_creates_empty_index_and_metadata(self, tmp_path, embed_client_768):
        """Test that new service creates empty FAISS index and metadata."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Check that index is created but not saved yet
            assert service.index is not None
            assert service.index.ntotal == 0  # Empty index
            assert not index_path.exists()  # Not saved yet

            # Check metadata is empty
            assert service.chunk_refs == []
            assert service.documents == {}
            assert not metadata_path.exists()

    @pytest.mark.asyncio
    async def test_load_existing_service_from_files(self, tmp_path, embed_client_768):
        """Test loading existing service from files."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Create and save initial service
        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service1:
            # Add mock data
            doc = Document(
                id="doc1",
                content={"field1": "test content"},
                chunks={"field1": DocumentChunk(faiss_id=0)}
            )
            service1.documents["doc1"] = doc
            service1.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="field1"))
            service1.save()

        # Load service from saved files
        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service2:
            # Verify data loaded correctly
            assert len(service2.chunk_refs) == 1
            assert service2.chunk_refs[0].document_id == "doc1"
            assert service2.chunk_refs[0].chunk_key == "field1"
            assert "doc1" in service2.documents
            assert service2.documents["doc1"].id == "doc1"

    @pytest.mark.asyncio
    async def test_correct_embedding_dimensions(self, tmp_path, embed_client_768):
        """Test that service respects embedding dimensions from client."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Test with 768-dim embed client
        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service1:
            assert service1.index.d == 768
            assert service1.embedding_dim == 768

        # Test with custom 384-dim embed client
        from tests.conftest import DeterministicEmbedClient
        embed_client_384 = DeterministicEmbedClient(embedding_dim=384)

        index_path2 = tmp_path / "test2.index"
        metadata_path2 = tmp_path / "test_meta2.json"
        async with FaissRAGService(
            embed_client=embed_client_384,
            index_path=index_path2,
            metadata_path=metadata_path2
        ) as service2:
            assert service2.index.d == 384
            assert service2.embedding_dim == 384


class TestFaissRAGServiceMetadata:
    """Test metadata save/load functionality."""

    @pytest.mark.asyncio
    async def test_save_and_load_chunk_refs(self, tmp_path, embed_client_768):
        """Test saving and loading chunk_refs."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add chunk refs
            service.chunk_refs = [
                ChunkRef(document_id="doc1", chunk_key="chunk1"),
                ChunkRef(document_id="doc2", chunk_key="chunk2"),
                ChunkRef(document_id="doc1", chunk_key="chunk3")
            ]
            service.save()

        # Load and verify
        with open(metadata_path) as f:
            data = json.load(f)

        assert len(data['chunk_refs']) == 3
        assert data['chunk_refs'][0]['document_id'] == "doc1"
        assert data['chunk_refs'][0]['chunk_key'] == "chunk1"
        assert data['chunk_refs'][1]['document_id'] == "doc2"
        assert data['chunk_refs'][2]['chunk_key'] == "chunk3"

    @pytest.mark.asyncio
    async def test_save_and_load_documents(self, tmp_path, embed_client_768):
        """Test saving and loading documents."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add documents
            doc1 = Document(
                id="doc1",
                content={"title": "Test Doc", "body": "Content here"},
                full_text="Test Doc\nContent here",
                metadata={"source": "test.txt"},
                chunks={
                    "title": DocumentChunk(faiss_id=0),
                    "body": DocumentChunk(faiss_id=1)
                }
            )
            doc2 = Document(
                id="doc2",
                content={"field": "value"},
                chunks={"field": DocumentChunk(faiss_id=2)}
            )

            service.documents = {"doc1": doc1, "doc2": doc2}
            service.save()

        # Load and verify
        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service2:
            assert len(service2.documents) == 2
            assert "doc1" in service2.documents
            assert service2.documents["doc1"].content["title"] == "Test Doc"
            assert service2.documents["doc1"].full_text == "Test Doc\nContent here"
            assert "title" in service2.documents["doc1"].chunks
            assert service2.documents["doc2"].content["field"] == "value"

    @pytest.mark.asyncio
    async def test_json_structure_matches_spec(self, tmp_path, embed_client_768):
        """Test that JSON structure matches specification."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add test data
            doc = Document(
                id="test_doc",
                content={"field": "value"},
                chunks={"field": DocumentChunk(faiss_id=0)}
            )
            service.documents["test_doc"] = doc
            service.chunk_refs.append(ChunkRef(document_id="test_doc", chunk_key="field"))
            service.save()

        # Verify JSON structure
        with open(metadata_path) as f:
            data = json.load(f)

        # Top-level keys (now includes model_name and embedding_dim)
        assert "chunk_refs" in data
        assert "documents" in data
        assert "model_name" in data
        assert "embedding_dim" in data

        # chunk_refs structure
        assert isinstance(data['chunk_refs'], list)
        assert all('document_id' in ref and 'chunk_key' in ref for ref in data['chunk_refs'])

        # documents structure
        assert isinstance(data['documents'], list)
        doc_data = data['documents'][0]
        assert 'id' in doc_data
        assert 'content' in doc_data
        assert 'full_text' in doc_data
        assert 'metadata' in doc_data
        assert 'chunks' in doc_data


class TestFaissRAGServiceStorageWrappers:
    """Test storage wrapper methods."""

    @pytest.mark.asyncio
    async def test_get_chunk_refs_returns_correct_objects(self, tmp_path, embed_client_768):
        """Test get_chunk_refs() returns correct ChunkRef objects."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Setup chunk refs
            service.chunk_refs = [
                ChunkRef(document_id="doc1", chunk_key="chunk0"),
                ChunkRef(document_id="doc2", chunk_key="chunk1"),
                ChunkRef(document_id="doc1", chunk_key="chunk2")
            ]

            # Test retrieval
            refs = service.get_chunk_refs([0, 2])

            assert len(refs) == 2
            assert refs[0].document_id == "doc1"
            assert refs[0].chunk_key == "chunk0"
            assert refs[1].document_id == "doc1"
            assert refs[1].chunk_key == "chunk2"

    @pytest.mark.asyncio
    async def test_get_chunk_text_extracts_text_efficiently(self, tmp_path, embed_client_768):
        """Test get_chunk_text() extracts text efficiently."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Setup documents with field-level chunks
            doc1 = Document(
                id="doc1",
                content={"title": "Test Title", "body": "Test Body"},
                chunks={
                    "title": DocumentChunk(faiss_id=0),
                    "body": DocumentChunk(faiss_id=1)
                }
            )
            service.documents["doc1"] = doc1

            # Test text extraction
            ref_title = ChunkRef(document_id="doc1", chunk_key="title")
            ref_body = ChunkRef(document_id="doc1", chunk_key="body")

            assert service.get_chunk_text(ref_title) == "Test Title"
            assert service.get_chunk_text(ref_body) == "Test Body"

    @pytest.mark.asyncio
    async def test_get_chunk_text_with_text_indices(self, tmp_path, embed_client_768):
        """Test get_chunk_text() with chunk-level (text indices) chunks."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Setup document with chunk-level chunks
            full_text = "This is a long document with multiple chunks. Here is the second chunk."
            doc = Document(
                id="doc1",
                full_text=full_text,
                chunks={
                    "0": DocumentChunk(faiss_id=0, i_start=0, i_end=45),
                    "1": DocumentChunk(faiss_id=1, i_start=46, i_end=71)
                }
            )
            service.documents["doc1"] = doc

            # Test text extraction
            ref_0 = ChunkRef(document_id="doc1", chunk_key="0")
            ref_1 = ChunkRef(document_id="doc1", chunk_key="1")

            assert service.get_chunk_text(ref_0) == "This is a long document with multiple chunks."
            assert service.get_chunk_text(ref_1) == "Here is the second chunk."

    @pytest.mark.asyncio
    async def test_get_document_texts_returns_full_text(self, tmp_path, embed_client_768):
        """Test get_document_texts() returns full document text."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Setup documents
            doc1 = Document(
                id="doc1",
                full_text="Full text for document 1"
            )
            doc2 = Document(
                id="doc2",
                content={"field1": "value1", "field2": "value2"}
            )
            service.documents = {"doc1": doc1, "doc2": doc2}

            # Test retrieval
            doc_texts = service.get_document_texts(["doc1", "doc2"])

            assert len(doc_texts) == 2
            assert doc_texts["doc1"] == "Full text for document 1"
            # doc2 has no full_text, should return JSON of content
            assert "field1" in doc_texts["doc2"]
            assert "value1" in doc_texts["doc2"]


class TestFaissRAGServiceDocumentManagement:
    """Test document management functionality."""

    @pytest.mark.asyncio
    async def test_add_document_with_embeddings(self, tmp_path, embed_client_768):
        """Test add_document() with embed_client generates embeddings."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Create document with chunks
            doc = Document(
                id="test_doc",
                content={"title": "Test", "body": "Content"},
                chunks={
                    "title": DocumentChunk(faiss_id=-1),
                    "body": DocumentChunk(faiss_id=-1)
                }
            )

            # Track calls before adding
            initial_call_count = embed_client_768.call_count

            # Add document
            await service.add_document(doc)

            # Verify embed_client.embed() was called for each chunk
            assert embed_client_768.call_count == initial_call_count + 2
            assert "Test" in embed_client_768.embedded_texts
            assert "Content" in embed_client_768.embedded_texts

            # Verify document stored
            assert "test_doc" in service.documents
            assert service.documents["test_doc"].id == "test_doc"

            # Verify chunk refs created
            assert len(service.chunk_refs) == 2
            assert service.chunk_refs[0].document_id == "test_doc"
            assert service.chunk_refs[0].chunk_key == "title"
            assert service.chunk_refs[1].document_id == "test_doc"
            assert service.chunk_refs[1].chunk_key == "body"

            # Verify faiss_ids assigned
            assert doc.chunks["title"].faiss_id == 0
            assert doc.chunks["body"].faiss_id == 1

            # Verify embeddings added to FAISS index
            assert service.index.ntotal == 2

    @pytest.mark.asyncio
    async def test_add_document_tracks_chunks_in_chunk_refs(self, tmp_path, embed_client_768):
        """Test that document chunks are tracked in chunk_refs."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add first document
            doc1 = Document(
                id="doc1",
                content={"field1": "value1"},
                chunks={"field1": DocumentChunk(faiss_id=-1)}
            )
            await service.add_document(doc1)

            # Add second document
            doc2 = Document(
                id="doc2",
                content={"field2": "value2", "field3": "value3"},
                chunks={
                    "field2": DocumentChunk(faiss_id=-1),
                    "field3": DocumentChunk(faiss_id=-1)
                }
            )
            await service.add_document(doc2)

            # Verify chunk_refs structure
            assert len(service.chunk_refs) == 3
            assert service.chunk_refs[0].document_id == "doc1"
            assert service.chunk_refs[1].document_id == "doc2"
            assert service.chunk_refs[2].document_id == "doc2"
            assert service.chunk_refs[1].chunk_key == "field2"
            assert service.chunk_refs[2].chunk_key == "field3"

    @pytest.mark.asyncio
    async def test_save_persists_to_disk(self, tmp_path, embed_client_768):
        """Test that save() persists index and metadata to disk."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add document
            doc = Document(
                id="doc1",
                content={"field": "value"},
                chunks={"field": DocumentChunk(faiss_id=-1)}
            )
            await service.add_document(doc)

            # Save
            service.save()

        # Verify files exist
        assert index_path.exists()
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            data = json.load(f)

        assert len(data['chunk_refs']) == 1
        assert len(data['documents']) == 1
        assert data['documents'][0]['id'] == "doc1"


class TestFaissRAGServiceSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_chunk_results(self, tmp_path, embed_client_768):
        """Test search() returns ChunkResult objects."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add document
            doc = Document(
                id="doc1",
                content={"field": "test content"},
                chunks={"field": DocumentChunk(faiss_id=-1)}
            )
            await service.add_document(doc)

            # Search
            results = await service.search("test query", k=1)

            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], ChunkResult)
            assert results[0].document_id == "doc1"
            assert results[0].chunk_key == "field"
            assert results[0].text == "test content"
            assert results[0].document_text is None

    @pytest.mark.asyncio
    async def test_search_include_documents_parameter(self, tmp_path, embed_client_768):
        """Test search() with include_documents=True."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add document with full_text
            doc = Document(
                id="doc1",
                content={"field": "test content"},
                full_text="Full document text goes here",
                chunks={"field": DocumentChunk(faiss_id=-1)}
            )
            await service.add_document(doc)

            # Search with include_documents=True
            results = await service.search("test query", k=1, include_documents=True)

            # Verify document_text is included
            assert len(results) == 1
            assert results[0].document_text == "Full document text goes here"

    @pytest.mark.asyncio
    async def test_search_result_contains_correct_text(self, tmp_path, embed_client_768):
        """Test that search results contain correct chunk text."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Add documents with different content
            doc1 = Document(
                id="doc1",
                content={"title": "First Title", "body": "First Body"},
                chunks={
                    "title": DocumentChunk(faiss_id=-1),
                    "body": DocumentChunk(faiss_id=-1)
                }
            )
            await service.add_document(doc1)

            doc2 = Document(
                id="doc2",
                content={"title": "Second Title"},
                chunks={"title": DocumentChunk(faiss_id=-1)}
            )
            await service.add_document(doc2)

            # Search
            results = await service.search("test query", k=3)

            # Verify correct text extracted
            assert len(results) == 3
            assert results[0].text in ["First Title", "First Body", "Second Title"]
            assert results[1].text in ["First Title", "First Body", "Second Title"]
            assert results[2].text in ["First Title", "First Body", "Second Title"]


class TestFaissRAGServiceLoaderIntegration:
    """Test integration with document loaders."""

    @pytest.mark.asyncio
    async def test_build_embeddings_from_csv(self, tmp_path, embed_client_768):
        """Test build_embeddings_from_documents() with CSV file."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Create test CSV
        csv_path = tmp_path / "test.csv"
        with open(csv_path, 'w') as f:
            f.write("name,bio\n")
            f.write("Alice,Software engineer\n")
            f.write("Bob,Data scientist\n")

        async with FaissRAGService(
            embed_client=embed_client_768,
            index_path=index_path,
            metadata_path=metadata_path
        ) as service:
            # Load documents using CSV loader
            from chatboti.loaders import load_csv
            documents = await load_csv(str(csv_path))

            # Add documents to service
            for doc in documents:
                await service.add_document(doc)

            service.save()

            # Verify documents loaded
            assert len(service.documents) == 2
            assert "test-0" in service.documents
            assert "test-1" in service.documents
            assert service.documents["test-0"].content["name"] == "Alice"
            assert service.documents["test-1"].content["bio"] == "Data scientist"

            # Verify files saved
            assert metadata_path.exists()
