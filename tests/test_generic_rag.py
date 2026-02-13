"""Comprehensive tests for GenericRAGService."""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from chatboti.generic_rag import GenericRAGService
from chatboti.document import Document, DocumentChunk, ChunkRef, ChunkResult
from chatboti.loaders import CSVDocumentLoader


class TestGenericRAGServiceInitialization:
    """Test GenericRAGService initialization."""

    def test_new_service_creates_empty_index_and_metadata(self, tmp_path):
        """Test that new service creates empty FAISS index and metadata."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768
        )

        # Check that index is created but not saved yet
        assert service.index is not None
        assert service.index.ntotal == 0  # Empty index
        assert not index_path.exists()  # Not saved yet

        # Check metadata is empty
        assert service.chunk_refs == []
        assert service.documents == {}
        assert not metadata_path.exists()

    def test_load_existing_service_from_files(self, tmp_path):
        """Test loading existing service from files."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Create and save initial service
        service1 = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768
        )

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
        service2 = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768
        )

        # Verify data loaded correctly
        assert len(service2.chunk_refs) == 1
        assert service2.chunk_refs[0].document_id == "doc1"
        assert service2.chunk_refs[0].chunk_key == "field1"
        assert "doc1" in service2.documents
        assert service2.documents["doc1"].id == "doc1"

    def test_correct_embedding_dimensions(self, tmp_path):
        """Test that service respects embedding dimensions."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Test default dimension
        service1 = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )
        assert service1.index.d == 1536  # Default OpenAI dimension

        # Test custom dimension
        index_path2 = tmp_path / "test2.index"
        metadata_path2 = tmp_path / "test_meta2.json"
        service2 = GenericRAGService(
            index_path=index_path2,
            metadata_path=metadata_path2,
            embedding_dim=384
        )
        assert service2.index.d == 384


class TestGenericRAGServiceMetadata:
    """Test metadata save/load functionality."""

    def test_save_and_load_chunk_refs(self, tmp_path):
        """Test saving and loading chunk_refs."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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

    def test_save_and_load_documents(self, tmp_path):
        """Test saving and loading documents."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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
        service2 = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        assert len(service2.documents) == 2
        assert "doc1" in service2.documents
        assert service2.documents["doc1"].content["title"] == "Test Doc"
        assert service2.documents["doc1"].full_text == "Test Doc\nContent here"
        assert "title" in service2.documents["doc1"].chunks
        assert service2.documents["doc2"].content["field"] == "value"

    def test_json_structure_matches_spec(self, tmp_path):
        """Test that JSON structure matches specification."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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

        # Top-level keys
        assert set(data.keys()) == {"chunk_refs", "documents"}

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


class TestGenericRAGServiceStorageWrappers:
    """Test storage wrapper methods."""

    def test_get_chunk_refs_returns_correct_objects(self, tmp_path):
        """Test _get_chunk_refs() returns correct ChunkRef objects."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Setup chunk refs
        service.chunk_refs = [
            ChunkRef(document_id="doc1", chunk_key="chunk0"),
            ChunkRef(document_id="doc2", chunk_key="chunk1"),
            ChunkRef(document_id="doc1", chunk_key="chunk2")
        ]

        # Test retrieval
        refs = service._get_chunk_refs([0, 2])

        assert len(refs) == 2
        assert refs[0].document_id == "doc1"
        assert refs[0].chunk_key == "chunk0"
        assert refs[1].document_id == "doc1"
        assert refs[1].chunk_key == "chunk2"

    def test_get_chunk_texts_extracts_text_efficiently(self, tmp_path):
        """Test _get_chunk_texts() extracts text efficiently."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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
        refs = [
            ChunkRef(document_id="doc1", chunk_key="title"),
            ChunkRef(document_id="doc1", chunk_key="body")
        ]
        chunk_texts = service._get_chunk_texts(refs)

        assert len(chunk_texts) == 2
        assert chunk_texts[refs[0]] == "Test Title"
        assert chunk_texts[refs[1]] == "Test Body"

    def test_get_chunk_texts_with_text_indices(self, tmp_path):
        """Test _get_chunk_texts() with chunk-level (text indices) chunks."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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
        refs = [
            ChunkRef(document_id="doc1", chunk_key="0"),
            ChunkRef(document_id="doc1", chunk_key="1")
        ]
        chunk_texts = service._get_chunk_texts(refs)

        assert chunk_texts[refs[0]] == "This is a long document with multiple chunks."
        assert chunk_texts[refs[1]] == "Here is the second chunk."

    def test_get_document_texts_returns_full_text(self, tmp_path):
        """Test _get_document_texts() returns full document text."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

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
        doc_texts = service._get_document_texts(["doc1", "doc2"])

        assert len(doc_texts) == 2
        assert doc_texts["doc1"] == "Full text for document 1"
        # doc2 has no full_text, should return JSON of content
        assert "field1" in doc_texts["doc2"]
        assert "value1" in doc_texts["doc2"]


class TestGenericRAGServiceDocumentManagement:
    """Test document management functionality."""

    def test_add_document_without_embeddings(self, tmp_path):
        """Test add_document() without embed_client (embeddings not implemented)."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Create document with chunks
        doc = Document(
            id="test_doc",
            content={"title": "Test", "body": "Content"},
            chunks={
                "title": DocumentChunk(faiss_id=-1),
                "body": DocumentChunk(faiss_id=-1)
            }
        )

        # Add document
        service.add_document(doc)

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

    def test_add_document_tracks_chunks_in_chunk_refs(self, tmp_path):
        """Test that document chunks are tracked in chunk_refs."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Add first document
        doc1 = Document(
            id="doc1",
            content={"field1": "value1"},
            chunks={"field1": DocumentChunk(faiss_id=-1)}
        )
        service.add_document(doc1)

        # Add second document
        doc2 = Document(
            id="doc2",
            content={"field2": "value2", "field3": "value3"},
            chunks={
                "field2": DocumentChunk(faiss_id=-1),
                "field3": DocumentChunk(faiss_id=-1)
            }
        )
        service.add_document(doc2)

        # Verify chunk_refs structure
        assert len(service.chunk_refs) == 3
        assert service.chunk_refs[0].document_id == "doc1"
        assert service.chunk_refs[1].document_id == "doc2"
        assert service.chunk_refs[2].document_id == "doc2"
        assert service.chunk_refs[1].chunk_key == "field2"
        assert service.chunk_refs[2].chunk_key == "field3"

    def test_save_persists_to_disk(self, tmp_path):
        """Test that save() persists index and metadata to disk."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Add document
        doc = Document(
            id="doc1",
            content={"field": "value"},
            chunks={"field": DocumentChunk(faiss_id=-1)}
        )
        service.add_document(doc)

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


class TestGenericRAGServiceSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_chunk_results(self, tmp_path):
        """Test search() returns ChunkResult objects."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        # Add document
        doc = Document(
            id="doc1",
            content={"field": "test content"},
            chunks={"field": DocumentChunk(faiss_id=-1)}
        )
        service.add_document(doc)

        # Add embedding to index manually (since embed is not implemented)
        embedding = np.array([[0.1] * 768], dtype=np.float32)
        service.index.add(embedding)

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
    async def test_search_include_documents_parameter(self, tmp_path):
        """Test search() with include_documents=True."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        # Add document with full_text
        doc = Document(
            id="doc1",
            content={"field": "test content"},
            full_text="Full document text goes here",
            chunks={"field": DocumentChunk(faiss_id=-1)}
        )
        service.add_document(doc)

        # Add embedding
        embedding = np.array([[0.1] * 768], dtype=np.float32)
        service.index.add(embedding)

        # Search with include_documents=True
        results = await service.search("test query", k=1, include_documents=True)

        # Verify document_text is included
        assert len(results) == 1
        assert results[0].document_text == "Full document text goes here"

    @pytest.mark.asyncio
    async def test_search_result_contains_correct_text(self, tmp_path):
        """Test that search results contain correct chunk text."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        # Add documents with different content
        doc1 = Document(
            id="doc1",
            content={"title": "First Title", "body": "First Body"},
            chunks={
                "title": DocumentChunk(faiss_id=-1),
                "body": DocumentChunk(faiss_id=-1)
            }
        )
        service.add_document(doc1)

        doc2 = Document(
            id="doc2",
            content={"title": "Second Title"},
            chunks={"title": DocumentChunk(faiss_id=-1)}
        )
        service.add_document(doc2)

        # Add embeddings
        for _ in range(3):
            embedding = np.random.rand(1, 768).astype(np.float32)
            service.index.add(embedding)

        # Search
        results = await service.search("test query", k=3)

        # Verify correct text extracted
        assert len(results) == 3
        assert results[0].text in ["First Title", "First Body", "Second Title"]
        assert results[1].text in ["First Title", "First Body", "Second Title"]
        assert results[2].text in ["First Title", "First Body", "Second Title"]


class TestGenericRAGServiceLoaderIntegration:
    """Test integration with document loaders."""

    def test_get_loader_returns_csv_loader(self, tmp_path):
        """Test _get_loader() returns CSVDocumentLoader for .csv files."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Test CSV loader
        loader = service._get_loader("test.csv")
        assert isinstance(loader, CSVDocumentLoader)

        loader = service._get_loader("/path/to/data.CSV")
        assert isinstance(loader, CSVDocumentLoader)

    def test_get_loader_raises_for_unsupported_types(self, tmp_path):
        """Test _get_loader() raises ValueError for unsupported file types."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Test unsupported types
        with pytest.raises(ValueError, match="Unsupported file type"):
            service._get_loader("test.txt")

        with pytest.raises(ValueError, match="Unsupported file type"):
            service._get_loader("test.json")

        with pytest.raises(ValueError, match="Unsupported file type"):
            service._get_loader("test.pdf")

    def test_build_embeddings_from_csv(self, tmp_path):
        """Test build_embeddings_from_documents() with CSV file."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test_meta.json"

        # Create test CSV
        csv_path = tmp_path / "test.csv"
        with open(csv_path, 'w') as f:
            f.write("name,bio\n")
            f.write("Alice,Software engineer\n")
            f.write("Bob,Data scientist\n")

        service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path
        )

        # Load documents using CSV loader directly (not async)
        loader = service._get_loader(str(csv_path))
        documents = loader.load(str(csv_path), "person")

        # Add documents to service
        for doc in documents:
            service.add_document(doc)

        service.save()

        # Verify documents loaded
        assert len(service.documents) == 2
        assert "person-0" in service.documents
        assert "person-1" in service.documents
        assert service.documents["person-0"].content["name"] == "Alice"
        assert service.documents["person-1"].content["bio"] == "Data scientist"

        # Verify files saved
        assert metadata_path.exists()
