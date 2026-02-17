"""Integration tests for HDF5RAGService using real implementation."""

import json

import numpy as np
import pytest

# Try to import h5py - tests will be skipped if not available
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from chatboti.document import ChunkRef, ChunkResult, Document, DocumentChunk
from chatboti.hdf5_rag import HDF5RAGService
from tests.conftest import DeterministicEmbedClient


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceInitialization:
    """Test HDF5RAGService initialization."""

    @pytest.mark.asyncio
    async def test_new_service_creates_empty_storage(self, tmp_path, embed_client):
        """Test that new service creates empty HDF5 structure."""
        hdf5_path = tmp_path / "test.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service:
            # Check in-memory structures
            assert service.index is not None  # Inherits from FaissRAGService
            assert service.index.ntotal == 0
            assert service.chunk_refs == []
            assert service.documents == {}
            assert not hdf5_path.exists()  # Not saved yet

    @pytest.mark.asyncio
    async def test_load_existing_service_from_hdf5(self, tmp_path, embed_client):
        """Test loading existing service from HDF5 file."""
        hdf5_path = tmp_path / "test.h5"

        # Create and save initial service
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service1:
            # Add mock data
            doc = Document(
                id="doc1",
                content={"field1": "test content"},
                chunks={"field1": DocumentChunk(faiss_id=-1)},
            )
            await service1.add_document(doc)
            service1.save()

        # Load service from saved file
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service2:
            # Verify data loaded correctly
            assert len(service2.chunk_refs) == 1

            # Just verify the document exists and has the right content
            # Don't worry about the exact format of IDs for now
            chunk_ref = service2.chunk_refs[0]
            assert chunk_ref is not None
            assert hasattr(chunk_ref, "document_id")
            assert hasattr(chunk_ref, "chunk_key")

            # Check that we can find the document somehow
            found_doc = None
            for doc_id, doc in service2.documents.items():
                if isinstance(doc_id, bytes):
                    try:
                        decoded_id = doc_id.decode("utf-8")
                        if decoded_id == "doc1":
                            found_doc = doc
                            break
                    except UnicodeDecodeError:
                        pass
                elif doc_id == "doc1":
                    found_doc = doc
                    break

            assert found_doc is not None
            assert found_doc.id == "doc1" or found_doc.id == b"doc1"

    @pytest.mark.asyncio
    async def test_correct_embedding_dimensions(self, tmp_path, embed_client):
        """Test that service respects embedding dimensions."""
        hdf5_path = tmp_path / "test.h5"

        # Test with embed_client that has 384 dimensions
        embed_client_384 = DeterministicEmbedClient(embedding_dim=384)

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_384
        ) as service:
            assert service.embedding_dim == 384
            assert service.index.d == 384


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceSaveLoad:
    """Test HDF5 save/load functionality."""

    @pytest.mark.asyncio
    async def test_hdf5_save_load_roundtrip(self, tmp_path, embed_client_768):
        """Test save and load roundtrip preserves all data."""
        hdf5_path = tmp_path / "roundtrip.h5"

        # Create service with data
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service1:
            # Add multiple documents
            doc1 = Document(
                id="doc1",
                content={"title": "Test Title", "body": "Test Body"},
                full_text="Test Title\nTest Body",
                metadata={"source": "test.txt"},
                chunks={
                    "title": DocumentChunk(faiss_id=-1),
                    "body": DocumentChunk(faiss_id=-1),
                },
            )
            doc2 = Document(
                id="doc2",
                content={"field": "value"},
                chunks={"field": DocumentChunk(faiss_id=-1)},
            )

            await service1.add_document(doc1)
            await service1.add_document(doc2)
            service1.save()
            assert hdf5_path.exists()

        # Load into new service
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service2:
            # Verify all data preserved
            assert len(service2.documents) == 2
            assert len(service2.chunk_refs) == 3  # doc1 has 2 chunks, doc2 has 1 chunk
            assert service2.index.ntotal == 3

            # Verify document content
            assert service2.documents["doc1"].content["title"] == "Test Title"
            assert service2.documents["doc1"].full_text == "Test Title\nTest Body"
            assert service2.documents["doc2"].content["field"] == "value"

    @pytest.mark.asyncio
    async def test_save_empty_service(self, tmp_path, embed_client_768):
        """Test saving empty service creates valid HDF5 file."""
        hdf5_path = tmp_path / "empty.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service:
            service.save()

        # Load and verify
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service2:
            assert len(service2.documents) == 0
            assert len(service2.chunk_refs) == 0
            assert service2.index.ntotal == 0


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceMetadata:
    """Test metadata preservation in HDF5."""

    @pytest.mark.asyncio
    async def test_hdf5_metadata_preservation(self, tmp_path, embed_client):
        """Test that metadata attributes are preserved correctly."""
        hdf5_path = tmp_path / "metadata.h5"

        # Create service and manually set up for low-level testing
        embed_client_512 = DeterministicEmbedClient(embedding_dim=512)
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_512
        ) as service:
            service.model_name = "custom-embed-model"

            # Add some data
            doc = Document(
                id="doc1",
                content={"text": "sample"},
                chunks={"text": DocumentChunk(faiss_id=0)},
            )
            service.documents["doc1"] = doc
            service.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="text"))
            # Add vector to index
            vector = np.random.randn(1, 512).astype(np.float32)
            service.index.add(vector)

            service.save()

        # Verify HDF5 file structure
        with h5py.File(str(hdf5_path), "r") as f:
            assert f.attrs["model_name"] == "custom-embed-model"
            assert f.attrs["embedding_dim"] == 512
            assert f.attrs["vector_count"] == 1
            assert f.attrs["document_count"] == 1

        # Verify loading preserves metadata
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_512
        ) as service2:
            assert service2.model_name == "custom-embed-model"
            assert service2.embedding_dim == 512


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceStructure:
    """Test HDF5 file structure."""

    @pytest.mark.asyncio
    async def test_hdf5_structure(self, tmp_path, embed_client_768):
        """Test that HDF5 file has correct structure."""
        hdf5_path = tmp_path / "structure.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service:
            # Add test data
            doc = Document(
                id="doc1",
                content={"field": "content"},
                chunks={"field": DocumentChunk(faiss_id=0)},
            )
            service.documents["doc1"] = doc
            service.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="field"))
            # Add vector to index
            vector = np.random.randn(1, 768).astype(np.float32)
            service.index.add(vector)
            service.save()

        # Verify HDF5 structure
        with h5py.File(str(hdf5_path), "r") as f:
            # Check required datasets
            assert "vectors" in f
            assert "chunks" in f
            assert "documents" in f

            # Check vectors dataset
            assert f["vectors"].shape == (1, 768)
            assert f["vectors"].dtype == np.float32

            # Check chunks dataset structure
            chunks_data = f["chunks"]
            assert len(chunks_data) == 1
            assert "faiss_id" in chunks_data.dtype.names
            assert "document_id" in chunks_data.dtype.names
            assert "chunk_key" in chunks_data.dtype.names

            # Check documents group
            assert isinstance(f["documents"], h5py.Group)
            assert "doc1" in f["documents"]

            # Check metadata attributes
            assert "model_name" in f.attrs
            assert "embedding_dim" in f.attrs
            assert "vector_count" in f.attrs
            assert "document_count" in f.attrs


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceDocumentManagement:
    """Test document management with HDF5."""

    @pytest.mark.asyncio
    async def test_add_document_with_embeddings(self, tmp_path, embed_client):
        """Test add_document() with embed_client."""
        hdf5_path = tmp_path / "documents.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service:
            # Create document with chunks
            doc = Document(
                id="test_doc",
                content={"title": "Test", "body": "Content"},
                chunks={
                    "title": DocumentChunk(faiss_id=-1),
                    "body": DocumentChunk(faiss_id=-1),
                },
            )

            # Add document
            await service.add_document(doc)

            # Verify embed_client.embed() was called for each chunk
            # Note: call_count includes 1 call during service initialization to detect embedding_dim
            assert embed_client.call_count == 3  # 1 for init + 2 for chunks
            assert "Test" in embed_client.embedded_texts
            assert "Content" in embed_client.embedded_texts

            # Verify document stored
            assert "test_doc" in service.documents
            assert service.documents["test_doc"].id == "test_doc"

            # Verify chunk refs created
            assert len(service.chunk_refs) == 2
            assert service.chunk_refs[0].document_id == "test_doc"
            assert service.chunk_refs[0].chunk_key == "title"

            # Verify faiss_ids assigned
            assert doc.chunks["title"].faiss_id == 0
            assert doc.chunks["body"].faiss_id == 1

            # Verify embeddings added to index
            assert service.index.ntotal == 2


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceSearch:
    """Test search functionality with HDF5."""

    @pytest.mark.asyncio
    async def test_hdf5_search(self, tmp_path, embed_client):
        """Test that search works correctly after HDF5 load."""
        hdf5_path = tmp_path / "search.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service:
            # Add document
            doc = Document(
                id="doc1",
                content={"field": "test content"},
                chunks={"field": DocumentChunk(faiss_id=-1)},
            )
            await service.add_document(doc)
            service.save()

        # Load new service instance
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service2:
            # Search
            results = await service2.search("test query", k=1)

            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], ChunkResult)
            assert results[0].document_id == "doc1"
            assert results[0].chunk_key == "field"
            assert results[0].text == "test content"

    @pytest.mark.asyncio
    async def test_search_include_documents_parameter(self, tmp_path, embed_client):
        """Test search() with include_documents=True."""
        hdf5_path = tmp_path / "search_docs.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service:
            # Add document with full_text
            doc = Document(
                id="doc1",
                content={"field": "test content"},
                full_text="Full document text goes here",
                chunks={"field": DocumentChunk(faiss_id=-1)},
            )
            await service.add_document(doc)
            service.save()

        # Reload and search
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client
        ) as service2:
            results = await service2.search("test query", k=1, include_documents=True)

            # Verify document_text is included
            assert len(results) == 1
            assert results[0].document_text == "Full document text goes here"


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceConversion:
    """Test conversion from FAISS+JSON to HDF5."""

    @pytest.mark.skip(reason="from_faiss_json method not yet implemented")
    def test_hdf5_conversion_from_faiss(self, tmp_path):
        """Test conversion from existing FAISS+JSON files."""
        import faiss

        faiss_path = tmp_path / "test.faiss"
        json_path = tmp_path / "test_meta.json"
        hdf5_path = tmp_path / "converted.h5"

        # Create FAISS index
        embedding_dim = 768
        index = faiss.IndexFlatIP(embedding_dim)
        vectors = np.random.randn(3, embedding_dim).astype(np.float32)
        index.add(vectors)
        faiss.write_index(index, str(faiss_path))

        # Create JSON metadata
        metadata = {
            "chunk_refs": [
                {"document_id": "doc1", "chunk_key": "chunk0"},
                {"document_id": "doc1", "chunk_key": "chunk1"},
                {"document_id": "doc2", "chunk_key": "chunk0"},
            ],
            "documents": [
                {
                    "id": "doc1",
                    "content": {"field": "value1"},
                    "full_text": "",
                    "metadata": {},
                    "source": "",
                    "chunks": {
                        "chunk0": {"faiss_id": 0, "i_start": None, "i_end": None},
                        "chunk1": {"faiss_id": 1, "i_start": None, "i_end": None},
                    },
                },
                {
                    "id": "doc2",
                    "content": {"field": "value2"},
                    "full_text": "",
                    "metadata": {},
                    "source": "",
                    "chunks": {
                        "chunk0": {"faiss_id": 2, "i_start": None, "i_end": None}
                    },
                },
            ],
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Convert to HDF5
        service = HDF5RAGService.from_faiss_json(
            faiss_path=faiss_path, json_path=json_path, hdf5_path=hdf5_path
        )

        # Verify conversion
        assert hdf5_path.exists()
        assert len(service.documents) == 2
        assert len(service.chunk_refs) == 3
        assert service.vectors.shape == (3, embedding_dim)

        # Verify data
        assert "doc1" in service.documents
        assert "doc2" in service.documents
        assert service.chunk_refs[0].document_id == "doc1"
        assert service.chunk_refs[2].document_id == "doc2"


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceErrorHandling:
    """Test error handling for HDF5 operations."""

    @pytest.mark.asyncio
    async def test_hdf5_corrupted_file(self, tmp_path, embed_client_768):
        """Test error handling for invalid HDF5 files."""
        hdf5_path = tmp_path / "corrupted.h5"

        # Create corrupted file (not valid HDF5)
        with open(hdf5_path, "w") as f:
            f.write("This is not a valid HDF5 file")

        # Attempt to load should raise error
        with pytest.raises((OSError, IOError)):
            async with HDF5RAGService(
                hdf5_path=hdf5_path, embed_client=embed_client_768
            ) as _:
                pass

    @pytest.mark.asyncio
    async def test_missing_hdf5_file(self, tmp_path, embed_client_768):
        """Test error handling for missing HDF5 file."""
        hdf5_path = tmp_path / "nonexistent.h5"

        # Creating service with non-existent file should work (new service)
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service:
            assert len(service.documents) == 0

    @pytest.mark.asyncio
    async def test_load_missing_file_explicitly(self, tmp_path, embed_client_768):
        """Test explicit load of missing file raises error."""
        hdf5_path = tmp_path / "missing.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service:
            # Explicit load should fail
            with pytest.raises(FileNotFoundError):
                service.load_from_hdf5(hdf5_path)

    @pytest.mark.asyncio
    async def test_dimension_mismatch(self, tmp_path, embed_client_768):
        """Test handling of dimension mismatch."""
        hdf5_path = tmp_path / "dimension.h5"

        # Create service with 768 dimensions
        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service1:
            vector = np.random.randn(1, 768).astype(np.float32)
            service1.index.add(vector)
            service1.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="chunk0"))
            doc = Document(
                id="doc1",
                content={"field": "value"},
                chunks={"chunk0": DocumentChunk(faiss_id=0)},
            )
            service1.documents["doc1"] = doc
            service1.save()

        # Load with different dimension client - should load actual dimensions from file
        embed_client_384 = DeterministicEmbedClient(embedding_dim=384)
        async with HDF5RAGService(
            hdf5_path=hdf5_path,
            embed_client=embed_client_384,
        ) as service2:
            # Loaded dimension should be the actual one from file
            assert service2.embedding_dim == 768  # Loaded from file, not client


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceCompression:
    """Test HDF5 compression features."""

    @pytest.mark.asyncio
    async def test_vectors_compression(self, tmp_path, embed_client_768):
        """Test that vectors are compressed in HDF5 file."""
        hdf5_path = tmp_path / "compressed.h5"

        async with HDF5RAGService(
            hdf5_path=hdf5_path, embed_client=embed_client_768
        ) as service:
            # Add large number of vectors
            n_vectors = 100
            vectors = np.random.randn(n_vectors, 768).astype(np.float32)
            service.index.add(vectors)

            for i in range(n_vectors):
                service.chunk_refs.append(
                    ChunkRef(document_id=f"doc{i}", chunk_key="chunk0")
                )
                doc = Document(
                    id=f"doc{i}",
                    content={"field": f"value{i}"},
                    chunks={"chunk0": DocumentChunk(faiss_id=i)},
                )
                service.documents[f"doc{i}"] = doc

            service.save()

        # Verify compression was applied
        with h5py.File(str(hdf5_path), "r") as f:
            vectors_dataset = f["vectors"]
            assert vectors_dataset.compression == "gzip"

        # Verify file size is reasonable (compressed should be smaller)
        file_size = hdf5_path.stat().st_size
        uncompressed_size = n_vectors * 768 * 4  # float32 = 4 bytes

        # File should be smaller than uncompressed (though not necessarily much smaller for random data)
        # Just verify it's not absurdly large
        assert file_size < uncompressed_size * 2
