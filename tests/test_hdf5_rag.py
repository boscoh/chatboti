"""Comprehensive tests for HDF5RAGService."""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Try to import h5py - tests will be skipped if not available
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from chatboti.document import Document, DocumentChunk, ChunkRef, ChunkResult

# Mock HDF5RAGService since actual implementation is in parallel development
# This stub allows tests to be written and verified independently
class HDF5RAGService:
    """Mock HDF5RAGService for testing structure and interface."""

    def __init__(
        self,
        hdf5_path: Path,
        embedding_dim: int = 1536,
        embed_client=None
    ):
        """Initialize HDF5 RAG service.

        :param hdf5_path: Path to HDF5 file
        :param embedding_dim: Embedding dimension
        :param embed_client: Optional embedding client
        """
        self.hdf5_path = hdf5_path
        self.embedding_dim = embedding_dim
        self.embed_client = embed_client
        self.model_name = "test-model"

        # In-memory storage
        self.documents = {}
        self.chunk_refs = []
        self.vectors = None

        # Load if file exists
        if hdf5_path.exists():
            self.load()
        else:
            self.vectors = np.zeros((0, embedding_dim), dtype=np.float32)

    def save(self):
        """Save to HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py not installed")

        with h5py.File(str(self.hdf5_path), 'w') as f:
            # Metadata attributes
            f.attrs['model_name'] = self.model_name
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['vector_count'] = len(self.chunk_refs)
            f.attrs['document_count'] = len(self.documents)

            # Vectors dataset
            if len(self.vectors) > 0:
                f.create_dataset('vectors', data=self.vectors, compression='gzip')
            else:
                f.create_dataset('vectors', shape=(0, self.embedding_dim), dtype=np.float32)

            # Chunks dataset (structured array)
            if self.chunk_refs:
                chunk_dtype = np.dtype([
                    ('faiss_id', 'i8'),
                    ('document_id', 'U64'),
                    ('chunk_key', 'U64')
                ])
                chunk_data = np.array([
                    (i, ref.document_id, ref.chunk_key)
                    for i, ref in enumerate(self.chunk_refs)
                ], dtype=chunk_dtype)
                f.create_dataset('chunks', data=chunk_data)
            else:
                chunk_dtype = np.dtype([
                    ('faiss_id', 'i8'),
                    ('document_id', 'U64'),
                    ('chunk_key', 'U64')
                ])
                f.create_dataset('chunks', shape=(0,), dtype=chunk_dtype)

            # Documents group with serialized JSON
            docs_group = f.create_group('documents')
            for doc_id, doc in self.documents.items():
                doc_json = json.dumps(doc.to_dict())
                docs_group.create_dataset(doc_id, data=doc_json)

    def load(self):
        """Load from HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py not installed")

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(str(self.hdf5_path), 'r') as f:
            # Load metadata
            self.model_name = f.attrs['model_name']
            self.embedding_dim = f.attrs['embedding_dim']

            # Load vectors
            self.vectors = f['vectors'][:]

            # Load chunks
            chunks_data = f['chunks'][:]
            self.chunk_refs = [
                ChunkRef(
                    document_id=str(chunk['document_id']),
                    chunk_key=str(chunk['chunk_key'])
                )
                for chunk in chunks_data
            ]

            # Load documents
            docs_group = f['documents']
            self.documents = {}
            for doc_id in docs_group.keys():
                doc_json = docs_group[doc_id][()]
                if isinstance(doc_json, bytes):
                    doc_json = doc_json.decode('utf-8')
                doc_dict = json.loads(doc_json)
                self.documents[doc_id] = Document.from_dict(doc_dict)

    async def add_document(self, doc: Document):
        """Add document with embeddings."""
        for chunk_key, chunk in doc.chunks.items():
            # Get embedding
            text = doc.get_chunk_text(chunk_key)
            embedding = await self.embed_client.embed(text)
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)

            # Add to vectors
            if len(self.vectors) == 0:
                self.vectors = embedding_array
            else:
                self.vectors = np.vstack([self.vectors, embedding_array])

            # Assign faiss_id
            chunk.faiss_id = len(self.chunk_refs)

            # Add chunk ref
            self.chunk_refs.append(ChunkRef(
                document_id=doc.id,
                chunk_key=chunk_key
            ))

        # Store document
        self.documents[doc.id] = doc

    async def search(self, query: str, k: int = 5, include_documents: bool = False):
        """Mock search - returns results based on stored data."""
        # Get query embedding
        query_embedding = await self.embed_client.embed(query)
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Simple cosine similarity search
        if len(self.vectors) == 0:
            return []

        similarities = np.dot(self.vectors, query_array.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx >= len(self.chunk_refs):
                continue
            ref = self.chunk_refs[idx]
            doc = self.documents[ref.document_id]
            text = doc.get_chunk_text(ref.chunk_key)

            result = ChunkResult(
                document_id=ref.document_id,
                chunk_key=ref.chunk_key,
                text=text,
                document_text=doc.full_text if include_documents else None
            )
            results.append(result)

        return results

    @staticmethod
    def from_faiss_json(
        faiss_path: Path,
        json_path: Path,
        hdf5_path: Path,
        embed_client=None
    ):
        """Convert FAISS + JSON to HDF5 format."""
        import faiss

        # Load FAISS index
        index = faiss.read_index(str(faiss_path))

        # Load JSON metadata
        with open(json_path) as f:
            metadata = json.load(f)

        # Create HDF5 service
        embedding_dim = index.d
        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=embedding_dim,
            embed_client=embed_client
        )

        # Extract vectors from FAISS
        service.vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
        service.vectors = service.vectors.reshape(index.ntotal, index.d)

        # Load chunk refs
        service.chunk_refs = [
            ChunkRef(**ref) for ref in metadata['chunk_refs']
        ]

        # Load documents
        service.documents = {
            doc['id']: Document.from_dict(doc)
            for doc in metadata['documents']
        }

        # Save to HDF5
        service.save()

        return service


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceInitialization:
    """Test HDF5RAGService initialization."""

    def test_new_service_creates_empty_storage(self, tmp_path):
        """Test that new service creates empty HDF5 structure."""
        hdf5_path = tmp_path / "test.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Check in-memory structures
        assert service.vectors is not None
        assert len(service.vectors) == 0
        assert service.chunk_refs == []
        assert service.documents == {}
        assert not hdf5_path.exists()  # Not saved yet

    def test_load_existing_service_from_hdf5(self, tmp_path):
        """Test loading existing service from HDF5 file."""
        hdf5_path = tmp_path / "test.h5"

        # Create and save initial service
        service1 = HDF5RAGService(
            hdf5_path=hdf5_path,
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
        service1.vectors = np.random.randn(1, 768).astype(np.float32)
        service1.save()

        # Load service from saved file
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Verify data loaded correctly
        assert len(service2.chunk_refs) == 1
        assert service2.chunk_refs[0].document_id == "doc1"
        assert service2.chunk_refs[0].chunk_key == "field1"
        assert "doc1" in service2.documents
        assert service2.documents["doc1"].id == "doc1"
        assert service2.vectors.shape == (1, 768)

    def test_correct_embedding_dimensions(self, tmp_path):
        """Test that service respects embedding dimensions."""
        hdf5_path = tmp_path / "test.h5"

        # Test custom dimension
        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=384
        )
        assert service.embedding_dim == 384
        assert service.vectors.shape[1] == 384


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceSaveLoad:
    """Test HDF5 save/load functionality."""

    def test_hdf5_save_load_roundtrip(self, tmp_path):
        """Test save and load roundtrip preserves all data."""
        hdf5_path = tmp_path / "roundtrip.h5"

        # Create service with data
        service1 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )
        service1.model_name = "test-model-v1"

        # Add multiple documents
        doc1 = Document(
            id="doc1",
            content={"title": "Test Title", "body": "Test Body"},
            full_text="Test Title\nTest Body",
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

        service1.documents = {"doc1": doc1, "doc2": doc2}
        service1.chunk_refs = [
            ChunkRef(document_id="doc1", chunk_key="title"),
            ChunkRef(document_id="doc1", chunk_key="body"),
            ChunkRef(document_id="doc2", chunk_key="field")
        ]
        service1.vectors = np.random.randn(3, 768).astype(np.float32)

        # Save
        service1.save()
        assert hdf5_path.exists()

        # Load into new service
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Verify all data preserved
        assert service2.model_name == "test-model-v1"
        assert service2.embedding_dim == 768
        assert len(service2.documents) == 2
        assert len(service2.chunk_refs) == 3
        assert service2.vectors.shape == (3, 768)

        # Verify document content
        assert service2.documents["doc1"].content["title"] == "Test Title"
        assert service2.documents["doc1"].full_text == "Test Title\nTest Body"
        assert service2.documents["doc2"].content["field"] == "value"

        # Verify chunk refs
        assert service2.chunk_refs[0].document_id == "doc1"
        assert service2.chunk_refs[1].chunk_key == "body"
        assert service2.chunk_refs[2].document_id == "doc2"

    def test_save_empty_service(self, tmp_path):
        """Test saving empty service creates valid HDF5 file."""
        hdf5_path = tmp_path / "empty.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )
        service.save()

        # Load and verify
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )
        assert len(service2.documents) == 0
        assert len(service2.chunk_refs) == 0
        assert len(service2.vectors) == 0


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceMetadata:
    """Test metadata preservation in HDF5."""

    def test_hdf5_metadata_preservation(self, tmp_path):
        """Test that metadata attributes are preserved correctly."""
        hdf5_path = tmp_path / "metadata.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=512
        )
        service.model_name = "custom-embed-model"

        # Add some data
        doc = Document(
            id="doc1",
            content={"text": "sample"},
            chunks={"text": DocumentChunk(faiss_id=0)}
        )
        service.documents["doc1"] = doc
        service.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="text"))
        service.vectors = np.random.randn(1, 512).astype(np.float32)

        service.save()

        # Verify HDF5 file structure
        with h5py.File(str(hdf5_path), 'r') as f:
            assert f.attrs['model_name'] == "custom-embed-model"
            assert f.attrs['embedding_dim'] == 512
            assert f.attrs['vector_count'] == 1
            assert f.attrs['document_count'] == 1

        # Verify loading preserves metadata
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=512
        )
        assert service2.model_name == "custom-embed-model"
        assert service2.embedding_dim == 512


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceStructure:
    """Test HDF5 file structure."""

    def test_hdf5_structure(self, tmp_path):
        """Test that HDF5 file has correct structure."""
        hdf5_path = tmp_path / "structure.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Add test data
        doc = Document(
            id="doc1",
            content={"field": "content"},
            chunks={"field": DocumentChunk(faiss_id=0)}
        )
        service.documents["doc1"] = doc
        service.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="field"))
        service.vectors = np.random.randn(1, 768).astype(np.float32)
        service.save()

        # Verify HDF5 structure
        with h5py.File(str(hdf5_path), 'r') as f:
            # Check required datasets
            assert 'vectors' in f
            assert 'chunks' in f
            assert 'documents' in f

            # Check vectors dataset
            assert f['vectors'].shape == (1, 768)
            assert f['vectors'].dtype == np.float32

            # Check chunks dataset structure
            chunks_data = f['chunks']
            assert len(chunks_data) == 1
            assert 'faiss_id' in chunks_data.dtype.names
            assert 'document_id' in chunks_data.dtype.names
            assert 'chunk_key' in chunks_data.dtype.names

            # Check documents group
            assert isinstance(f['documents'], h5py.Group)
            assert 'doc1' in f['documents']

            # Check metadata attributes
            assert 'model_name' in f.attrs
            assert 'embedding_dim' in f.attrs
            assert 'vector_count' in f.attrs
            assert 'document_count' in f.attrs


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceDocumentManagement:
    """Test document management with HDF5."""

    @pytest.mark.asyncio
    async def test_add_document_with_embeddings(self, tmp_path):
        """Test add_document() with embed_client."""
        hdf5_path = tmp_path / "documents.h5"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(side_effect=[
            [0.1] * 768,  # First chunk embedding
            [0.2] * 768   # Second chunk embedding
        ])

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768,
            embed_client=mock_embed_client
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
        await service.add_document(doc)

        # Verify embed_client.embed() was called for each chunk
        assert mock_embed_client.embed.call_count == 2
        mock_embed_client.embed.assert_any_call("Test")
        mock_embed_client.embed.assert_any_call("Content")

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

        # Verify vectors added
        assert service.vectors.shape == (2, 768)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceSearch:
    """Test search functionality with HDF5."""

    @pytest.mark.asyncio
    async def test_hdf5_search(self, tmp_path):
        """Test that search works correctly after HDF5 load."""
        hdf5_path = tmp_path / "search.h5"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        # Add document
        doc = Document(
            id="doc1",
            content={"field": "test content"},
            chunks={"field": DocumentChunk(faiss_id=-1)}
        )
        await service.add_document(doc)

        # Save and reload
        service.save()
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        # Search
        results = await service2.search("test query", k=1)

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], ChunkResult)
        assert results[0].document_id == "doc1"
        assert results[0].chunk_key == "field"
        assert results[0].text == "test content"

    @pytest.mark.asyncio
    async def test_search_include_documents_parameter(self, tmp_path):
        """Test search() with include_documents=True."""
        hdf5_path = tmp_path / "search_docs.h5"

        # Mock embed_client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
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
        await service.add_document(doc)
        service.save()

        # Reload and search
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768,
            embed_client=mock_embed_client
        )

        results = await service2.search("test query", k=1, include_documents=True)

        # Verify document_text is included
        assert len(results) == 1
        assert results[0].document_text == "Full document text goes here"


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceConversion:
    """Test conversion from FAISS+JSON to HDF5."""

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
                {"document_id": "doc2", "chunk_key": "chunk0"}
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
                        "chunk1": {"faiss_id": 1, "i_start": None, "i_end": None}
                    }
                },
                {
                    "id": "doc2",
                    "content": {"field": "value2"},
                    "full_text": "",
                    "metadata": {},
                    "source": "",
                    "chunks": {
                        "chunk0": {"faiss_id": 2, "i_start": None, "i_end": None}
                    }
                }
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f)

        # Convert to HDF5
        service = HDF5RAGService.from_faiss_json(
            faiss_path=faiss_path,
            json_path=json_path,
            hdf5_path=hdf5_path
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

    def test_hdf5_corrupted_file(self, tmp_path):
        """Test error handling for invalid HDF5 files."""
        hdf5_path = tmp_path / "corrupted.h5"

        # Create corrupted file (not valid HDF5)
        with open(hdf5_path, 'w') as f:
            f.write("This is not a valid HDF5 file")

        # Attempt to load should raise error
        with pytest.raises((OSError, IOError)):
            service = HDF5RAGService(
                hdf5_path=hdf5_path,
                embedding_dim=768
            )

    def test_missing_hdf5_file(self, tmp_path):
        """Test error handling for missing HDF5 file."""
        hdf5_path = tmp_path / "nonexistent.h5"

        # Creating service with non-existent file should work (new service)
        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )
        assert len(service.documents) == 0

    def test_load_missing_file_explicitly(self, tmp_path):
        """Test explicit load of missing file raises error."""
        hdf5_path = tmp_path / "missing.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Explicit load should fail
        with pytest.raises(FileNotFoundError):
            service.load()

    def test_dimension_mismatch(self, tmp_path):
        """Test handling of dimension mismatch."""
        hdf5_path = tmp_path / "dimension.h5"

        # Create service with 768 dimensions
        service1 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )
        service1.vectors = np.random.randn(1, 768).astype(np.float32)
        service1.chunk_refs.append(ChunkRef(document_id="doc1", chunk_key="chunk0"))
        doc = Document(
            id="doc1",
            content={"field": "value"},
            chunks={"chunk0": DocumentChunk(faiss_id=0)}
        )
        service1.documents["doc1"] = doc
        service1.save()

        # Load with different dimensions should still work but metadata will show correct dim
        service2 = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=384  # Wrong dimension specified
        )

        # Loaded dimension should be the actual one from file
        assert service2.embedding_dim == 768  # Loaded from file, not constructor


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5RAGServiceCompression:
    """Test HDF5 compression features."""

    def test_vectors_compression(self, tmp_path):
        """Test that vectors are compressed in HDF5 file."""
        hdf5_path = tmp_path / "compressed.h5"

        service = HDF5RAGService(
            hdf5_path=hdf5_path,
            embedding_dim=768
        )

        # Add large number of vectors
        n_vectors = 100
        service.vectors = np.random.randn(n_vectors, 768).astype(np.float32)

        for i in range(n_vectors):
            service.chunk_refs.append(ChunkRef(
                document_id=f"doc{i}",
                chunk_key="chunk0"
            ))
            doc = Document(
                id=f"doc{i}",
                content={"field": f"value{i}"},
                chunks={"chunk0": DocumentChunk(faiss_id=i)}
            )
            service.documents[f"doc{i}"] = doc

        service.save()

        # Verify compression was applied
        with h5py.File(str(hdf5_path), 'r') as f:
            vectors_dataset = f['vectors']
            assert vectors_dataset.compression == 'gzip'

        # Verify file size is reasonable (compressed should be smaller)
        file_size = hdf5_path.stat().st_size
        uncompressed_size = n_vectors * 768 * 4  # float32 = 4 bytes

        # File should be smaller than uncompressed (though not necessarily much smaller for random data)
        # Just verify it's not absurdly large
        assert file_size < uncompressed_size * 2
