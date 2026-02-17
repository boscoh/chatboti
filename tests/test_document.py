"""Comprehensive tests for Document class and dataclasses."""


from chatboti.document import ChunkRef, ChunkResult, Document, DocumentChunk


class TestDocumentChunk:
    """Test DocumentChunk dataclass - keep only meaningful tests."""

    def test_chunk_with_indices(self):
        """Test chunk-level chunk with text indices."""
        chunk = DocumentChunk(faiss_id=456, i_start=100, i_end=200)
        assert chunk.faiss_id == 456
        assert chunk.i_start == 100
        assert chunk.i_end == 200


class TestChunkRef:
    """Test ChunkRef dataclass - keep only meaningful tests."""

    # This test is kept as it verifies the basic structure
    def test_chunk_ref_construction(self):
        """Test ChunkRef construction."""
        ref = ChunkRef(document_id="doc123", chunk_key="bio")
        assert ref.document_id == "doc123"
        assert ref.chunk_key == "bio"


class TestChunkResult:
    """Test ChunkResult dataclass - keep only meaningful tests."""

    def test_chunk_result_with_document_text(self):
        """Test ChunkResult with optional document_text."""
        result = ChunkResult(
            document_id="doc123",
            chunk_key="bio",
            text="Sample text",
            document_text="Full document text",
        )
        assert result.document_text == "Full document text"


class TestDocumentGetChunkText:
    """Test Document.get_chunk_text() method."""

    def test_field_level_chunking(self):
        """Test field-level: returns content[key] when no indices."""
        content = {"bio": "Alice is a software engineer", "skills": "Python, Go, Rust"}
        chunks = {"bio": DocumentChunk(faiss_id=1), "skills": DocumentChunk(faiss_id=2)}
        doc = Document(id="doc1", content=content, chunks=chunks)

        assert doc.get_chunk_text("bio") == "Alice is a software engineer"
        assert doc.get_chunk_text("skills") == "Python, Go, Rust"

    def test_chunk_level_chunking(self):
        """Test chunk-level: returns sliced text with i_start/i_end."""
        full_text = "The quick brown fox jumps over the lazy dog. The end."
        chunks = {
            "0": DocumentChunk(faiss_id=1, i_start=0, i_end=45),
            "1": DocumentChunk(faiss_id=2, i_start=45, i_end=53),
        }
        doc = Document(id="doc2", full_text=full_text, chunks=chunks)

        assert (
            doc.get_chunk_text("0") == "The quick brown fox jumps over the lazy dog. "
        )
        assert doc.get_chunk_text("1") == "The end."

    def test_mixed_content_types(self):
        """Test document with both content dict and full_text."""
        doc = Document(
            id="doc3",
            content={"title": "Introduction"},
            full_text="A longer text document for chunking.",
            chunks={
                "title": DocumentChunk(faiss_id=1),
                "0": DocumentChunk(faiss_id=2, i_start=0, i_end=20),
            },
        )

        assert doc.get_chunk_text("title") == "Introduction"
        assert doc.get_chunk_text("0") == "A longer text docume"


class TestDocumentGetChunkWithContext:
    """Test Document.get_chunk_with_context() method."""

    def test_field_level_no_context(self):
        """Test field-level: returns ('', chunk_text, '')."""
        content = {"bio": "Alice is a software engineer"}
        chunks = {"bio": DocumentChunk(faiss_id=1)}
        doc = Document(id="doc1", content=content, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("bio")
        assert before == ""
        assert chunk == "Alice is a software engineer"
        assert after == ""

    def test_chunk_level_with_context(self):
        """Test chunk-level: returns proper before/chunk/after."""
        full_text = "The quick brown fox jumps over the lazy dog and continues running."
        chunks = {"0": DocumentChunk(faiss_id=1, i_start=16, i_end=44)}
        doc = Document(id="doc2", full_text=full_text, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("0", context_chars=10)
        assert before == "ick brown "
        assert chunk == "fox jumps over the lazy dog "
        assert after == "and contin"

    def test_chunk_at_start_with_context(self):
        """Test context when chunk is at start of text."""
        full_text = "Start of document with more text following."
        chunks = {"0": DocumentChunk(faiss_id=1, i_start=0, i_end=17)}
        doc = Document(id="doc3", full_text=full_text, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("0", context_chars=10)
        assert before == ""
        assert chunk == "Start of document"
        assert after == " with more"

    def test_chunk_at_end_with_context(self):
        """Test context when chunk is at end of text."""
        full_text = "Text leading up to the end."
        chunks = {"0": DocumentChunk(faiss_id=1, i_start=23, i_end=27)}
        doc = Document(id="doc4", full_text=full_text, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("0", context_chars=10)
        assert before == "up to the "
        assert chunk == "end."
        assert after == ""

    def test_context_chars_larger_than_available(self):
        """Test when context_chars exceeds available text."""
        full_text = "Short text."
        chunks = {"0": DocumentChunk(faiss_id=1, i_start=6, i_end=10)}
        doc = Document(id="doc5", full_text=full_text, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("0", context_chars=100)
        assert before == "Short "
        assert chunk == "text"
        assert after == "."

    def test_default_context_chars(self):
        """Test default context_chars=200."""
        full_text = "a" * 500
        chunks = {"0": DocumentChunk(faiss_id=1, i_start=250, i_end=260)}
        doc = Document(id="doc6", full_text=full_text, chunks=chunks)

        before, chunk, after = doc.get_chunk_with_context("0")
        assert len(before) == 200
        assert chunk == "a" * 10
        assert len(after) == 200


class TestDocumentSerialization:
    """Test Document serialization (to_dict/from_dict)."""

    def test_to_dict_complete(self):
        """Test to_dict() with all fields populated."""
        doc = Document(
            id="doc1",
            content={"title": "Test", "bio": "Sample bio"},
            full_text="Full text content",
            metadata={"source": "test.txt", "timestamp": "2026-01-01"},
            chunks={
                "title": DocumentChunk(faiss_id=1),
                "0": DocumentChunk(faiss_id=2, i_start=0, i_end=10),
            },
        )

        result = doc.to_dict()
        assert result["id"] == "doc1"
        assert result["content"] == {"title": "Test", "bio": "Sample bio"}
        assert result["full_text"] == "Full text content"
        assert result["metadata"] == {"source": "test.txt", "timestamp": "2026-01-01"}
        assert "chunks" in result
        assert result["chunks"]["title"] == {
            "faiss_id": 1,
            "i_start": None,
            "i_end": None,
        }
        assert result["chunks"]["0"] == {"faiss_id": 2, "i_start": 0, "i_end": 10}

    def test_from_dict_complete(self):
        """Test from_dict() reconstruction."""
        data = {
            "id": "doc2",
            "content": {"bio": "Engineer"},
            "full_text": "Full text",
            "metadata": {"source": "db"},
            "chunks": {
                "bio": {"faiss_id": 10, "i_start": None, "i_end": None},
                "0": {"faiss_id": 20, "i_start": 5, "i_end": 15},
            },
        }

        doc = Document.from_dict(data)
        assert doc.id == "doc2"
        assert doc.content == {"bio": "Engineer"}
        assert doc.full_text == "Full text"
        assert doc.metadata == {"source": "db"}
        assert "bio" in doc.chunks
        assert doc.chunks["bio"].faiss_id == 10
        assert doc.chunks["bio"].i_start is None
        assert doc.chunks["0"].faiss_id == 20
        assert doc.chunks["0"].i_start == 5
        assert doc.chunks["0"].i_end == 15

    def test_round_trip_serialization(self):
        """Test from_dict(to_dict()) equals original."""
        original = Document(
            id="doc3",
            content={"title": "Round Trip", "body": "Content"},
            full_text="Long text for chunking",
            metadata={"version": "1.0"},
            chunks={
                "title": DocumentChunk(faiss_id=100),
                "0": DocumentChunk(faiss_id=101, i_start=0, i_end=10),
                "1": DocumentChunk(faiss_id=102, i_start=11, i_end=22),
            },
        )

        serialized = original.to_dict()
        restored = Document.from_dict(serialized)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.full_text == original.full_text
        assert restored.metadata == original.metadata
        assert len(restored.chunks) == len(original.chunks)

        for key in original.chunks:
            assert restored.chunks[key].faiss_id == original.chunks[key].faiss_id
            assert restored.chunks[key].i_start == original.chunks[key].i_start
            assert restored.chunks[key].i_end == original.chunks[key].i_end

    def test_from_dict_minimal(self):
        """Test from_dict() with minimal data."""
        data = {"id": "doc4"}
        doc = Document.from_dict(data)

        assert doc.id == "doc4"
        assert doc.content == {}
        assert doc.full_text == ""
        assert doc.metadata == {}
        assert doc.chunks == {}

    def test_from_dict_partial_chunks(self):
        """Test from_dict() with chunks missing optional fields."""
        data = {"id": "doc5", "chunks": {"bio": {"faiss_id": 42}}}
        doc = Document.from_dict(data)

        assert doc.chunks["bio"].faiss_id == 42
        assert doc.chunks["bio"].i_start is None
        assert doc.chunks["bio"].i_end is None


class TestDocumentInitialization:
    """Test Document initialization - keep only meaningful tests."""

    def test_init_none_defaults(self):
        """Test that None parameters default to empty dicts."""
        doc = Document(id="doc2", content=None, metadata=None, chunks=None)

        assert doc.content == {}
        assert doc.metadata == {}
        assert doc.chunks == {}
