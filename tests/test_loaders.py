"""Comprehensive tests for document loaders."""

import csv
import os
import pytest
from pathlib import Path

from chatboti.loaders import load_documents, load_csv
from chatboti.document import Document, DocumentChunk


class TestLoadDocuments:
    """Test load_documents function."""

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test that unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            await load_documents("dummy.txt", "test")


class TestLoadCSV:
    """Test load_csv function."""

    @pytest.fixture
    def sample_csv_path(self):
        """Path to sample CSV file."""
        return Path(__file__).parent / "sample_speakers.csv"

    @pytest.fixture
    def temp_csv(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        data = [
            {"name": "Alice", "age": "30", "bio": "Software engineer", "notes": ""},
            {"name": "Bob", "age": "25", "bio": "Data scientist", "notes": "Expert in ML"},
            {"name": "Charlie", "age": "", "bio": "", "notes": ""}
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "age", "bio", "notes"])
            writer.writeheader()
            writer.writerows(data)
        return csv_path

    @pytest.mark.asyncio
    async def test_load_with_sample_csv(self, sample_csv_path):
        """Test loading documents from sample_speakers.csv."""
        docs = await load_csv(str(sample_csv_path))

        # Check correct number of documents
        assert len(docs) == 3

        # Check first document
        doc0 = docs[0]
        assert doc0.id == "sample-speakers-0"
        assert doc0.content["name"] == "Dr. Jane Smith"
        assert doc0.content["title"] == "AI Research Lead"
        assert "machine learning" in doc0.content["bio"]
        assert "transformer architectures" in doc0.content["abstract"]

        # Check second document
        doc1 = docs[1]
        assert doc1.id == "sample-speakers-1"
        assert doc1.content["name"] == "Prof. John Doe"
        assert doc1.content["title"] == "Computer Science Professor"

        # Check third document
        doc2 = docs[2]
        assert doc2.id == "sample-speakers-2"
        assert doc2.content["name"] == "Dr. Alice Wong"

    @pytest.mark.asyncio
    async def test_document_id_format(self, temp_csv):
        """Test that document IDs follow correct format."""
        docs = await load_csv(str(temp_csv))

        assert docs[0].id == "test-data-0"
        assert docs[1].id == "test-data-1"
        assert docs[2].id == "test-data-2"

    @pytest.mark.asyncio
    async def test_doc_type_inferred_from_filename(self, temp_csv):
        """Test that doc_type is inferred from filename when not provided."""
        docs = await load_csv(str(temp_csv))

        # temp_csv is "test_data.csv", so doc_type should be "test_data"
        assert docs[0].id == "test-data-0"
        assert docs[1].id == "test-data-1"
        assert docs[2].id == "test-data-2"

    @pytest.mark.asyncio
    async def test_content_dict_contains_all_fields(self, temp_csv):
        """Test that content dict contains all CSV fields."""
        docs = await load_csv(str(temp_csv))

        # Check all fields present
        assert set(docs[0].content.keys()) == {"name", "age", "bio", "notes"}
        assert docs[0].content["name"] == "Alice"
        assert docs[0].content["age"] == "30"
        assert docs[0].content["bio"] == "Software engineer"
        assert docs[0].content["notes"] == ""

    @pytest.mark.asyncio
    async def test_chunks_created_for_all_fields_by_default(self, temp_csv):
        """Test that chunks are created for all non-empty fields by default."""
        docs = await load_csv(str(temp_csv))

        # First row: name, age, bio have values (notes is empty)
        assert "name" in docs[0].chunks
        assert "age" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "notes" not in docs[0].chunks  # Empty field should not have chunk

        # Second row: all fields have values
        assert "name" in docs[1].chunks
        assert "age" in docs[1].chunks
        assert "bio" in docs[1].chunks
        assert "notes" in docs[1].chunks

        # Third row: only name has value
        assert "name" in docs[2].chunks
        assert "age" not in docs[2].chunks  # Empty
        assert "bio" not in docs[2].chunks  # Empty
        assert "notes" not in docs[2].chunks  # Empty

    @pytest.mark.asyncio
    async def test_chunks_created_for_specific_embed_fields(self, temp_csv):
        """Test that only specified embed_fields get chunks."""
        docs = await load_csv(str(temp_csv), embed_fields=["name", "bio"])

        # Only name and bio should have chunks
        assert "name" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "age" not in docs[0].chunks
        assert "notes" not in docs[0].chunks

        # Even if bio is empty, it should still be in chunks dict if value exists
        # Actually based on code line 56-57, only creates chunk if field has value
        assert "name" in docs[2].chunks
        assert "bio" not in docs[2].chunks  # Empty value

    @pytest.mark.asyncio
    async def test_chunks_have_unassigned_faiss_id(self, temp_csv):
        """Test that all chunks have faiss_id=-1 (unassigned)."""
        docs = await load_csv(str(temp_csv))

        for doc in docs:
            for chunk in doc.chunks.values():
                assert isinstance(chunk, DocumentChunk)
                assert chunk.faiss_id == -1
                assert chunk.i_start is None
                assert chunk.i_end is None

    @pytest.mark.asyncio
    async def test_get_chunk_text_integration(self, sample_csv_path):
        """Test that loaded documents can retrieve chunk text via get_chunk_text()."""
        docs = await load_csv(str(sample_csv_path), embed_fields=["bio", "abstract"])

        # Test get_chunk_text for bio field
        bio_text = docs[0].get_chunk_text("bio")
        assert "Dr. Smith has 15 years of experience" in bio_text
        assert "machine learning" in bio_text

        # Test get_chunk_text for abstract field
        abstract_text = docs[0].get_chunk_text("abstract")
        assert "transformer architectures" in abstract_text
        assert "multi-modal learning" in abstract_text

        # Test second document
        bio_text_2 = docs[1].get_chunk_text("bio")
        assert "Prof. Doe teaches at MIT" in bio_text_2
        assert "distributed systems" in bio_text_2

    @pytest.mark.asyncio
    async def test_get_chunk_text_returns_correct_field_values(self, temp_csv):
        """Test that get_chunk_text returns exact field values."""
        docs = await load_csv(str(temp_csv))

        # Test exact value retrieval
        assert docs[0].get_chunk_text("name") == "Alice"
        assert docs[0].get_chunk_text("age") == "30"
        assert docs[0].get_chunk_text("bio") == "Software engineer"

        assert docs[1].get_chunk_text("name") == "Bob"
        assert docs[1].get_chunk_text("notes") == "Expert in ML"

    @pytest.mark.asyncio
    async def test_embed_fields_none_means_all_fields(self, temp_csv):
        """Test that embed_fields=None results in all non-empty fields being embedded."""
        docs = await load_csv(str(temp_csv), embed_fields=None)

        # Should behave same as default (all non-empty fields)
        assert "name" in docs[0].chunks
        assert "age" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "notes" not in docs[0].chunks

    @pytest.mark.asyncio
    async def test_embed_fields_with_empty_list(self, temp_csv):
        """Test that embed_fields=[] falls back to all non-empty fields."""
        docs = await load_csv(str(temp_csv), embed_fields=[])

        # Empty list is falsy, so should fall back to all non-empty fields
        assert "name" in docs[0].chunks
        assert "age" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "notes" not in docs[0].chunks

    @pytest.mark.asyncio
    async def test_embed_fields_with_nonexistent_field(self, temp_csv):
        """Test behavior when embed_fields contains fields not in CSV."""
        docs = await load_csv(str(temp_csv), embed_fields=["name", "nonexistent"])

        # Should only create chunk for name
        assert "name" in docs[0].chunks
        assert "nonexistent" not in docs[0].chunks

    @pytest.mark.asyncio
    async def test_multiple_documents_from_same_csv(self, sample_csv_path):
        """Test loading multiple documents maintains independence."""
        docs = await load_csv(str(sample_csv_path), embed_fields=["bio"])

        # Each document should be independent
        assert len(docs) == 3
        assert docs[0].id != docs[1].id
        assert docs[0].content != docs[1].content

        # Chunks should be independent objects (not the same instance)
        assert docs[0].chunks is not docs[1].chunks
        assert docs[0].chunks["bio"] is not docs[1].chunks["bio"]

        # All should have bio chunks
        for doc in docs:
            assert "bio" in doc.chunks
