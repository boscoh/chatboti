"""Comprehensive tests for document loaders."""

import csv
import os
import pytest
from pathlib import Path

from chatboti.loaders import DocumentLoader, CSVDocumentLoader
from chatboti.document import Document, DocumentChunk


class TestDocumentLoader:
    """Test DocumentLoader base class."""

    def test_base_class_is_abstract(self):
        """Test that DocumentLoader.load() raises NotImplementedError."""
        loader = DocumentLoader()
        with pytest.raises(NotImplementedError):
            loader.load("dummy.txt", "test")

    def test_custom_loader_can_override(self):
        """Test that custom loader can override load() method."""
        class CustomLoader(DocumentLoader):
            def load(self, source: str, doc_type: str):
                return [Document(id=f"{doc_type}-custom", content={"data": "test"})]

        loader = CustomLoader()
        docs = loader.load("dummy.txt", "custom")
        assert len(docs) == 1
        assert docs[0].id == "custom-custom"
        assert docs[0].content["data"] == "test"


class TestCSVDocumentLoader:
    """Test CSVDocumentLoader class."""

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

    def test_load_with_sample_csv(self, sample_csv_path):
        """Test loading documents from sample_speakers.csv."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(sample_csv_path), "speaker")

        # Check correct number of documents
        assert len(docs) == 3

        # Check first document
        doc0 = docs[0]
        assert doc0.id == "speaker-0"
        assert doc0.content["name"] == "Dr. Jane Smith"
        assert doc0.content["title"] == "AI Research Lead"
        assert "machine learning" in doc0.content["bio"]
        assert "transformer architectures" in doc0.content["abstract"]

        # Check second document
        doc1 = docs[1]
        assert doc1.id == "speaker-1"
        assert doc1.content["name"] == "Prof. John Doe"
        assert doc1.content["title"] == "Computer Science Professor"

        # Check third document
        doc2 = docs[2]
        assert doc2.id == "speaker-2"
        assert doc2.content["name"] == "Dr. Alice Wong"

    def test_document_id_format(self, temp_csv):
        """Test that document IDs follow correct format."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(temp_csv), "person")

        assert docs[0].id == "person-0"
        assert docs[1].id == "person-1"
        assert docs[2].id == "person-2"

    def test_content_dict_contains_all_fields(self, temp_csv):
        """Test that content dict contains all CSV fields."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(temp_csv), "person")

        # Check all fields present
        assert set(docs[0].content.keys()) == {"name", "age", "bio", "notes"}
        assert docs[0].content["name"] == "Alice"
        assert docs[0].content["age"] == "30"
        assert docs[0].content["bio"] == "Software engineer"
        assert docs[0].content["notes"] == ""

    def test_chunks_created_for_all_fields_by_default(self, temp_csv):
        """Test that chunks are created for all non-empty fields by default."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(temp_csv), "person")

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

    def test_chunks_created_for_specific_embed_fields(self, temp_csv):
        """Test that only specified embed_fields get chunks."""
        loader = CSVDocumentLoader(embed_fields=["name", "bio"])
        docs = loader.load(str(temp_csv), "person")

        # Only name and bio should have chunks
        assert "name" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "age" not in docs[0].chunks
        assert "notes" not in docs[0].chunks

        # Even if bio is empty, it should still be in chunks dict if value exists
        # Actually based on code line 56-57, only creates chunk if field has value
        assert "name" in docs[2].chunks
        assert "bio" not in docs[2].chunks  # Empty value

    def test_chunks_have_unassigned_faiss_id(self, temp_csv):
        """Test that all chunks have faiss_id=-1 (unassigned)."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(temp_csv), "person")

        for doc in docs:
            for chunk in doc.chunks.values():
                assert isinstance(chunk, DocumentChunk)
                assert chunk.faiss_id == -1
                assert chunk.i_start is None
                assert chunk.i_end is None

    def test_get_chunk_text_integration(self, sample_csv_path):
        """Test that loaded documents can retrieve chunk text via get_chunk_text()."""
        loader = CSVDocumentLoader(embed_fields=["bio", "abstract"])
        docs = loader.load(str(sample_csv_path), "speaker")

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

    def test_get_chunk_text_returns_correct_field_values(self, temp_csv):
        """Test that get_chunk_text returns exact field values."""
        loader = CSVDocumentLoader()
        docs = loader.load(str(temp_csv), "person")

        # Test exact value retrieval
        assert docs[0].get_chunk_text("name") == "Alice"
        assert docs[0].get_chunk_text("age") == "30"
        assert docs[0].get_chunk_text("bio") == "Software engineer"

        assert docs[1].get_chunk_text("name") == "Bob"
        assert docs[1].get_chunk_text("notes") == "Expert in ML"

    def test_embed_fields_none_means_all_fields(self, temp_csv):
        """Test that embed_fields=None results in all non-empty fields being embedded."""
        loader = CSVDocumentLoader(embed_fields=None)
        docs = loader.load(str(temp_csv), "person")

        # Should behave same as default (all non-empty fields)
        assert "name" in docs[0].chunks
        assert "age" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "notes" not in docs[0].chunks

    def test_embed_fields_with_empty_list(self, temp_csv):
        """Test that embed_fields=[] falls back to all non-empty fields."""
        loader = CSVDocumentLoader(embed_fields=[])
        docs = loader.load(str(temp_csv), "person")

        # Empty list is falsy, so should fall back to all non-empty fields
        assert "name" in docs[0].chunks
        assert "age" in docs[0].chunks
        assert "bio" in docs[0].chunks
        assert "notes" not in docs[0].chunks

    def test_embed_fields_with_nonexistent_field(self, temp_csv):
        """Test behavior when embed_fields contains fields not in CSV."""
        loader = CSVDocumentLoader(embed_fields=["name", "nonexistent"])
        docs = loader.load(str(temp_csv), "person")

        # Should only create chunk for name
        assert "name" in docs[0].chunks
        assert "nonexistent" not in docs[0].chunks

    def test_multiple_documents_from_same_csv(self, sample_csv_path):
        """Test loading multiple documents maintains independence."""
        loader = CSVDocumentLoader(embed_fields=["bio"])
        docs = loader.load(str(sample_csv_path), "speaker")

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
