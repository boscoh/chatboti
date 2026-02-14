"""Tests for search result building changes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from chatboti.faiss_rag import FaissRAGService
from chatboti.document import Document, DocumentChunk, ChunkRef


class TestSearchResultBuilding:
    """Test the updated search result building logic."""

    @pytest.mark.asyncio
    async def test_search_adds_document_text_when_full_text_exists(self, tmp_path):
        """Test that search adds document_text when doc.full_text exists."""
        # Mock the embed client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)
        mock_embed_client.connect = AsyncMock()
        mock_embed_client.close = AsyncMock()
        mock_embed_client.model = "test-model"

        with patch('microeval.llm.get_llm_client', return_value=mock_embed_client):
            with patch('microeval.llm.load_config', return_value={'embed_models': {'test': 'test-model'}}):
                async with FaissRAGService(
                    service_name="test",
                    model="test-model",
                    data_dir=tmp_path,
                    index_path=tmp_path / "test.index",
                    metadata_path=tmp_path / "test_meta.json"
                ) as rag:
                    # Add document with full_text
                    doc = Document(
                        id="doc1",
                        content={"field": "test content"},
                        full_text="This is the full document text",
                        chunks={"field": DocumentChunk(faiss_id=-1)}
                    )
                    await rag.add_document(doc)

                    # Search
                    results = await rag.search("test query", k=1)

                    # Verify document_text is included
                    assert len(results) == 1
                    assert results[0].document_text == "This is the full document text"

    @pytest.mark.asyncio
    async def test_search_includes_content_when_requested(self, tmp_path):
        """Test that search adds content when include_documents=True."""
        # Mock the embed client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)
        mock_embed_client.connect = AsyncMock()
        mock_embed_client.close = AsyncMock()
        mock_embed_client.model = "test-model"

        with patch('microeval.llm.get_llm_client', return_value=mock_embed_client):
            with patch('microeval.llm.load_config', return_value={'embed_models': {'test': 'test-model'}}):
                async with FaissRAGService(
                    service_name="test",
                    model="test-model",
                    data_dir=tmp_path,
                    index_path=tmp_path / "test.index",
                    metadata_path=tmp_path / "test_meta.json"
                ) as rag:
                    # Add document
                    doc = Document(
                        id="doc1",
                        content={"field": "test content", "extra": "more data"},
                        chunks={"field": DocumentChunk(faiss_id=-1)}
                    )
                    await rag.add_document(doc)

                    # Search with include_documents=True
                    results = await rag.search("test query", k=1, include_documents=True)

                    # Verify content is included
                    assert len(results) == 1
                    assert results[0].content == {"field": "test content", "extra": "more data"}

    @pytest.mark.asyncio
    async def test_get_chunk_text_extracts_field_level_chunk(self, tmp_path):
        """Test get_chunk_text with field-level chunks."""
        # Mock the embed client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)
        mock_embed_client.connect = AsyncMock()
        mock_embed_client.close = AsyncMock()
        mock_embed_client.model = "test-model"

        with patch('microeval.llm.get_llm_client', return_value=mock_embed_client):
            with patch('microeval.llm.load_config', return_value={'embed_models': {'test': 'test-model'}}):
                async with FaissRAGService(
                    service_name="test",
                    model="test-model",
                    data_dir=tmp_path,
                    index_path=tmp_path / "test.index",
                    metadata_path=tmp_path / "test_meta.json"
                ) as rag:
                    # Setup document
                    doc = Document(
                        id="doc1",
                        content={"title": "Test Title", "body": "Test Body"},
                        chunks={
                            "title": DocumentChunk(faiss_id=-1),
                            "body": DocumentChunk(faiss_id=-1)
                        }
                    )
                    rag.documents["doc1"] = doc

                    # Test get_chunk_text
                    ref_title = ChunkRef(document_id="doc1", chunk_key="title")
                    ref_body = ChunkRef(document_id="doc1", chunk_key="body")

                    assert rag.get_chunk_text(ref_title) == "Test Title"
                    assert rag.get_chunk_text(ref_body) == "Test Body"

    @pytest.mark.asyncio
    async def test_get_chunk_text_extracts_text_indices_chunk(self, tmp_path):
        """Test get_chunk_text with chunk-level (text indices) chunks."""
        # Mock the embed client
        mock_embed_client = Mock()
        mock_embed_client.embed = AsyncMock(return_value=[0.1] * 768)
        mock_embed_client.connect = AsyncMock()
        mock_embed_client.close = AsyncMock()
        mock_embed_client.model = "test-model"

        with patch('microeval.llm.get_llm_client', return_value=mock_embed_client):
            with patch('microeval.llm.load_config', return_value={'embed_models': {'test': 'test-model'}}):
                async with FaissRAGService(
                    service_name="test",
                    model="test-model",
                    data_dir=tmp_path,
                    index_path=tmp_path / "test.index",
                    metadata_path=tmp_path / "test_meta.json"
                ) as rag:
                    # Setup document with chunk-level chunks
                    full_text = "This is a long document with multiple chunks. Here is the second chunk."
                    doc = Document(
                        id="doc1",
                        full_text=full_text,
                        chunks={
                            "0": DocumentChunk(faiss_id=-1, i_start=0, i_end=45),
                            "1": DocumentChunk(faiss_id=-1, i_start=46, i_end=71)
                        }
                    )
                    rag.documents["doc1"] = doc

                    # Test get_chunk_text
                    ref_0 = ChunkRef(document_id="doc1", chunk_key="0")
                    ref_1 = ChunkRef(document_id="doc1", chunk_key="1")

                    assert rag.get_chunk_text(ref_0) == "This is a long document with multiple chunks."
                    assert rag.get_chunk_text(ref_1) == "Here is the second chunk."
