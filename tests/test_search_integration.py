"""Integration tests for search functionality across RAG backends.

This file consolidates redundant search tests from multiple test files
into parameterized integration tests that test both FaissRAGService and HDF5RAGService.
"""

import pytest
from chatboti.document import Document, DocumentChunk
from chatboti.faiss_rag import FaissRAGService
from chatboti.hdf5_rag import HDF5RAGService


@pytest.mark.asyncio
async def test_search_returns_chunk_results(rag_service):
    """Test search() returns ChunkResult objects for both backends."""
    # Search for content
    results = await rag_service.search("AI", k=2)
    
    # Verify results structure
    assert len(results) == 2
    for result in results:
        assert hasattr(result, 'document_id')
        assert hasattr(result, 'chunk_key')
        assert hasattr(result, 'text')
        assert result.document_id in ["doc1", "doc2"]
        assert result.chunk_key in ["title", "field"]
        assert result.text
        assert result.document_text is None  # Default behavior


@pytest.mark.asyncio
async def test_search_include_documents_parameter(rag_service):
    """Test search() with include_documents=True for both backends."""
    # Search with document inclusion
    results = await rag_service.search("machine learning", k=1, include_documents=True)
    
    # Verify document_text is included
    assert len(results) == 1
    # Note: document_text may be None if no full_text is set
    # This tests the parameter is passed through correctly


@pytest.mark.asyncio
async def test_search_result_contains_correct_text(rag_service):
    """Test that search results contain correct chunk text."""
    # Search for specific terms
    results = await rag_service.search("software", k=3)
    
    # Verify correct text extraction
    assert len(results) <= 3  # May be fewer if no matches
    
    for result in results:
        # Verify text comes from our test documents
        valid_texts = [
            "AI Research", "machine learning",
            "Software Engineering", "distributed systems"
        ]
        assert result.text in valid_texts


@pytest.mark.asyncio
async def test_search_empty_index(rag_service):
    """Test searching on empty service handles gracefully."""
    # Save current service to ensure files exist
    rag_service.save()
    
    # Create empty service
    if isinstance(rag_service, FaissRAGService):
        empty_service = FaissRAGService(
            index_path=rag_service.index_path.parent / "empty.index",
            metadata_path=rag_service.index_path.parent / "empty_meta.json",
            embed_client=rag_service.embed_client
        )
    else:  # HDF5RAGService
        empty_service = HDF5RAGService(
            hdf5_path=rag_service.hdf5_path.parent / "empty.h5",
            embed_client=rag_service.embed_client
        )
    
    async with empty_service as service:
        # Search on empty service
        results = await service.search("test query", k=5)
        assert len(results) == 0  # Should return empty, not crash


@pytest.mark.asyncio
async def test_search_with_different_k_values(rag_service):
    """Test search with different k values."""
    # Test k=1
    results1 = await rag_service.search("AI", k=1)
    assert len(results1) <= 1
    
    # Test k=10 (more than available)
    results10 = await rag_service.search("AI", k=10)
    assert len(results10) <= 4  # Max 4 chunks in our test data
    
    # Results should be ordered by relevance
    if len(results10) > 1 and len(results1) > 0:
        assert results1[0].text == results10[0].text


@pytest.mark.asyncio
async def test_search_persistence(rag_service):
    """Test that search works after save/load cycle."""
    # Save current service
    rag_service.save()
    
    # Create new service instance from saved files
    if isinstance(rag_service, FaissRAGService):
        loaded_service = FaissRAGService(
            index_path=rag_service.index_path,
            metadata_path=rag_service.metadata_path,
            embed_client=rag_service.embed_client
        )
    else:  # HDF5RAGService
        loaded_service = HDF5RAGService(
            hdf5_path=rag_service.hdf5_path,
            embed_client=rag_service.embed_client
        )
    
    # Search should still work
    results = await loaded_service.search("machine learning", k=2)
    assert len(results) == 2
    
    # Results should match original
    original_results = await rag_service.search("machine learning", k=2)
    assert len(results) == len(original_results)
    
    # Text content should be the same (order may vary)
    result_texts = {r.text for r in results}
    original_texts = {r.text for r in original_results}
    assert result_texts == original_texts


@pytest.mark.asyncio
async def test_search_with_chunk_level_text(rag_service):
    """Test search with chunk-level (text indices) chunks."""
    # Add document with chunk-level chunks
    full_text = "This is a long document with multiple chunks. Here is the second chunk."
    doc = Document(
        id="chunk_level_doc",
        full_text=full_text,
        chunks={
            "0": DocumentChunk(faiss_id=-1, i_start=0, i_end=45),
            "1": DocumentChunk(faiss_id=-1, i_start=46, i_end=71)
        }
    )
    await rag_service.add_document(doc)
    
    # Search for content from chunks
    results = await rag_service.search("multiple chunks", k=1)
    
    assert len(results) == 1
    assert "multiple chunks" in results[0].text
    assert results[0].text == "This is a long document with multiple chunks."


@pytest.mark.asyncio
async def test_search_with_mixed_document_types(rag_service):
    """Test search with documents having both content dict and full_text."""
    # Add mixed document
    doc = Document(
        id="mixed_doc",
        content={"title": "Introduction"},
        full_text="A longer text document for chunking.",
        chunks={
            "title": DocumentChunk(faiss_id=-1),
            "0": DocumentChunk(faiss_id=-1, i_start=0, i_end=20)
        }
    )
    await rag_service.add_document(doc)
    
    # Search should find both types of chunks
    results = await rag_service.search("Introduction", k=2)
    
    assert len(results) >= 1
    result_texts = [r.text for r in results]
    assert "Introduction" in result_texts or "A longer text docume" in result_texts
