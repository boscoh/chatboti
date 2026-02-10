"""
Test embedding storage using ChromaDB, Parquet, FAISS, and JSON.
"""

import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import chromadb


def generate_sample_embeddings(n_items: int = 10, dim: int = 1536) -> list[dict]:
    """Generate sample speaker data with random embeddings."""
    speakers = []
    for i in range(n_items):
        speakers.append({
            "id": f"speaker_{i}",
            "name": f"Speaker {i}",
            "abstract_embedding": np.random.randn(dim).tolist(),
            "bio_embedding": np.random.randn(dim).tolist(),
        })
    return speakers


class TestChromaDBStorage:
    """Test embedding storage using ChromaDB."""

    def test_store_and_retrieve_embeddings(self):
        speakers = generate_sample_embeddings(n_items=5, dim=128)

        client = chromadb.Client()
        collection = client.create_collection(name="speakers")

        ids = []
        embeddings = []
        metadatas = []
        for speaker in speakers:
            ids.append(speaker["id"])
            embeddings.append(speaker["abstract_embedding"])
            metadatas.append({"name": speaker["name"]})

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

        query_embedding = speakers[0]["abstract_embedding"]
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        assert results["ids"][0][0] == speakers[0]["id"]
        assert len(results["ids"][0]) == 3

    def test_persistent_storage(self, tmp_path):
        speakers = generate_sample_embeddings(n_items=3, dim=64)

        client = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        collection = client.create_collection(name="speakers")

        for speaker in speakers:
            collection.add(
                ids=[speaker["id"]],
                embeddings=[speaker["abstract_embedding"]],
                metadatas=[{"name": speaker["name"]}],
            )

        del client

        client2 = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        collection2 = client2.get_collection(name="speakers")
        assert collection2.count() == 3


class TestParquetStorage:
    """Test embedding storage using Parquet with int8 quantization."""

    @staticmethod
    def quantize_to_int8(embedding: list[float]) -> np.ndarray:
        """Quantize float embedding to int8 for compact storage."""
        arr = np.array(embedding, dtype=np.float32)
        max_val = np.abs(arr).max()
        if max_val > 0:
            scaled = arr / max_val * 127
        else:
            scaled = arr
        return scaled.astype(np.int8)

    @staticmethod
    def dequantize_from_int8(
        quantized: np.ndarray, original_max: float
    ) -> np.ndarray:
        """Dequantize int8 back to float."""
        return quantized.astype(np.float32) / 127 * original_max

    def test_quantization_roundtrip(self):
        original = np.random.randn(128).tolist()
        quantized = self.quantize_to_int8(original)

        assert quantized.dtype == np.int8
        assert len(quantized) == 128
        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_store_embeddings_as_parquet(self, tmp_path):
        speakers = generate_sample_embeddings(n_items=5, dim=128)

        ids = []
        names = []
        abstract_embeddings = []
        abstract_maxvals = []
        bio_embeddings = []
        bio_maxvals = []

        for speaker in speakers:
            ids.append(speaker["id"])
            names.append(speaker["name"])

            abstract_arr = np.array(speaker["abstract_embedding"])
            abstract_max = float(np.abs(abstract_arr).max())
            abstract_maxvals.append(abstract_max)
            abstract_embeddings.append(self.quantize_to_int8(speaker["abstract_embedding"]))

            bio_arr = np.array(speaker["bio_embedding"])
            bio_max = float(np.abs(bio_arr).max())
            bio_maxvals.append(bio_max)
            bio_embeddings.append(self.quantize_to_int8(speaker["bio_embedding"]))

        table = pa.table({
            "id": ids,
            "name": names,
            "abstract_embedding": abstract_embeddings,
            "abstract_max": abstract_maxvals,
            "bio_embedding": bio_embeddings,
            "bio_max": bio_maxvals,
        })

        parquet_path = tmp_path / "embeddings.parquet"
        pq.write_table(table, parquet_path)

        assert parquet_path.exists()
        file_size = parquet_path.stat().st_size
        print(f"Parquet file size: {file_size} bytes for {len(speakers)} speakers")

    def test_load_and_search_parquet(self, tmp_path):
        speakers = generate_sample_embeddings(n_items=10, dim=64)

        ids = []
        embeddings_int8 = []
        max_vals = []

        for speaker in speakers:
            ids.append(speaker["id"])
            arr = np.array(speaker["abstract_embedding"])
            max_vals.append(float(np.abs(arr).max()))
            embeddings_int8.append(self.quantize_to_int8(speaker["abstract_embedding"]))

        table = pa.table({
            "id": ids,
            "embedding": embeddings_int8,
            "max_val": max_vals,
        })

        parquet_path = tmp_path / "embeddings.parquet"
        pq.write_table(table, parquet_path)

        loaded_table = pq.read_table(parquet_path)
        loaded_ids = loaded_table["id"].to_pylist()
        loaded_embeddings = loaded_table["embedding"].to_numpy()
        loaded_max_vals = loaded_table["max_val"].to_pylist()

        assert len(loaded_ids) == 10

        query = np.array(speakers[0]["abstract_embedding"])
        query_normalized = query / np.linalg.norm(query)

        distances = []
        for i, emb_int8 in enumerate(loaded_embeddings):
            emb_float = self.dequantize_from_int8(emb_int8, loaded_max_vals[i])
            emb_normalized = emb_float / np.linalg.norm(emb_float)
            cosine_sim = np.dot(query_normalized, emb_normalized)
            distances.append(1 - cosine_sim)

        best_idx = np.argmin(distances)
        assert loaded_ids[best_idx] == speakers[0]["id"]

    def test_store_and_search_float32(self, tmp_path):
        """Test parquet with float32 - no quantization, exact results."""
        speakers = generate_sample_embeddings(n_items=10, dim=64)

        ids = [s["id"] for s in speakers]
        embeddings = [
            np.array(s["abstract_embedding"], dtype=np.float32) for s in speakers
        ]

        table = pa.table({"id": ids, "embedding": embeddings})
        parquet_path = tmp_path / "embeddings_f32.parquet"
        pq.write_table(table, parquet_path)

        file_size = parquet_path.stat().st_size
        print(f"Parquet (float32) file size: {file_size} bytes for {len(speakers)} speakers")

        loaded_table = pq.read_table(parquet_path)
        loaded_ids = loaded_table["id"].to_pylist()
        loaded_embeddings = [
            np.array(e, dtype=np.float32)
            for e in loaded_table["embedding"].to_pylist()
        ]

        assert len(loaded_ids) == 10

        query = np.array(speakers[0]["abstract_embedding"], dtype=np.float32)

        def cosine_distance(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        distances = [cosine_distance(query, emb) for emb in loaded_embeddings]
        best_idx = np.argmin(distances)

        print(f"Query: {speakers[0]['id']}")
        print(f"Best match: {loaded_ids[best_idx]} (distance: {distances[best_idx]:.2e})")

        assert loaded_ids[best_idx] == speakers[0]["id"]
        assert distances[best_idx] < 1e-6


class TestFAISSStorage:
    """Test embedding storage using FAISS."""

    def test_store_and_retrieve_embeddings(self):
        speakers = generate_sample_embeddings(n_items=5, dim=128)

        embeddings = np.array(
            [s["abstract_embedding"] for s in speakers], dtype=np.float32
        )

        index = faiss.IndexFlatL2(128)
        index.add(embeddings)

        query = embeddings[0:1]
        distances, indices = index.search(query, k=3)

        assert indices[0][0] == 0
        assert len(indices[0]) == 3

    def test_persistent_storage(self, tmp_path):
        speakers = generate_sample_embeddings(n_items=10, dim=64)

        embeddings = np.array(
            [s["abstract_embedding"] for s in speakers], dtype=np.float32
        )

        index = faiss.IndexFlatL2(64)
        index.add(embeddings)

        index_path = tmp_path / "faiss.index"
        faiss.write_index(index, str(index_path))

        loaded_index = faiss.read_index(str(index_path))
        assert loaded_index.ntotal == 10

        query = embeddings[0:1]
        distances, indices = loaded_index.search(query, k=3)
        assert indices[0][0] == 0

    def test_cosine_similarity_index(self):
        """FAISS IndexFlatIP with normalized vectors for cosine similarity."""
        speakers = generate_sample_embeddings(n_items=10, dim=128)

        embeddings = np.array(
            [s["abstract_embedding"] for s in speakers], dtype=np.float32
        )
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(128)
        index.add(embeddings)

        query = np.array([speakers[0]["abstract_embedding"]], dtype=np.float32)
        faiss.normalize_L2(query)

        similarities, indices = index.search(query, k=3)

        assert indices[0][0] == 0
        assert similarities[0][0] > 0.99


class TestStorageComparison:
    """Compare storage sizes and retrieval accuracy between formats."""

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        return 1.0 - dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def quantize_to_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
        max_val = np.abs(arr).max()
        scaled = arr / max_val * 127 if max_val > 0 else arr
        return scaled.astype(np.int8), max_val

    @staticmethod
    def dequantize_from_int8(quantized: np.ndarray, max_val: float) -> np.ndarray:
        return quantized.astype(np.float32) / 127 * max_val

    def test_tabulated_comparison(self, tmp_path):
        """Combined storage size and retrieval accuracy comparison."""
        np.random.seed(42)
        n_items = 100
        dim = 1536
        k = 10
        n_queries = 20

        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        ids = [f"item_{i}" for i in range(n_items)]

        ground_truth = []
        for query in queries:
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            ground_truth.append(np.argsort(distances)[:k].tolist())

        def recall_at_k(predicted: list[int], truth: list[int]) -> float:
            return len(set(predicted) & set(truth)) / len(truth)

        def get_dir_size(path):
            total = 0
            for p in Path(path).rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
            return total

        results = []

        # JSON (indented)
        json_path = tmp_path / "embeddings.json"
        json_data = [{"id": ids[i], "embedding": embeddings[i].tolist()} for i in range(n_items)]
        json_path.write_text(json.dumps(json_data, indent=2))
        recalls = []
        for i, query in enumerate(queries):
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            recalls.append(recall_at_k(np.argsort(distances)[:k].tolist(), ground_truth[i]))
        results.append(("JSON (indented)", json_path.stat().st_size, np.mean(recalls)))

        # JSON (compact)
        json_compact_path = tmp_path / "embeddings_compact.json"
        json_compact_path.write_text(json.dumps(json_data))
        results.append(("JSON (compact)", json_compact_path.stat().st_size, np.mean(recalls)))

        # ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        collection = chroma_client.create_collection(name="test", metadata={"hnsw:space": "cosine"})
        collection.add(ids=ids, embeddings=embeddings.tolist())
        recalls = []
        for i, query in enumerate(queries):
            res = collection.query(query_embeddings=[query.tolist()], n_results=k)
            top_k = [int(id_.split("_")[1]) for id_ in res["ids"][0]]
            recalls.append(recall_at_k(top_k, ground_truth[i]))
        del chroma_client
        results.append(("ChromaDB", get_dir_size(tmp_path / "chroma"), np.mean(recalls)))

        # FAISS
        embeddings_norm = embeddings.copy()
        faiss.normalize_L2(embeddings_norm)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_norm)
        faiss_path = tmp_path / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        recalls = []
        for i, query in enumerate(queries):
            q = query.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            _, indices = index.search(q, k)
            recalls.append(recall_at_k(indices[0].tolist(), ground_truth[i]))
        results.append(("FAISS", faiss_path.stat().st_size, np.mean(recalls)))

        # Parquet (float32)
        table_f32 = pa.table({"id": ids, "embedding": [emb for emb in embeddings]})
        parquet_f32_path = tmp_path / "embeddings_f32.parquet"
        pq.write_table(table_f32, parquet_f32_path)
        recalls = []
        for i, query in enumerate(queries):
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            recalls.append(recall_at_k(np.argsort(distances)[:k].tolist(), ground_truth[i]))
        results.append(("Parquet (float32)", parquet_f32_path.stat().st_size, np.mean(recalls)))

        # Parquet (int8)
        quantized = []
        max_vals = []
        for emb in embeddings:
            q, m = self.quantize_to_int8(emb)
            quantized.append(q)
            max_vals.append(m)
        table_int8 = pa.table({"id": ids, "embedding": quantized, "max_val": max_vals})
        parquet_int8_path = tmp_path / "embeddings_int8.parquet"
        pq.write_table(table_int8, parquet_int8_path)
        recalls = []
        for i, query in enumerate(queries):
            distances = []
            for j, q_emb in enumerate(quantized):
                dequant = self.dequantize_from_int8(q_emb, max_vals[j])
                distances.append(self.cosine_distance(query, dequant))
            recalls.append(recall_at_k(np.argsort(distances)[:k].tolist(), ground_truth[i]))
        results.append(("Parquet (int8)", parquet_int8_path.stat().st_size, np.mean(recalls)))

        results.sort(key=lambda x: x[1], reverse=True)
        min_size = min(r[1] for r in results)

        print(f"\n{'='*70}")
        print(f"Embedding Storage Comparison ({n_items} items, {dim}-dim, Recall@{k})")
        print(f"{'='*70}")
        print(f"{'Format':<20} {'Size':>12} {'Size (KB)':>12} {'Ratio':>8} {'Recall':>10}")
        print(f"{'-'*70}")
        for name, size, recall in results:
            ratio = size / min_size
            print(f"{name:<20} {size:>12,} {size/1024:>12.1f} {ratio:>7.1f}x {recall:>10.1%}")
        print(f"{'='*70}\n")

        assert results[-1][1] < results[0][1]


class TestRetrievalAccuracy:
    """Compare retrieval accuracy across different storage methods."""

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return 1.0 - (dot_product / (norm_a * norm_b))

    @staticmethod
    def quantize_to_int8(embedding: np.ndarray) -> tuple[np.ndarray, float]:
        max_val = np.abs(embedding).max()
        if max_val > 0:
            scaled = embedding / max_val * 127
        else:
            scaled = embedding
        return scaled.astype(np.int8), max_val

    @staticmethod
    def dequantize_from_int8(quantized: np.ndarray, max_val: float) -> np.ndarray:
        return quantized.astype(np.float32) / 127 * max_val

    def test_retrieval_accuracy_comparison(self, tmp_path):
        """Compare retrieval accuracy of all methods against ground truth."""
        np.random.seed(42)
        n_items = 100
        dim = 256
        k = 10
        n_queries = 20

        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        ground_truth = []
        for query in queries:
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            top_k = np.argsort(distances)[:k].tolist()
            ground_truth.append(top_k)

        def recall_at_k(predicted: list[int], truth: list[int]) -> float:
            return len(set(predicted) & set(truth)) / len(truth)

        # JSON/NumPy baseline (exact, should be 100%)
        json_recalls = []
        for i, query in enumerate(queries):
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            top_k = np.argsort(distances)[:k].tolist()
            json_recalls.append(recall_at_k(top_k, ground_truth[i]))

        # ChromaDB
        client = chromadb.Client()
        collection = client.create_collection(
            name="test", metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=[str(i) for i in range(n_items)],
            embeddings=embeddings.tolist(),
        )

        chroma_recalls = []
        for i, query in enumerate(queries):
            results = collection.query(query_embeddings=[query.tolist()], n_results=k)
            top_k = [int(id_) for id_ in results["ids"][0]]
            chroma_recalls.append(recall_at_k(top_k, ground_truth[i]))

        # FAISS (cosine via normalized inner product)
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_normalized)

        faiss_recalls = []
        for i, query in enumerate(queries):
            q = query.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            _, indices = index.search(q, k)
            top_k = indices[0].tolist()
            faiss_recalls.append(recall_at_k(top_k, ground_truth[i]))

        # Parquet with float32 (exact, same as JSON)
        parquet_f32_recalls = []
        for i, query in enumerate(queries):
            distances = [self.cosine_distance(query, emb) for emb in embeddings]
            top_k = np.argsort(distances)[:k].tolist()
            parquet_f32_recalls.append(recall_at_k(top_k, ground_truth[i]))

        # Parquet with int8 quantization
        quantized_embeddings = []
        max_vals = []
        for emb in embeddings:
            q, m = self.quantize_to_int8(emb)
            quantized_embeddings.append(q)
            max_vals.append(m)

        parquet_int8_recalls = []
        for i, query in enumerate(queries):
            distances = []
            for j, q_emb in enumerate(quantized_embeddings):
                dequantized = self.dequantize_from_int8(q_emb, max_vals[j])
                distances.append(self.cosine_distance(query, dequantized))
            top_k = np.argsort(distances)[:k].tolist()
            parquet_int8_recalls.append(recall_at_k(top_k, ground_truth[i]))

        avg_json = np.mean(json_recalls)
        avg_chroma = np.mean(chroma_recalls)
        avg_faiss = np.mean(faiss_recalls)
        avg_parquet_f32 = np.mean(parquet_f32_recalls)
        avg_parquet_int8 = np.mean(parquet_int8_recalls)

        print(f"\nRetrieval Recall@{k} for {n_queries} queries on {n_items} items:")
        print(f"  JSON/NumPy (exact):   {avg_json:.1%}")
        print(f"  ChromaDB:             {avg_chroma:.1%}")
        print(f"  FAISS:                {avg_faiss:.1%}")
        print(f"  Parquet (float32):    {avg_parquet_f32:.1%}")
        print(f"  Parquet (int8):       {avg_parquet_int8:.1%}")

        assert avg_json == 1.0
        assert avg_chroma >= 0.95
        assert avg_faiss >= 0.95
        assert avg_parquet_f32 == 1.0
        assert avg_parquet_int8 >= 0.90

    def test_quantization_impact_on_ranking(self):
        """Test how int8 quantization affects ranking quality."""
        np.random.seed(123)
        n_items = 50
        dim = 128

        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        exact_distances = [self.cosine_distance(query, emb) for emb in embeddings]
        exact_ranking = np.argsort(exact_distances)

        quantized_distances = []
        for emb in embeddings:
            q, m = self.quantize_to_int8(emb)
            dequantized = self.dequantize_from_int8(q, m)
            quantized_distances.append(self.cosine_distance(query, dequantized))
        quantized_ranking = np.argsort(quantized_distances)

        def rank_correlation(rank1: np.ndarray, rank2: np.ndarray) -> float:
            n = len(rank1)
            rank1_pos = np.zeros(n)
            rank2_pos = np.zeros(n)
            for pos, item in enumerate(rank1):
                rank1_pos[item] = pos
            for pos, item in enumerate(rank2):
                rank2_pos[item] = pos
            d_squared = np.sum((rank1_pos - rank2_pos) ** 2)
            return 1 - (6 * d_squared) / (n * (n**2 - 1))

        correlation = rank_correlation(exact_ranking, quantized_ranking)
        top_5_match = len(set(exact_ranking[:5]) & set(quantized_ranking[:5])) / 5
        top_10_match = len(set(exact_ranking[:10]) & set(quantized_ranking[:10])) / 10

        print(f"\nQuantization impact on ranking:")
        print(f"  Spearman correlation: {correlation:.3f}")
        print(f"  Top-5 overlap:        {top_5_match:.0%}")
        print(f"  Top-10 overlap:       {top_10_match:.0%}")

        assert correlation > 0.95
        assert top_5_match >= 0.8
        assert top_10_match >= 0.8


class TestDocumentChunkEmbeddingSchema:
    """Test efficient schema for documents, chunks, and embeddings."""

    @staticmethod
    def quantize_to_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
        max_val = np.abs(arr).max()
        scaled = arr / max_val * 127 if max_val > 0 else arr
        return scaled.astype(np.int8), max_val

    @staticmethod
    def dequantize_from_int8(quantized: np.ndarray, max_val: float) -> np.ndarray:
        return quantized.astype(np.float32) / 127 * max_val

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_separate_tables_with_faiss_index(self, tmp_path):
        """
        Schema 1: Separate files for documents, chunks, and FAISS index.
        - documents.parquet: doc_id, title, source, metadata
        - chunks.parquet: chunk_id, doc_id, text, start_char, end_char
        - embeddings.index: FAISS index (chunk_id = row index)
        
        Pros: Fast vector search, chunks/docs loaded only when needed
        Cons: Multiple files to manage
        """
        np.random.seed(42)
        n_docs = 10
        chunks_per_doc = 5
        dim = 256

        # Generate sample data
        documents = []
        chunks = []
        embeddings = []

        for doc_id in range(n_docs):
            documents.append({
                "doc_id": doc_id,
                "title": f"Document {doc_id}",
                "source": f"file_{doc_id}.pdf",
            })
            for chunk_idx in range(chunks_per_doc):
                chunk_id = doc_id * chunks_per_doc + chunk_idx
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": f"This is chunk {chunk_idx} of document {doc_id}. " * 10,
                    "start_char": chunk_idx * 500,
                    "end_char": (chunk_idx + 1) * 500,
                })
                embeddings.append(np.random.randn(dim).astype(np.float32))

        # Save documents
        doc_table = pa.table({
            "doc_id": [d["doc_id"] for d in documents],
            "title": [d["title"] for d in documents],
            "source": [d["source"] for d in documents],
        })
        pq.write_table(doc_table, tmp_path / "documents.parquet")

        # Save chunks (without embeddings)
        chunk_table = pa.table({
            "chunk_id": [c["chunk_id"] for c in chunks],
            "doc_id": [c["doc_id"] for c in chunks],
            "text": [c["text"] for c in chunks],
            "start_char": [c["start_char"] for c in chunks],
            "end_char": [c["end_char"] for c in chunks],
        })
        pq.write_table(chunk_table, tmp_path / "chunks.parquet")

        # Save FAISS index
        embeddings_np = np.array(embeddings)
        faiss.normalize_L2(embeddings_np)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)
        faiss.write_index(index, str(tmp_path / "embeddings.index"))

        # Query workflow
        query = np.random.randn(dim).astype(np.float32)
        query_norm = query.copy().reshape(1, -1)
        faiss.normalize_L2(query_norm)

        loaded_index = faiss.read_index(str(tmp_path / "embeddings.index"))
        scores, chunk_ids = loaded_index.search(query_norm, k=3)

        loaded_chunks = pq.read_table(tmp_path / "chunks.parquet")
        loaded_docs = pq.read_table(tmp_path / "documents.parquet")

        print("\nSchema 1: Separate tables + FAISS index")
        print("-" * 50)
        for i, chunk_id in enumerate(chunk_ids[0]):
            chunk_row = loaded_chunks.filter(
                pa.compute.equal(loaded_chunks["chunk_id"], int(chunk_id))
            ).to_pydict()
            doc_id = chunk_row["doc_id"][0]
            doc_row = loaded_docs.filter(
                pa.compute.equal(loaded_docs["doc_id"], doc_id)
            ).to_pydict()
            print(f"  #{i+1} chunk_id={chunk_id}, doc='{doc_row['title'][0]}', score={scores[0][i]:.4f}")

        # File sizes
        doc_size = (tmp_path / "documents.parquet").stat().st_size
        chunk_size = (tmp_path / "chunks.parquet").stat().st_size
        index_size = (tmp_path / "embeddings.index").stat().st_size
        total = doc_size + chunk_size + index_size
        print(f"\nFile sizes:")
        print(f"  documents.parquet: {doc_size:,} bytes")
        print(f"  chunks.parquet:    {chunk_size:,} bytes")
        print(f"  embeddings.index:  {index_size:,} bytes")
        print(f"  Total:             {total:,} bytes")

        assert len(chunk_ids[0]) == 3

    def test_single_table_with_int8_embeddings(self, tmp_path):
        """
        Schema 2: Single denormalized parquet with int8 embeddings.
        - chunks.parquet: chunk_id, doc_id, doc_title, text, embedding, emb_scale
        
        Pros: Single file, simple to manage, portable
        Cons: Slower search (linear scan), duplicated doc metadata
        """
        np.random.seed(42)
        n_docs = 10
        chunks_per_doc = 5
        dim = 256

        chunk_ids = []
        doc_ids = []
        doc_titles = []
        texts = []
        embeddings_int8 = []
        emb_scales = []

        for doc_id in range(n_docs):
            for chunk_idx in range(chunks_per_doc):
                chunk_id = doc_id * chunks_per_doc + chunk_idx
                chunk_ids.append(chunk_id)
                doc_ids.append(doc_id)
                doc_titles.append(f"Document {doc_id}")
                texts.append(f"This is chunk {chunk_idx} of document {doc_id}. " * 10)

                emb = np.random.randn(dim).astype(np.float32)
                q, scale = self.quantize_to_int8(emb)
                embeddings_int8.append(q)
                emb_scales.append(scale)

        table = pa.table({
            "chunk_id": chunk_ids,
            "doc_id": doc_ids,
            "doc_title": doc_titles,
            "text": texts,
            "embedding": embeddings_int8,
            "emb_scale": emb_scales,
        })
        pq.write_table(table, tmp_path / "chunks.parquet")

        # Query workflow
        loaded = pq.read_table(tmp_path / "chunks.parquet")
        loaded_embeddings = loaded["embedding"].to_pylist()
        loaded_scales = loaded["emb_scale"].to_pylist()

        query = np.random.randn(dim).astype(np.float32)
        query_norm = query / np.linalg.norm(query)

        distances = []
        for i, emb_int8 in enumerate(loaded_embeddings):
            emb = self.dequantize_from_int8(np.array(emb_int8), loaded_scales[i])
            emb_norm = emb / np.linalg.norm(emb)
            distances.append(1 - np.dot(query_norm, emb_norm))

        top_k_idx = np.argsort(distances)[:3]

        print("\nSchema 2: Single table with int8 embeddings")
        print("-" * 50)
        for i, idx in enumerate(top_k_idx):
            row = {col: loaded[col][idx].as_py() for col in ["chunk_id", "doc_id", "doc_title"]}
            print(f"  #{i+1} chunk_id={row['chunk_id']}, doc='{row['doc_title']}', dist={distances[idx]:.4f}")

        file_size = (tmp_path / "chunks.parquet").stat().st_size
        print(f"\nFile size: {file_size:,} bytes")

        assert len(top_k_idx) == 3

    def test_hybrid_parquet_with_faiss(self, tmp_path):
        """
        Schema 3: Parquet for metadata + FAISS for search (recommended).
        - chunks.parquet: chunk_id, doc_id, doc_title, text (NO embeddings)
        - embeddings.index: FAISS index where row = chunk_id
        
        Pros: Fast search, compact storage, text separate from vectors
        Cons: Two files, need to keep in sync
        """
        np.random.seed(42)
        n_docs = 10
        chunks_per_doc = 5
        dim = 256

        chunk_ids = []
        doc_ids = []
        doc_titles = []
        texts = []
        embeddings = []

        for doc_id in range(n_docs):
            for chunk_idx in range(chunks_per_doc):
                chunk_id = doc_id * chunks_per_doc + chunk_idx
                chunk_ids.append(chunk_id)
                doc_ids.append(doc_id)
                doc_titles.append(f"Document {doc_id}")
                texts.append(f"This is chunk {chunk_idx} of document {doc_id}. " * 10)
                embeddings.append(np.random.randn(dim).astype(np.float32))

        # Save chunks (no embeddings - keeps file small)
        table = pa.table({
            "chunk_id": chunk_ids,
            "doc_id": doc_ids,
            "doc_title": doc_titles,
            "text": texts,
        })
        pq.write_table(table, tmp_path / "chunks.parquet")

        # Save FAISS index
        embeddings_np = np.array(embeddings)
        faiss.normalize_L2(embeddings_np)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)
        faiss.write_index(index, str(tmp_path / "embeddings.index"))

        # Query workflow
        query = np.random.randn(dim).astype(np.float32)
        query_norm = query.copy().reshape(1, -1)
        faiss.normalize_L2(query_norm)

        loaded_index = faiss.read_index(str(tmp_path / "embeddings.index"))
        scores, indices = loaded_index.search(query_norm, k=3)

        loaded_chunks = pq.read_table(tmp_path / "chunks.parquet")

        print("\nSchema 3: Parquet metadata + FAISS index (recommended)")
        print("-" * 50)
        for i, idx in enumerate(indices[0]):
            row = {col: loaded_chunks[col][idx].as_py() for col in ["chunk_id", "doc_id", "doc_title"]}
            print(f"  #{i+1} chunk_id={row['chunk_id']}, doc='{row['doc_title']}', score={scores[0][i]:.4f}")

        chunk_size = (tmp_path / "chunks.parquet").stat().st_size
        index_size = (tmp_path / "embeddings.index").stat().st_size
        total = chunk_size + index_size
        print(f"\nFile sizes:")
        print(f"  chunks.parquet:    {chunk_size:,} bytes")
        print(f"  embeddings.index:  {index_size:,} bytes")
        print(f"  Total:             {total:,} bytes")

        assert len(indices[0]) == 3

    def test_schema_comparison(self, tmp_path):
        """Compare storage efficiency of different schemas."""
        np.random.seed(42)
        n_docs = 50
        chunks_per_doc = 10
        dim = 1536
        n_chunks = n_docs * chunks_per_doc

        embeddings = [np.random.randn(dim).astype(np.float32) for _ in range(n_chunks)]
        texts = [f"Sample text for chunk {i}. " * 20 for i in range(n_chunks)]

        # Schema 1: JSON (baseline)
        json_data = [{"id": i, "text": texts[i], "embedding": embeddings[i].tolist()} for i in range(n_chunks)]
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps(json_data))

        # Schema 2: Single parquet with float32
        table_f32 = pa.table({
            "chunk_id": list(range(n_chunks)),
            "text": texts,
            "embedding": embeddings,
        })
        pq.write_table(table_f32, tmp_path / "chunks_f32.parquet")

        # Schema 3: Single parquet with int8
        quantized = []
        scales = []
        for emb in embeddings:
            q, s = self.quantize_to_int8(emb)
            quantized.append(q)
            scales.append(s)
        table_int8 = pa.table({
            "chunk_id": list(range(n_chunks)),
            "text": texts,
            "embedding": quantized,
            "emb_scale": scales,
        })
        pq.write_table(table_int8, tmp_path / "chunks_int8.parquet")

        # Schema 4: Parquet + FAISS
        table_meta = pa.table({
            "chunk_id": list(range(n_chunks)),
            "text": texts,
        })
        pq.write_table(table_meta, tmp_path / "chunks_meta.parquet")
        embeddings_np = np.array(embeddings)
        faiss.normalize_L2(embeddings_np)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)
        faiss.write_index(index, str(tmp_path / "faiss.index"))

        sizes = {
            "JSON": json_path.stat().st_size,
            "Parquet (float32)": (tmp_path / "chunks_f32.parquet").stat().st_size,
            "Parquet (int8)": (tmp_path / "chunks_int8.parquet").stat().st_size,
            "Parquet + FAISS": (
                (tmp_path / "chunks_meta.parquet").stat().st_size +
                (tmp_path / "faiss.index").stat().st_size
            ),
        }

        min_size = min(sizes.values())

        print(f"\n{'='*60}")
        print(f"Schema Comparison ({n_chunks} chunks, {dim}-dim embeddings)")
        print(f"{'='*60}")
        print(f"{'Schema':<25} {'Size':>12} {'Size (MB)':>12} {'Ratio':>8}")
        print(f"{'-'*60}")
        for name, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<25} {size:>12,} {size/1024/1024:>12.2f} {size/min_size:>7.1f}x")
        print(f"{'='*60}")

        assert sizes["Parquet (int8)"] < sizes["JSON"]
