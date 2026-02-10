import csv
import json
import logging
import os
from io import StringIO
from typing import List, Optional, Union

import numpy as np
from dotenv import load_dotenv
from microeval.llm import get_llm_client, load_config
from path import Path
from pydash import py_

model_config = load_config()
embed_models = model_config["embed_models"]

logger = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"


class RAGService:
    """Service class for Retrieval-Augmented Generation functionality."""

    def __init__(self, llm_service: Optional[str] = None):
        self.llm_service = llm_service or os.getenv("LLM_SERVICE", "openai").lower()

        model = (
            os.getenv("EMBED_MODEL")
            or os.getenv(f"{self.llm_service.upper()}_MODEL")
            or embed_models.get(self.llm_service)
        )
        if model is None:
            raise ValueError(f"Unsupported service: {self.llm_service}")

        self.embed_client = get_llm_client(self.llm_service, model=model)
        self.embed_json = f"embeddings-{py_.kebab_case(model)}.json"
        logger.info(f"RAG LLM Service: '{llm_service}:{model}'")

        self.speakers_with_embeddings: Optional[List[dict]] = None
        self.speakers: Optional[List[dict]] = None

    async def __aenter__(self):
        await self.embed_client.connect()
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.embed_client.close()
        return False

    async def connect(self):
        if self.speakers_with_embeddings and self.speakers:
            return
        elif self.is_exists(self.embed_json):
            self.speakers_with_embeddings = json.loads(
                self.read_text_file(self.embed_json)
            )
        else:
            logger.info(
                f"Generating embeddings with '{self.llm_service}:{self.embed_client.model}'"
            )
            self.speakers_with_embeddings = await self._generate_speaker_embeddings()
            self.save_text_file(
                json.dumps(self.speakers_with_embeddings, indent=2), self.embed_json
            )
            logger.info(f"Embeddings saved to '{self.embed_json}'")
        self.speakers = py_.map(self.speakers_with_embeddings, self._strip_embeddings)

    def _resolve_data_path(self, file_path: Union[Path, str]) -> Path:
        path = Path(file_path)
        return path if path.isabs() else data_dir / path

    def is_exists(self, file_path: Union[Path, str]) -> bool:
        return self._resolve_data_path(file_path).exists()

    def read_text_file(self, file_path: Union[Path, str]) -> str:
        return self._resolve_data_path(file_path).read_text()

    def save_text_file(self, text: str, file_path: Union[Path, str]):
        self._resolve_data_path(file_path).write_text(text)

    async def _generate_speaker_embeddings(self) -> List[dict]:
        csv_text = self.read_text_file("2025-09-02-speaker-bio.csv")
        csv_reader = csv.DictReader(StringIO(csv_text))
        speakers = [dict(row) for row in csv_reader]
        logger.info(f"Generating embeddings for {len(speakers)} speakers")

        result = []
        for speaker in speakers:
            speaker = py_.map_keys(speaker, lambda v, k: py_.snake_case(k))
            logger.info(f"Getting text embeddings for '{speaker['name']}'")
            for embed_key, field in [
                ("abstract_embedding", "final_abstract_max_150_words"),
                ("bio_embedding", "bio_max_120_words"),
            ]:
                speaker[embed_key] = await self.embed_client.embed(speaker[field])
            result.append(speaker)
            logger.info(f"Finished text embeddings for '{speaker['name']}'")
        return result

    @staticmethod
    def cosine_distance(
        vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]
    ) -> float:
        """Return cosine distance between 2 vectors, 0 being the best match."""
        a = np.asarray(vec1, dtype=np.float64)
        b = np.asarray(vec2, dtype=np.float64)

        if a.size != b.size:
            raise ValueError(
                f"Vectors must be of the same length (got {a.size} and {b.size})"
            )
        if a.size == 0:
            raise ValueError("Vectors cannot be empty")

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("One or both vectors have zero magnitude")

        cosine_similarity = dot_product / (norm_a * norm_b)
        return 1.0 - cosine_similarity

    @staticmethod
    def _strip_embeddings(speaker: dict) -> dict:
        return {k: v for k, v in speaker.items() if "embedding" not in k}

    def get_speaker_distance(self, embedding, speaker: dict) -> float:
        if "abstract_embedding" in speaker and "bio_embedding" in speaker:
            d1 = self.cosine_distance(embedding, speaker["abstract_embedding"])
            d2 = self.cosine_distance(embedding, speaker["bio_embedding"])
            return (d1 + d2) / 2
        else:
            return float("inf")

    async def get_best_speaker(self, query: str) -> Optional[dict]:
        await self.connect()
        embedding = await self.embed_client.embed(query)
        distances = [
            self.get_speaker_distance(embedding, s)
            for s in self.speakers_with_embeddings
        ]
        i_speaker_best = int(np.argmin(distances))
        return self.speakers[i_speaker_best]

    async def get_speakers(self) -> List[dict]:
        await self.connect()
        return self.speakers


async def build_embeddings():
    """Run embeddings generation."""
    load_dotenv()
    async with RAGService():
        pass


async def search_loop():
    """Interactive search loop - query the RAG database directly."""
    load_dotenv()
    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    if not service:
        print("Error: EMBED_SERVICE or CHAT_SERVICE environment variable is not set")
        return

    async with RAGService(llm_service=service) as rag_service:
        print("RAG Search - Find the best matching speaker")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                query = input("Query: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not query.strip():
                continue

            speaker = await rag_service.get_best_speaker(query)
            if speaker:
                print("\n--- Best Match ---")
                print(f"Name: {speaker.get('name', 'N/A')}")
                print(f"Bio: {speaker.get('bio_max_120_words', 'N/A')}")
                print(f"Abstract: {speaker.get('final_abstract_max_150_words', 'N/A')}")
                print("")
            else:
                print("No match found.\n")
