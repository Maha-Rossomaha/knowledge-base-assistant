import math
from collections import Counter
from collections.abc import Sequence

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.models import SearchResult
from knowledge_base_assistant.retrieval.tokenizer import tokenize


class BM25Index:
    def __init__(
        self,
        *,
        chunks: tuple[Chunk, ...],
        term_frequencies: tuple[Counter[str], ...],
        document_frequencies: dict[str, int],
        document_lengths: tuple[int, ...],
        average_document_length: float,
        k1: float,
        b: float,
    ) -> None:
        self._chunks = chunks
        self._term_frequencies = term_frequencies
        self._document_frequencies = document_frequencies
        self._document_lengths = document_lengths
        self._average_document_length = average_document_length
        self._k1 = k1
        self._b = b

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        _validate_parameters(
            k1=k1,
            b=b,
        )

        stored_chunks = tuple(chunks)

        term_frequencies: list[Counter[str]] = []
        document_lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for chunk in stored_chunks:
            tokens = tokenize(chunk.searchable_text)
            frequencies = Counter(tokens)

            term_frequencies.append(frequencies)
            document_lengths.append(len(tokens))

            document_frequencies.update(
                frequencies.keys()
            )

        average_document_length = (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 0.0
        )

        return cls(
            chunks=stored_chunks,
            term_frequencies=tuple(term_frequencies),
            document_frequencies=dict(document_frequencies),
            document_lengths=tuple(document_lengths),
            average_document_length=average_document_length,
            k1=k1,
            b=b,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError(
                f"top_k must be at least 1, got {top_k}"
            )

        query_tokens = tokenize(query)

        if not query_tokens or not self._chunks:
            return []

        # Повтор одного слова в запросе пока не увеличивает его вес.
        unique_query_terms = tuple(
            dict.fromkeys(query_tokens)
        )

        scored_chunks: list[tuple[int, float]] = []

        for document_index, frequencies in enumerate(
            self._term_frequencies
        ):
            score = self._score_document(
                document_index=document_index,
                frequencies=frequencies,
                query_terms=unique_query_terms,
            )

            if score > 0.0:
                scored_chunks.append(
                    (document_index, score)
                )

        scored_chunks.sort(
            key=lambda item: (
                -item[1],
                self._chunks[item[0]].chunk_id,
            )
        )

        return [
            SearchResult(
                chunk=self._chunks[document_index],
                score=score,
                rank=rank,
            )
            for rank, (document_index, score) in enumerate(
                scored_chunks[:top_k],
                start=1,
            )
        ]

    def _score_document(
        self,
        *,
        document_index: int,
        frequencies: Counter[str],
        query_terms: tuple[str, ...],
    ) -> float:
        document_length = self._document_lengths[
            document_index
        ]

        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            document_frequency = (
                self._document_frequencies.get(term, 0)
            )

            inverse_document_frequency = self._idf(
                document_frequency
            )

            denominator = (
                term_frequency
                + self._k1
                * (
                    1.0
                    - self._b
                    + self._b
                    * self._length_ratio(document_length)
                )
            )

            score += (
                inverse_document_frequency
                * term_frequency
                * (self._k1 + 1.0)
                / denominator
            )

        return score

    def _idf(
        self,
        document_frequency: int,
    ) -> float:
        document_count = len(self._chunks)

        return math.log(
            1.0
            + (
                document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def _length_ratio(
        self,
        document_length: int,
    ) -> float:
        if self._average_document_length == 0.0:
            return 0.0

        return (
            document_length
            / self._average_document_length
        )


def _validate_parameters(
    *,
    k1: float,
    b: float,
) -> None:
    if k1 <= 0.0:
        raise ValueError(
            f"k1 must be greater than 0, got {k1}"
        )

    if not 0.0 <= b <= 1.0:
        raise ValueError(
            f"b must be between 0 and 1, got {b}"
        )