import re
from dataclasses import dataclass

from knowledge_base_assistant.domain.ids import make_chunk_id, make_content_hash
from knowledge_base_assistant.domain.models import Chunk, Document, MarkdownSection

_FENCE_PATTERN = re.compile(
    r"^[ \t]{0,3}(?P<marker>`{3,}|~{3,})"
)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    max_lines: int = 40
    max_chars: int = 3_000
    overlap_lines: int = 5

    # Небольшие code fence можно полностью повторять в overlap.
    # Большие code fence повторно не добавляем.
    max_fence_overlap_lines: int = 10

    def __post_init__(self) -> None:
        if self.max_lines <= 0:
            raise ValueError("max_lines must be positive")

        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")

        if self.overlap_lines < 0:
            raise ValueError("overlap_lines must be non-negative")

        if self.overlap_lines >= self.max_lines:
            raise ValueError("overlap_lines must be smaller than max_lines")

        if self.max_fence_overlap_lines < 0:
            raise ValueError(
                "max_fence_overlap_lines must be non-negative"
            )


DEFAULT_CHUNKER_CONFIG = ChunkerConfig()


def chunk_sections(
    document: Document,
    sections: list[MarkdownSection],
    config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        if section.document_id != document.document_id:
            raise ValueError(
                "Section document_id does not match Document document_id"
            )

        section_chunks = _chunk_section_content(section, config)

        for section_chunk_index, (
            content,
            start_line,
            end_line,
        ) in enumerate(section_chunks):
            searchable_text = _make_searchable_text(
                section_path=section.section_path,
                content=content,
            )

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        document_id=document.document_id,
                        section_path=section.section_path,
                        section_chunk_index=section_chunk_index,
                        content=content,
                    ),
                    document_id=document.document_id,
                    source_name=document.source_name,
                    relative_path=document.relative_path,
                    title=(
                        section.section_path[-1]
                        if section.section_path
                        else None
                    ),
                    section_path=section.section_path,
                    content=content,
                    searchable_text=searchable_text,
                    chunk_index=chunk_index,
                    start_line=start_line,
                    section_chunk_index=section_chunk_index,
                    end_line=end_line,
                    content_hash=make_content_hash(content),
                )
            )

            chunk_index += 1

    return chunks


def _chunk_section_content(
    section: MarkdownSection,
    config: ChunkerConfig,
) -> list[tuple[str, int, int]]:
    if not section.content.strip():
        return []

    if section.content_start_line is None:
        raise ValueError("Non-empty section must have content_start_line")

    lines = section.content.splitlines()
    chunks: list[tuple[str, int, int]] = []

    start_index = 0

    while start_index < len(lines):
        end_index = _find_chunk_end(
            lines=lines,
            start_index=start_index,
            config=config,
        )

        chunk_lines = lines[start_index:end_index]

        leading_empty_lines = 0
        while (
            leading_empty_lines < len(chunk_lines)
            and not chunk_lines[leading_empty_lines].strip()
        ):
            leading_empty_lines += 1

        trailing_index = len(chunk_lines)
        while (
            trailing_index > leading_empty_lines
            and not chunk_lines[trailing_index - 1].strip()
        ):
            trailing_index -= 1

        trimmed_lines = chunk_lines[leading_empty_lines:trailing_index]

        if trimmed_lines:
            content = "\n".join(trimmed_lines)

            start_line = (
                section.content_start_line
                + start_index
                + leading_empty_lines
            )
            end_line = (
                section.content_start_line
                + start_index
                + trailing_index
                - 1
            )

            chunks.append((content, start_line, end_line))

        if end_index >= len(lines):
            break

        candidate_start = end_index - config.overlap_lines

        next_start_index = _adjust_start_for_code_fence(
            lines=lines,
            candidate_start=candidate_start,
            max_fence_overlap_lines=config.max_fence_overlap_lines,
        )

        # Защита от отсутствия продвижения вперёд.
        if next_start_index <= start_index:
            next_start_index = end_index

        start_index = next_start_index

    return chunks


def _find_chunk_end(
    lines: list[str],
    start_index: int,
    config: ChunkerConfig,
) -> int:
    end_index = start_index
    current_chars = 0

    fence_character: str | None = None
    fence_length = 0
    limit_reached = False

    while end_index < len(lines):
        line = lines[end_index]
        added_chars = len(line) + (1 if end_index > start_index else 0)

        reaches_line_limit = (
            end_index - start_index >= config.max_lines
        )
        exceeds_char_limit = (
            current_chars + added_chars > config.max_chars
        )

        if (
            reaches_line_limit or exceeds_char_limit
        ) and end_index > start_index:
            limit_reached = True

            # Вне code fence можно закончить chunk сразу.
            if fence_character is None:
                break

        fence_character, fence_length = _update_fence_state(
            line=line,
            fence_character=fence_character,
            fence_length=fence_length,
        )

        current_chars += added_chars
        end_index += 1

        # Лимит уже был достигнут, а code fence только что закрылся.
        if limit_reached and fence_character is None:
            break

    return end_index


def _adjust_start_for_code_fence(
    lines: list[str],
    candidate_start: int,
    max_fence_overlap_lines: int,
) -> int:
    """
    Корректирует начало следующего chunk, если overlap попал в code fence.

    Для короткого fence возвращает индекс его открытия, чтобы повторить
    весь блок в следующем chunk.

    Для длинного fence возвращает строку после его закрытия, чтобы не
    дублировать большой блок кода.
    """
    fence_character: str | None = None
    fence_length = 0
    opening_index: int | None = None

    for index, line in enumerate(lines):
        previous_character = fence_character

        fence_character, fence_length = _update_fence_state(
            line=line,
            fence_character=fence_character,
            fence_length=fence_length,
        )

        # Открылся новый code fence.
        if previous_character is None and fence_character is not None:
            opening_index = index
            continue

        # Закрылся текущий code fence.
        if (
            previous_character is not None
            and fence_character is None
            and opening_index is not None
        ):
            closing_index = index

            # candidate_start попал на любую строку fence,
            # включая открывающую и закрывающую строки.
            if opening_index <= candidate_start <= closing_index:
                fence_lines = closing_index - opening_index + 1

                if fence_lines <= max_fence_overlap_lines:
                    return opening_index

                return closing_index + 1

            opening_index = None

        # Все fence до candidate_start уже проверены.
        if index > candidate_start and fence_character is None:
            break

    return candidate_start


def _update_fence_state(
    line: str,
    fence_character: str | None,
    fence_length: int,
) -> tuple[str | None, int]:
    match = _FENCE_PATTERN.match(line)

    if match is None:
        return fence_character, fence_length

    marker = match.group("marker")

    if fence_character is None:
        return marker[0], len(marker)

    if marker[0] == fence_character and len(marker) >= fence_length:
        return None, 0

    return fence_character, fence_length


def _make_searchable_text(
    section_path: tuple[str, ...],
    content: str,
) -> str:
    if not section_path:
        return content

    heading_context = " > ".join(section_path)
    return f"{heading_context}\n\n{content}"
