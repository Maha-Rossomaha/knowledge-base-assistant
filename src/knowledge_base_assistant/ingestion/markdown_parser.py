import re
from dataclasses import dataclass

from knowledge_base_assistant.domain.models import Document, MarkdownSection

_HEADING_PATTERN = re.compile(
    r"^(?P<marks>#{1,6})[ \t]+(?P<title>.*?)[ \t]*#*[ \t]*$"
)

_FENCE_PATTERN = re.compile(
    r"^[ \t]{0,3}(?P<marker>`{3,}|~{3,})"
)


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    title: str


def parse_markdown_sections(document: Document) -> list[MarkdownSection]:
    lines = document.content.splitlines()

    if not lines:
        return []

    sections: list[MarkdownSection] = []
    heading_stack: list[_Heading] = []

    current_section_path: tuple[str, ...] = ()
    current_start_line = 1
    current_content_lines: list[str] = []

    fence_character: str | None = None
    fence_length = 0

    for line_number, line in enumerate(lines, start=1):
        fence_match = _FENCE_PATTERN.match(line)

        if fence_match is not None:
            marker = fence_match.group("marker")

            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0

            current_content_lines.append(line)
            continue

        if fence_character is not None:
            current_content_lines.append(line)
            continue

        heading_match = _HEADING_PATTERN.match(line)

        if heading_match is None:
            current_content_lines.append(line)
            continue

        sections.append(
            _make_section(
                document=document,
                section_path=current_section_path,
                content_lines=current_content_lines,
                start_line=current_start_line,
                end_line=line_number - 1,
            )
        )

        level = len(heading_match.group("marks"))
        title = heading_match.group("title").strip()

        heading_stack = [
            heading for heading in heading_stack if heading.level < level
        ]
        heading_stack.append(_Heading(level=level, title=title))

        current_section_path = tuple(
            heading.title for heading in heading_stack
        )
        current_start_line = line_number
        current_content_lines = []

    sections.append(
        _make_section(
            document=document,
            section_path=current_section_path,
            content_lines=current_content_lines,
            start_line=current_start_line,
            end_line=len(lines),
        )
    )

    return [
        section
        for section in sections
        if section.section_path or section.content
    ]


def _make_section(
    document: Document,
    section_path: tuple[str, ...],
    content_lines: list[str],
    start_line: int,
    end_line: int,
) -> MarkdownSection:
    first_content_line = start_line + 1 if section_path else start_line

    leading_empty_lines = 0
    while (
        leading_empty_lines < len(content_lines)
        and not content_lines[leading_empty_lines].strip()
    ):
        leading_empty_lines += 1

    trailing_index = len(content_lines)
    while (
        trailing_index > leading_empty_lines
        and not content_lines[trailing_index - 1].strip()
    ):
        trailing_index -= 1

    trimmed_lines = content_lines[leading_empty_lines:trailing_index]

    if trimmed_lines:
        content_start_line = first_content_line + leading_empty_lines
        section_end_line = content_start_line + len(trimmed_lines) - 1
    else:
        content_start_line = None
        section_end_line = start_line

    return MarkdownSection(
        document_id=document.document_id,
        section_path=section_path,
        content="\n".join(trimmed_lines),
        start_line=start_line,
        content_start_line=content_start_line,
        end_line=min(section_end_line, end_line),
    )