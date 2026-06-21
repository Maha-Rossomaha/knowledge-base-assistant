import re

_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[_-][^\W_]+)*",
    flags=re.UNICODE,
)


def tokenize(text: str) -> tuple[str, ...]:
    normalized_text = text.casefold()

    return tuple(
        _TOKEN_PATTERN.findall(normalized_text)
    )