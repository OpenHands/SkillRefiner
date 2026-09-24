"""Context-budget helpers for trace-summary refinement prompts."""

try:
    import litellm
except Exception:  # pragma: no cover - fallback path for minimal installs
    litellm = None

_FALLBACK_CONTEXT = 16_000
OUTPUT_RESERVE = 6_000


def context_window(model: str) -> int:
    """Return the model input-token limit, falling back to a conservative default."""
    if litellm is None:
        return _FALLBACK_CONTEXT
    try:
        info = litellm.get_model_info(model)
        return int(info.get("max_input_tokens") or _FALLBACK_CONTEXT)
    except Exception:
        return _FALLBACK_CONTEXT


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in text, falling back to an approximate character heuristic."""
    if litellm is None:
        return max(1, len(text) // 4)
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return max(1, len(text) // 4)


def pack_sequential(
    lines: list[str],
    token_budget: int,
    model: str = "",
    min_items: int = 3,
) -> int:
    """Greedily include leading lines without exceeding the token budget.

    ``min_items`` is a caller requirement, not permission to over-pack: if fewer
    than ``min_items`` fit, the actual fitting count is returned so callers can
    decide whether to decline, truncate, or use a smaller prompt.
    """
    if token_budget <= 0:
        return 0

    spent = 0
    n = 0
    for line in lines:
        cost = count_tokens(line, model)
        if spent + cost > token_budget:
            break
        n += 1
        spent += cost
    return n


def truncate_to_tokens(text: str, max_tokens: int, model: str = "") -> str:
    """Truncate text from the end until it fits within max_tokens."""
    if max_tokens <= 0:
        return ""

    tokens = count_tokens(text, model)
    while tokens > max_tokens and text:
        keep = max(1, int(len(text) * max_tokens / tokens) - 64)
        if keep >= len(text):
            keep = len(text) - 1
        text = text[:keep]
        tokens = count_tokens(text, model)
    return text
