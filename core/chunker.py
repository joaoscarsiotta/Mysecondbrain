"""Divisão de texto em chunks com sobreposição baseada em tokens."""

import tiktoken


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    model: str = "cl100k_base",
) -> list[dict]:
    """Divide texto em chunks respeitando limites de tokens.

    Usa separadores hierárquicos: \\n\\n -> \\n -> '. ' -> ' '
    """
    encoder = tiktoken.get_encoding(model)
    tokens = encoder.encode(text)

    if len(tokens) <= chunk_size:
        return [{"text": text.strip(), "chunk_index": 0}]

    # Dividir por separadores hierárquicos para manter coerência
    segments = _split_by_separators(text, chunk_size, encoder)

    chunks = []
    current_tokens = []
    current_text_parts = []

    for segment in segments:
        seg_tokens = encoder.encode(segment)

        if len(current_tokens) + len(seg_tokens) > chunk_size and current_tokens:
            # Salvar chunk atual
            chunk_text_str = "".join(current_text_parts).strip()
            if chunk_text_str:
                chunks.append({"text": chunk_text_str, "chunk_index": len(chunks)})

            # Manter overlap
            overlap_text = encoder.decode(current_tokens[-chunk_overlap:]) if len(current_tokens) > chunk_overlap else "".join(current_text_parts)
            current_tokens = encoder.encode(overlap_text)
            current_text_parts = [overlap_text]

        current_tokens.extend(seg_tokens)
        current_text_parts.append(segment)

    # Último chunk
    remaining = "".join(current_text_parts).strip()
    if remaining:
        chunks.append({"text": remaining, "chunk_index": len(chunks)})

    return chunks


def _split_by_separators(
    text: str,
    chunk_size: int,
    encoder: "tiktoken.Encoding",
    _sep_index: int = 0,
) -> list[str]:
    """Divide texto recursivamente usando separadores hierárquicos.

    Se um segmento ainda excede chunk_size após a divisão, tenta o
    próximo separador na hierarquia.
    """
    separators = ["\n\n", "\n", ". ", " "]

    if _sep_index >= len(separators):
        return [text]

    sep = separators[_sep_index]
    if sep not in text:
        return _split_by_separators(text, chunk_size, encoder, _sep_index + 1)

    parts = text.split(sep)
    # Reunir separador com cada parte (exceto a última)
    raw_segments = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            raw_segments.append(part + sep)
        else:
            raw_segments.append(part)

    # Recursivamente dividir segmentos que ainda excedem chunk_size
    result = []
    for segment in raw_segments:
        if len(encoder.encode(segment)) > chunk_size:
            result.extend(
                _split_by_separators(segment, chunk_size, encoder, _sep_index + 1)
            )
        else:
            result.append(segment)

    return result
