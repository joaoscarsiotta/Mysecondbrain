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
    segments = _split_by_separators(text)

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


def _split_by_separators(text: str) -> list[str]:
    """Divide texto mantendo os separadores junto aos segmentos."""
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            # Reunir separador com cada parte (exceto a última)
            result = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    result.append(part + sep)
                else:
                    result.append(part)
            return result

    # Sem separador encontrado, retornar texto inteiro
    return [text]
