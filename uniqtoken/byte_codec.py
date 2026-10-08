from __future__ import annotations

import re
from typing import List


def validate_dropout_prob(dropout_prob: float) -> None:
    """Validate a BPE/SuperBPE dropout probability (Provilkov et al., 2020).

    Single shared definition used by ``tokenizer``, ``bpe_model`` and
    ``batch_collator``. Raises ``ValueError`` for anything outside
    ``[0.0, 1.0)``, including non-numeric values such as strings and
    booleans (``bool`` is an ``int`` subclass, so it is rejected
    explicitly before the numeric range check).
    """
    if isinstance(dropout_prob, bool) or not isinstance(dropout_prob, (int, float)) or not (0.0 <= dropout_prob < 1.0):
        raise ValueError(f"dropout_prob must be in range [0.0, 1.0), got {dropout_prob!r}")


BYTE_TOKEN_TO_VAL: dict[str, int] = {f"<0x{b:02X}>": b for b in range(256)}
BYTE_TOKEN_TO_VAL.update({f"<0x{b:02x}>": b for b in range(256)})
BYTE_VAL_TO_TOKEN: tuple[str, ...] = tuple(f"<0x{b:02X}>" for b in range(256))


class ByteFallbackEngine:
    """
    Executable Byte-Fallback codec.

    Bridges the gap between raw UTF-8 bytes and token representations (<0x00> through <0xFF>).
    Guarantees executable OOV recovery and lossless roundtripping for valid UTF-8
    byte sequences produced by ``char_to_byte_tokens``.
    """

    BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

    @classmethod
    def is_byte_token(cls, token: str) -> bool:
        if len(token) != 6 or not token.startswith("<0x") or token[-1] != ">":
            return False
        return token in BYTE_TOKEN_TO_VAL

    @classmethod
    def byte_to_token(cls, byte_val: int) -> str:
        if not 0 <= byte_val <= 255:
            raise ValueError(f"Byte value must be in range 0-255, got {byte_val}")
        return BYTE_VAL_TO_TOKEN[byte_val]

    @classmethod
    def token_to_byte(cls, token: str) -> int:
        byte_val = BYTE_TOKEN_TO_VAL.get(token)
        if byte_val is None:
            raise ValueError(f"Token {token!r} is not a valid byte fallback token")
        return byte_val

    @classmethod
    def char_to_byte_tokens(cls, char_or_str: str) -> List[str]:
        """
        Converts an un-embedded or OOV string into a sequence of byte tokens.
        Uses ``surrogateescape`` so raw binary decoded with Python's
        ``errors='surrogateescape'`` (common for POSIX file reads) maps its lone
        surrogate chars (U+DC80..DCFF) to the exact original bytes.
        """
        if not isinstance(char_or_str, str):
            raise TypeError(f"char_or_str must be a string, got {type(char_or_str).__name__}")
        try:
            # Preserve POSIX surrogateescape bytes, while rejecting all other
            # unpaired surrogates deterministically instead of leaking an
            # implementation-dependent UnicodeEncodeError message.
            raw_bytes = char_or_str.encode("utf-8", errors="surrogateescape")
        except UnicodeEncodeError as exc:
            raise ValueError("input contains an unpaired surrogate outside the surrogateescape byte range") from exc
        byte_to_tok = BYTE_VAL_TO_TOKEN
        return [byte_to_tok[b] for b in raw_bytes]

    @classmethod
    def decode_tokens(cls, tokens: List[str], space_char: str = "\u2581") -> str:
        """
        Reconstructs a human-readable string from a stream of subwords and byte fallback tokens.
        Accumulates adjacent byte tokens and decodes them as UTF-8 sequences.
        Invalid byte sequences raise UnicodeDecodeError rather than silently
        replacing data. Metaspace conversion applies only to learned subwords,
        never to text reconstructed from byte fallback tokens.
        """
        output_segments: List[str] = []
        byte_buffer = bytearray()
        token_to_val = BYTE_TOKEN_TO_VAL

        for tok in tokens:
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">") and tok in token_to_val:
                byte_buffer.append(token_to_val[tok])
            else:
                if byte_buffer:
                    output_segments.append(byte_buffer.decode("utf-8"))
                    byte_buffer.clear()
                if space_char in tok:
                    output_segments.append(tok.replace(space_char, " "))
                else:
                    output_segments.append(tok)

        if byte_buffer:
            output_segments.append(byte_buffer.decode("utf-8"))

        return "".join(output_segments)
