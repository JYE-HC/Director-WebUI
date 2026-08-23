"""Single serialization authority for workflow identities and compatibility hashes."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final

from pydantic import BaseModel


MAX_SAFE_JSON_INTEGER: Final = 2**53 - 1


def _validate_canonical_string(value: str, *, context: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{context} contains an unpaired UTF-16 surrogate")
    return value


def utf16_sort_key(value: str) -> bytes:
    """Match ECMAScript string comparison by sorting UTF-16 code units."""

    _validate_canonical_string(value, context="JSON object key")
    return value.encode("utf-16-be")


def ecmascript_number_text(
    value: int | float,
    *,
    reject_negative_zero: bool,
) -> str:
    """Return the frozen ECMAScript-compatible spelling of one JSON number."""

    if isinstance(value, bool):
        raise TypeError("boolean is not a JSON number for this operation")
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise ValueError("JSON integer exceeds the interoperable safe range")
        return str(value)
    if not math.isfinite(value):
        raise ValueError("canonical JSON does not support NaN or Infinity")
    if value == 0:
        if reject_negative_zero and math.copysign(1.0, value) < 0:
            raise ValueError("canonical JSON does not support negative zero")
        return "0"

    sign = "-" if value < 0 else ""
    absolute = abs(value)
    rendered = repr(absolute).lower()
    if "e" in rendered:
        mantissa, exponent_text = rendered.split("e", 1)
        exponent = int(exponent_text)
        digits = mantissa.replace(".", "").rstrip("0")
        decimal_position = 1 + exponent
        if 1e-6 <= absolute < 1e21:
            if decimal_position <= 0:
                return f"{sign}0.{('0' * -decimal_position)}{digits}"
            if decimal_position >= len(digits):
                return f"{sign}{digits}{('0' * (decimal_position - len(digits)))}"
            return f"{sign}{digits[:decimal_position]}.{digits[decimal_position:]}"
        exponent_sign = "+" if exponent >= 0 else "-"
        exponent_digits = str(abs(exponent))
        normalized_mantissa = (
            digits if len(digits) == 1 else f"{digits[0]}.{digits[1:]}"
        )
        return f"{sign}{normalized_mantissa}e{exponent_sign}{exponent_digits}"

    if rendered.endswith(".0"):
        rendered = rendered[:-2]
    return f"{sign}{rendered}"


def _json_string(value: str, *, allow_lone_surrogates: bool) -> str:
    pieces = ['"']
    index = 0
    while index < len(value):
        character = value[index]
        codepoint = ord(character)
        if character == '"':
            pieces.append('\\"')
        elif character == "\\":
            pieces.append("\\\\")
        elif character == "\b":
            pieces.append("\\b")
        elif character == "\f":
            pieces.append("\\f")
        elif character == "\n":
            pieces.append("\\n")
        elif character == "\r":
            pieces.append("\\r")
        elif character == "\t":
            pieces.append("\\t")
        elif codepoint < 0x20:
            pieces.append(f"\\u{codepoint:04x}")
        elif 0xD800 <= codepoint <= 0xDBFF:
            paired = (
                index + 1 < len(value)
                and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF
            )
            if paired and allow_lone_surrogates:
                pieces.extend((character, value[index + 1]))
                index += 1
            elif allow_lone_surrogates:
                pieces.append(f"\\u{codepoint:04x}")
            else:
                raise ValueError("JSON string contains an unpaired UTF-16 surrogate")
        elif 0xDC00 <= codepoint <= 0xDFFF:
            if allow_lone_surrogates:
                pieces.append(f"\\u{codepoint:04x}")
            else:
                raise ValueError("JSON string contains an unpaired UTF-16 surrogate")
        else:
            pieces.append(character)
        index += 1
    pieces.append('"')
    return "".join(pieces)


def _model_or_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _canonical_json(value: Any, ancestors: set[int]) -> str:
    value = _model_or_value(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ecmascript_number_text(value, reject_negative_zero=True)
    if isinstance(value, str):
        return _json_string(value, allow_lone_surrogates=False)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("canonical JSON cannot contain reference cycles")
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("canonical JSON object keys must be strings")
        ancestors.add(identity)
        try:
            entries = (
                f"{_json_string(key, allow_lone_surrogates=False)}:"
                f"{_canonical_json(value[key], ancestors)}"
                for key in sorted(keys, key=utf16_sort_key)
            )
            return "{" + ",".join(entries) + "}"
        finally:
            ancestors.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("canonical JSON cannot contain reference cycles")
        ancestors.add(identity)
        try:
            return "[" + ",".join(
                _canonical_json(item, ancestors) for item in value
            ) + "]"
        finally:
            ancestors.remove(identity)
    raise ValueError(f"unsupported canonical JSON value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize with UTF-16 key order and ECMAScript number formatting."""

    return _canonical_json(value, set())


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _canonical_values_equal(
    left: Any,
    right: Any,
    left_ancestors: set[int],
    right_ancestors: set[int],
) -> bool:
    """Compare two valid canonical-JSON values without materializing bytes."""

    left = _model_or_value(left)
    right = _model_or_value(right)
    if left is None or isinstance(left, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            return False
        return ecmascript_number_text(
            left, reject_negative_zero=True
        ) == ecmascript_number_text(right, reject_negative_zero=True)
    if isinstance(left, str):
        _validate_canonical_string(left, context="JSON string")
        if not isinstance(right, str):
            return False
        _validate_canonical_string(right, context="JSON string")
        return left == right
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping):
            return False
        left_identity = id(left)
        right_identity = id(right)
        if left_identity in left_ancestors or right_identity in right_ancestors:
            raise ValueError("canonical JSON cannot contain reference cycles")
        left_keys = tuple(left.keys())
        right_keys = tuple(right.keys())
        if any(not isinstance(key, str) for key in (*left_keys, *right_keys)):
            raise ValueError("canonical JSON object keys must be strings")
        for key in (*left_keys, *right_keys):
            _validate_canonical_string(key, context="JSON object key")
        if len(left_keys) != len(right_keys) or set(left_keys) != set(right_keys):
            return False
        left_ancestors.add(left_identity)
        right_ancestors.add(right_identity)
        try:
            return all(
                _canonical_values_equal(
                    left[key],
                    right[key],
                    left_ancestors,
                    right_ancestors,
                )
                for key in left_keys
            )
        finally:
            left_ancestors.remove(left_identity)
            right_ancestors.remove(right_identity)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)):
            return False
        left_identity = id(left)
        right_identity = id(right)
        if left_identity in left_ancestors or right_identity in right_ancestors:
            raise ValueError("canonical JSON cannot contain reference cycles")
        if len(left) != len(right):
            return False
        left_ancestors.add(left_identity)
        right_ancestors.add(right_identity)
        try:
            return all(
                _canonical_values_equal(
                    left_item,
                    right_item,
                    left_ancestors,
                    right_ancestors,
                )
                for left_item, right_item in zip(left, right, strict=True)
            )
        finally:
            left_ancestors.remove(left_identity)
            right_ancestors.remove(right_identity)
    if right is None or isinstance(
        right, (bool, int, float, str, Mapping, list, tuple)
    ):
        return False
    raise ValueError(
        f"unsupported canonical JSON value {type(left).__name__}"
    )


def canonical_values_equal(left: Any, right: Any) -> bool:
    """Return canonical-JSON equality without building either JSON string.

    Inputs must belong to the same finite, acyclic JSON domain accepted by
    :func:`canonical_json`. Object order and list/tuple representation are
    ignored, while booleans remain distinct from numbers and numeric spelling
    follows the frozen ECMAScript rules.
    """

    return _canonical_values_equal(left, right, set(), set())


def _is_array_index(key: str) -> bool:
    if not key or not key.isascii() or not key.isdigit():
        return False
    if key != "0" and key.startswith("0"):
        return False
    number = int(key)
    return number < 2**32 - 1 and str(number) == key


def _javascript_object_keys(value: Mapping[str, Any]) -> tuple[str, ...]:
    keys = tuple(value.keys())
    if any(not isinstance(key, str) for key in keys):
        raise ValueError("JSON.stringify object keys must be strings")
    integer_keys = sorted((key for key in keys if _is_array_index(key)), key=int)
    other_keys = [key for key in keys if not _is_array_index(key)]
    return tuple(integer_keys + other_keys)


def _javascript_json_stringify(value: Any, ancestors: set[int]) -> str:
    value = _model_or_value(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return ecmascript_number_text(value, reject_negative_zero=False)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        return ecmascript_number_text(value, reject_negative_zero=False)
    if isinstance(value, str):
        return _json_string(value, allow_lone_surrogates=True)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("JSON.stringify cannot contain reference cycles")
        ancestors.add(identity)
        try:
            entries = (
                f"{_json_string(key, allow_lone_surrogates=True)}:"
                f"{_javascript_json_stringify(value[key], ancestors)}"
                for key in _javascript_object_keys(value)
            )
            return "{" + ",".join(entries) + "}"
        finally:
            ancestors.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("JSON.stringify cannot contain reference cycles")
        ancestors.add(identity)
        try:
            return "[" + ",".join(
                _javascript_json_stringify(item, ancestors) for item in value
            ) + "]"
        finally:
            ancestors.remove(identity)
    raise ValueError(f"unsupported JSON.stringify value {type(value).__name__}")


def javascript_json_stringify(value: Any) -> str:
    """Serialize the JSON domain with current JSON.stringify ordering rules."""

    return _javascript_json_stringify(value, set())


def fnv1a32_utf16(value: str) -> int:
    """Hash JavaScript UTF-16 code units exactly like App.tsx."""

    hash_value = 0x811C9DC5
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    for index in range(0, len(encoded), 2):
        code_unit = encoded[index] | (encoded[index + 1] << 8)
        hash_value ^= code_unit
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value


__all__ = [
    "MAX_SAFE_JSON_INTEGER",
    "canonical_json",
    "canonical_json_bytes",
    "ecmascript_number_text",
    "fnv1a32_utf16",
    "javascript_json_stringify",
    "utf16_sort_key",
]
