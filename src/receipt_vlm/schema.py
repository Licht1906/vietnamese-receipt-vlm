"""Output schema, prompt and JSON parsing shared by every method in the project.

Every baseline and model must emit exactly this schema so that the same
evaluation code can score all of them.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

FIELDS: tuple[str, ...] = ("seller", "address", "timestamp", "total_cost")

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {field: _NULLABLE_STRING for field in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

PROMPT = (
    "Trích xuất thông tin từ ảnh hóa đơn/biên lai. "
    "Trả về DUY NHẤT một đối tượng JSON với đúng 4 khóa:\n"
    '- "seller": tên cửa hàng/đơn vị bán\n'
    '- "address": địa chỉ cửa hàng\n'
    '- "timestamp": ngày giờ in trên hóa đơn\n'
    '- "total_cost": tổng tiền thanh toán, giữ nguyên cách in trên hóa đơn\n'
    "Chép đúng chữ trên ảnh, giữ nguyên dấu tiếng Việt. "
    "Trường nào không có hoặc không đọc được thì để null. Không giải thích thêm."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_text(value: str | None) -> str | None:
    """NFC-normalise and collapse whitespace. Returns None for empty values."""
    if value is None:
        return None
    text = unicodedata.normalize("NFC", str(value))
    text = " ".join(text.split())
    return text or None


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} block in text, respecting JSON strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None


def parse_json_output(text: str) -> tuple[dict[str, str | None] | None, str | None]:
    """Parse a model completion into the project schema.

    Returns (prediction, error). prediction is None when the output cannot be
    parsed. Missing keys are filled with None and unknown keys are dropped, but
    either case is reported in error so JSON validity can be measured strictly.
    """
    if not text or not text.strip():
        return None, "empty_output"

    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    obj: Any = None
    for candidate in candidates:
        block = _first_json_object(candidate)
        if block is None:
            continue
        try:
            obj = json.loads(block)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(obj, dict):
        return None, "no_json_object"

    problems: list[str] = []
    missing = [field for field in FIELDS if field not in obj]
    extra = [key for key in obj if key not in FIELDS]
    if missing:
        problems.append("missing_keys:" + ",".join(missing))
    if extra:
        problems.append("extra_keys:" + ",".join(sorted(extra)))

    prediction: dict[str, str | None] = {}
    for field in FIELDS:
        value = obj.get(field)
        if value is not None and not isinstance(value, str):
            problems.append(f"non_string:{field}")
            value = str(value)
        prediction[field] = normalize_text(value)

    return prediction, (";".join(problems) or None)


def to_target_json(record: dict[str, str | None]) -> str:
    """Serialise a ground-truth record as the exact training target string."""
    ordered = {field: normalize_text(record.get(field)) for field in FIELDS}
    return json.dumps(ordered, ensure_ascii=False)


def field_exact_match(
    prediction: dict[str, str | None] | None, target: dict[str, str | None]
) -> dict[str, bool]:
    """Per-field exact match after normalisation (quick sanity metric only)."""
    if prediction is None:
        return {field: False for field in FIELDS}
    return {
        field: normalize_text(prediction.get(field)) == normalize_text(target.get(field))
        for field in FIELDS
    }
