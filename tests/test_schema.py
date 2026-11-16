import json
import unicodedata

from receipt_vlm.schema import (
    FIELDS,
    JSON_SCHEMA,
    field_exact_match,
    normalize_text,
    parse_json_output,
    to_target_json,
)

GOOD = {"seller": "Bách Hóa Xanh", "address": "12 Trần Phú", "timestamp": "09/08/2020 07:44", "total_cost": "115.000đ"}


def test_parse_plain_json():
    pred, err = parse_json_output(json.dumps(GOOD, ensure_ascii=False))
    assert err is None
    assert pred == GOOD


def test_parse_fenced_json_with_chatter():
    text = "Đây là kết quả:\n```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```\nXong."
    pred, err = parse_json_output(text)
    assert err is None
    assert pred == GOOD


def test_parse_braces_inside_strings():
    record = dict(GOOD, seller="Shop {Vui} }")
    pred, err = parse_json_output("prefix " + json.dumps(record, ensure_ascii=False) + " suffix")
    assert err is None
    assert pred["seller"] == "Shop {Vui} }"


def test_missing_and_extra_keys_are_reported():
    pred, err = parse_json_output('{"seller": "A", "phone": "0123"}')
    assert pred == {"seller": "A", "address": None, "timestamp": None, "total_cost": None}
    assert "missing_keys:address,timestamp,total_cost" in err
    assert "extra_keys:phone" in err


def test_non_string_value_is_reported():
    pred, err = parse_json_output('{"seller": "A", "address": null, "timestamp": null, "total_cost": 115000}')
    assert pred["total_cost"] == "115000"
    assert "non_string:total_cost" in err


def test_invalid_outputs():
    assert parse_json_output("") == (None, "empty_output")
    assert parse_json_output("không có json") == (None, "no_json_object")
    assert parse_json_output('{"seller": "A",') == (None, "no_json_object")


def test_nfc_normalisation_makes_nfd_match():
    nfd = unicodedata.normalize("NFD", "Hóa đơn   tiếng  Việt ")
    assert normalize_text(nfd) == "Hóa đơn tiếng Việt"
    pred = dict(GOOD, seller=unicodedata.normalize("NFD", GOOD["seller"]))
    assert all(field_exact_match(pred, GOOD).values())


def test_exact_match_none_prediction():
    assert field_exact_match(None, GOOD) == {field: False for field in FIELDS}


def test_target_json_roundtrip_and_key_order():
    text = to_target_json({"total_cost": "1đ", "seller": "A"})
    assert list(json.loads(text)) == list(FIELDS)
    pred, err = parse_json_output(text)
    assert err is None and pred["address"] is None


def test_schema_covers_fields():
    assert set(JSON_SCHEMA["properties"]) == set(FIELDS)
    assert JSON_SCHEMA["required"] == list(FIELDS)
