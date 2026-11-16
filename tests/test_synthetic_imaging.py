from PIL import Image

from receipt_vlm.imaging import resize_max_side, to_data_uri
from receipt_vlm.schema import FIELDS
from receipt_vlm.synthetic import format_vnd, make_receipt


def test_format_vnd():
    assert format_vnd(115000) == "115.000đ"
    assert format_vnd(4500) == "4.500đ"
    assert format_vnd(1234567) == "1.234.567đ"


def test_make_receipt_is_deterministic():
    a, b = make_receipt(seed=3), make_receipt(seed=3)
    assert a.target == b.target
    assert a.image.tobytes() == b.image.tobytes()
    assert make_receipt(seed=4).target != a.target


def test_make_receipt_target_shape():
    receipt = make_receipt(seed=1)
    assert set(receipt.target) == set(FIELDS)
    assert receipt.target["total_cost"].endswith("đ")
    assert receipt.image.mode == "RGB"
    assert receipt.image.size[0] == 640


def test_resize_max_side_never_upscales():
    image = Image.new("RGB", (400, 1600))
    resized = resize_max_side(image, 800)
    assert resized.size == (200, 800)
    assert resize_max_side(image, 4000).size == (400, 1600)


def test_data_uri():
    uri = to_data_uri(Image.new("RGB", (8, 8)))
    assert uri.startswith("data:image/jpeg;base64,")
