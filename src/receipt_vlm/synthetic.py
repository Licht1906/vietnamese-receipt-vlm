"""Synthetic Vietnamese receipts with known ground truth.

Used only for smoke tests (Phase 0), so the pipeline can be checked end to end
before the real MC-OCR data is downloaded. Never mix these into evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SELLERS = (
    "Siêu thị Co.opmart Hà Đông",
    "Cửa hàng tiện lợi Minh Châu",
    "Nhà thuốc An Khang",
    "Bách Hóa Xanh Cầu Giấy",
    "Tạp hóa Thu Hương",
    "Cà phê Phố Cổ",
)
STREETS = ("Trần Phú", "Nguyễn Trãi", "Lê Duẩn", "Hoàng Quốc Việt", "Xuân Thủy", "Giải Phóng")
WARDS = ("phường Hà Cầu", "phường Dịch Vọng", "phường Bách Khoa", "phường Láng Hạ")
CITIES = ("Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Nha Trang")
ITEMS = (
    ("Sữa tươi Vinamilk 1L", 32000),
    ("Bánh mì sandwich", 18000),
    ("Nước suối Lavie 500ml", 6000),
    ("Mì Hảo Hảo tôm chua cay", 4500),
    ("Cà phê sữa đá", 29000),
    ("Trứng gà (vỉ 10)", 35000),
    ("Rau cải ngọt", 12000),
    ("Dầu gội Clear 340g", 98000),
)


@dataclass
class SyntheticReceipt:
    image: Image.Image
    target: dict[str, str | None]


def format_vnd(amount: int) -> str:
    """115000 -> '115.000đ' (Vietnamese thousands separator)."""
    return f"{amount:,}".replace(",", ".") + "đ"


def _find_font_path(bold: bool) -> str | None:
    """Find a TTF font with Vietnamese glyphs. DejaVu ships with matplotlib."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    try:
        from matplotlib import font_manager

        props = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
        path = font_manager.findfont(props, fallback_to_default=True)
        if path and Path(path).exists():
            return path
    except Exception:  # matplotlib missing or broken font cache
        pass
    return None


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _find_font_path(bold)
    if path is None:
        raise RuntimeError(
            "No TTF font with Vietnamese glyphs found. Install fonts-dejavu-core or "
            "matplotlib, or pass a real receipt image with --image."
        )
    return ImageFont.truetype(path, size=size)


def _wrap(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def make_receipt(seed: int = 0, width: int = 640) -> SyntheticReceipt:
    """Render a simple receipt image and return it with its ground truth."""
    rng = random.Random(seed)
    seller = rng.choice(SELLERS).upper()  # printed in capitals, and the target must match what is printed
    address = f"{rng.randint(1, 300)} {rng.choice(STREETS)}, {rng.choice(WARDS)}, {rng.choice(CITIES)}"
    when = datetime(2026, rng.randint(1, 12), rng.randint(1, 28), rng.randint(7, 22), rng.randint(0, 59))
    timestamp = when.strftime("%d/%m/%Y %H:%M")

    chosen = rng.sample(ITEMS, k=rng.randint(2, 5))
    rows = []
    total = 0
    for name, price in chosen:
        qty = rng.randint(1, 3)
        amount = qty * price
        total += amount
        rows.append((name, qty, price, amount))
    total_cost = format_vnd(total)

    title_font = _load_font(26, bold=True)
    body_font = _load_font(20)
    bold_font = _load_font(22, bold=True)
    margin = 28
    text_width = width - 2 * margin

    scratch = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    lines: list[tuple[str, object, str]] = []  # (text, font, align)
    for line in _wrap(seller, title_font, text_width, scratch):
        lines.append((line, title_font, "center"))
    for line in _wrap(address, body_font, text_width, scratch):
        lines.append((line, body_font, "center"))
    lines.append(("", body_font, "left"))
    lines.append(("HÓA ĐƠN BÁN HÀNG", bold_font, "center"))
    lines.append((f"Ngày: {timestamp}", body_font, "left"))
    lines.append(("-" * 48, body_font, "left"))
    for name, qty, price, amount in rows:
        lines.append((name, body_font, "left"))
        lines.append((f"   {qty} x {format_vnd(price)} = {format_vnd(amount)}", body_font, "left"))
    lines.append(("-" * 48, body_font, "left"))
    lines.append((f"Tổng thanh toán: {total_cost}", bold_font, "left"))
    lines.append(("", body_font, "left"))
    lines.append(("Cảm ơn quý khách!", body_font, "center"))

    line_height = 34
    height = 2 * margin + line_height * len(lines)
    image = Image.new("RGB", (width, height), (250, 249, 244))
    draw = ImageDraw.Draw(image)
    y = margin
    for text, font, align in lines:
        x = margin
        if align == "center":
            x = (width - draw.textlength(text, font=font)) / 2
        draw.text((x, y), text, fill=(25, 25, 25), font=font)
        y += line_height

    target = {"seller": seller, "address": address, "timestamp": timestamp, "total_cost": total_cost}
    return SyntheticReceipt(image=image, target=target)
