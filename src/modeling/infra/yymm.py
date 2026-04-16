from __future__ import annotations

def shift_yymm(yymm: str | int, delta_months: int) -> str:
    """Shift YYMM by delta months. Example: '2509' -> '2510' or 2509 -> '2510'."""
    yymm_str = str(yymm).zfill(4)  # Convert to string and pad with zeros if needed
    y = 2000 + int(yymm_str[:2])
    m = int(yymm_str[2:])
    total = y * 12 + (m - 1) + delta_months
    ny = total // 12
    nm = (total % 12) + 1
    return f"{str(ny)[2:]:0>2}{nm:02d}"
