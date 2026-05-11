"""
visualization/chart.py
═══════════════════════════════════════════════════
South Indian Jathakam Chart Renderer
Produces a beautiful SVG of the traditional 4×4 grid
═══════════════════════════════════════════════════

South Indian House Layout (fixed Rasi positions):
┌──────┬──────┬──────┬──────┐
│  12  │   1  │   2  │   3  │
├──────┼──────┴──────┼──────┤
│  11  │   (center)  │   4  │
├──────┤             ├──────┤
│  10  │             │   5  │
├──────┼──────┬──────┼──────┤
│   9  │   8  │   7  │   6  │
└──────┴──────┴──────┴──────┘
"""

from typing import Dict, List, Optional


PLANET_ABBR = {
    "Surya": "Su", "Sun": "Su",
    "Chandra": "Mo", "Moon": "Mo",
    "Kuja": "Ma", "Mars": "Ma",
    "Budha": "Me", "Mercury": "Me",
    "Guru": "Ju", "Jupiter": "Ju",
    "Shukra": "Ve", "Venus": "Ve",
    "Shani": "Sa", "Saturn": "Sa",
    "Rahu": "Ra",
    "Ketu": "Ke",
    "Lagna": "La", "Asc": "La",
}

PLANET_COLORS = {
    "Su": "#FFB800", "Mo": "#90CAF9", "Ma": "#FF5252",
    "Me": "#69F0AE", "Ju": "#FFD54F", "Ve": "#F48FB1",
    "Sa": "#9FA8DA", "Ra": "#CE93D8", "Ke": "#FFAB91",
    "La": "#00E5A0",
}

# Fixed Rasi → grid cell (row, col) mapping in South Indian chart
RASI_CELL = {
    1: (0,1), 2: (0,2), 3: (0,3),
    4: (1,3), 5: (2,3), 6: (3,3),
    7: (3,2), 8: (3,1), 9: (3,0),
    10:(2,0), 11:(1,0), 12:(0,0),
}

RASI_NAMES = {
    1:"Mesha",2:"Vrishabha",3:"Mithuna",4:"Kataka",
    5:"Simha",6:"Kanya",7:"Tula",8:"Vrischika",
    9:"Dhanus",10:"Makara",11:"Kumbha",12:"Meena",
}


def render_south_indian_chart(
    planet_positions: Dict[str, int],   # planet_name → house_number (1-12)
    lagna_house: int = 1,
    title: str = "ജാതകം",
    width: int = 540,
    height: int = 540,
    theme: str = "dark",
) -> str:
    """
    Render a South Indian Jathakam chart as SVG string.

    Args:
        planet_positions: dict like {"Surya": 5, "Chandra": 9, "Rahu": 3, ...}
        lagna_house: house number of Lagna/Ascendant
        title: chart title (default: ജാതകം in Malayalam)
        width, height: SVG dimensions
        theme: "dark" | "light"
    """

    # ── Theme colors ──────────────────────────────────────
    if theme == "dark":
        bg_page   = "#0e1218"
        bg_cell   = "#141c26"
        bg_center = "#0e1218"
        border    = "#1e3050"
        text_rasi = "#4a6880"
        text_main = "#ccdde8"
        lagna_bg  = "rgba(0,229,160,0.08)"
        lagna_bd  = "#00e5a0"
    else:
        bg_page   = "#f8f4ee"
        bg_cell   = "#fff8f0"
        bg_center = "#f0e8dc"
        border    = "#c4a882"
        text_rasi = "#8a7060"
        text_main = "#2a1a0a"
        lagna_bg  = "rgba(180,120,20,0.12)"
        lagna_bd  = "#c07820"

    cell_w = width  // 4
    cell_h = height // 4

    # Group planets by house
    house_planets: Dict[int, List[str]] = {h: [] for h in range(1, 13)}
    for planet, house in planet_positions.items():
        if 1 <= house <= 12:
            abbr = PLANET_ABBR.get(planet, planet[:2])
            house_planets[house].append((planet, abbr))

    # ── Build SVG ─────────────────────────────────────────
    lines = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:\'Space Grotesk\',\'Noto Sans Malayalam\',sans-serif;background:{bg_page};border-radius:16px">',

        # Background
        f'<rect width="{width}" height="{height}" fill="{bg_page}" rx="16"/>',
    ]

    # ── Draw 16 cells ────────────────────────────────────
    for rasi_num, (row, col) in RASI_CELL.items():
        x = col * cell_w
        y = row * cell_h

        is_lagna_house = (rasi_num == lagna_house)
        fill_color = lagna_bg if is_lagna_house else bg_cell
        stroke_color = lagna_bd if is_lagna_house else border
        stroke_w = "2" if is_lagna_house else "1"

        # Cell background
        lines.append(
            f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
            f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_w}"/>'
        )

        # Rasi number (small, top-left)
        lines.append(
            f'<text x="{x+6}" y="{y+16}" font-size="11" fill="{text_rasi}" font-weight="500">'
            f'{rasi_num}</text>'
        )

        # Rasi name (tiny, below number)
        rasi_label = RASI_NAMES.get(rasi_num, "")
        lines.append(
            f'<text x="{x+6}" y="{y+28}" font-size="8.5" fill="{text_rasi}" opacity="0.7">'
            f'{rasi_label}</text>'
        )

        # Lagna marker
        if is_lagna_house:
            lines.append(
                f'<text x="{x + cell_w - 22}" y="{y+16}" font-size="10" fill="{lagna_bd}" font-weight="700">'
                f'La</text>'
            )

        # Planet abbreviations
        planets = house_planets.get(rasi_num, [])
        py_start = y + 42
        for idx, (planet_full, abbr) in enumerate(planets):
            color = PLANET_COLORS.get(abbr, text_main)
            px = x + 8
            py = py_start + idx * 18

            # Planet chip background
            lines.append(
                f'<rect x="{px-2}" y="{py-12}" width="{len(abbr)*9+8}" height="16" '
                f'rx="4" fill="{color}" opacity="0.18"/>'
            )
            lines.append(
                f'<text x="{px+2}" y="{py}" font-size="13" font-weight="700" fill="{color}">'
                f'{abbr}</text>'
            )

    # ── Center 2×2 block (rows 1-2, cols 1-2) ────────────
    cx = cell_w
    cy = cell_h
    cw = cell_w * 2
    ch = cell_h * 2

    lines.append(
        f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" '
        f'fill="{bg_center}" stroke="{border}" stroke-width="1"/>'
    )

    # Decorative mandala circle
    mx, my = cx + cw // 2, cy + ch // 2
    lines.append(
        f'<circle cx="{mx}" cy="{my}" r="{min(cw,ch)//2 - 10}" '
        f'stroke="{border}" stroke-width="1" fill="none" opacity="0.4"/>'
    )
    lines.append(
        f'<circle cx="{mx}" cy="{my}" r="{min(cw,ch)//2 - 20}" '
        f'stroke="{lagna_bd}" stroke-width="1" fill="none" opacity="0.15"/>'
    )

    # Om symbol
    lines.append(
        f'<text x="{mx}" y="{my - 18}" font-size="36" text-anchor="middle" '
        f'fill="{lagna_bd}" opacity="0.35">ॐ</text>'
    )

    # Title
    lines.append(
        f'<text x="{mx}" y="{my + 20}" font-size="16" text-anchor="middle" '
        f'font-weight="700" fill="{text_main}" opacity="0.8">{title}</text>'
    )
    lines.append(
        f'<text x="{mx}" y="{my + 40}" font-size="11" text-anchor="middle" '
        f'fill="{text_rasi}">South Indian Chart</text>'
    )

    lines.append('</svg>')
    return "\n".join(lines)


def parse_planet_positions(extracted: dict) -> dict:
    """
    Convert extracted jathakam dict to planet_positions format.
    Input: {"planets": {"Surya": {"house": 5}, "Chandra": {"house": 9}, ...}}
    Output: {"Surya": 5, "Chandra": 9, ...}
    """
    positions = {}
    planets = extracted.get("planets", {})
    for planet, info in planets.items():
        house = info.get("house") or info.get("rasi_number")
        if house and isinstance(house, int) and 1 <= house <= 12:
            positions[planet] = house
    return positions


def demo_chart() -> str:
    """Generate a demo chart for UI testing."""
    positions = {
        "Surya": 10, "Chandra": 4, "Kuja": 1,
        "Budha": 10, "Guru": 6, "Shukra": 9,
        "Shani": 12, "Rahu": 7, "Ketu": 1,
    }
    return render_south_indian_chart(positions, lagna_house=1, title="Sample ജാതകം")