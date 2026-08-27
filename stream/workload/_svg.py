"""A small SVG writer for workload graphs, so a picture needs no graphviz binary."""

from __future__ import annotations

import html

import networkx as nx

CHAR, LINE, PAD = 6.6, 15.0, 10.0
GAP_X, GAP_Y, MARGIN = 64.0, 22.0, 24.0
FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _size(lines: list[str]) -> tuple[float, float]:
    width = max((len(x) for x in lines), default=1) * CHAR + 2 * PAD
    return max(width, 90.0), len(lines) * LINE + 2 * PAD


def _layers(graph: nx.DiGraph) -> list[list[str]]:
    try:
        return [sorted(layer) for layer in nx.topological_generations(graph)]
    except nx.NetworkXUnfeasible:
        return [sorted(graph.nodes)]


def _place(boxes, graph):
    placed, x = {}, MARGIN
    for layer in _layers(graph):
        sizes = [_size(boxes[name][0]) for name in layer]
        column = max((w for w, _ in sizes), default=90.0)
        y = MARGIN
        for name, (w, h) in zip(layer, sizes, strict=True):
            placed[name] = (x + (column - w) / 2, y, w, h)
            y += h + GAP_Y
        x += column + GAP_X
    return placed


def _edge(one, other) -> str:
    x1, y1, w1, h1 = one
    x2, y2, _, h2 = other
    start, end = (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2)
    bend = max(18.0, (end[0] - start[0]) / 2)
    return (
        f'<path d="M{start[0]:.1f},{start[1]:.1f} C{start[0] + bend:.1f},{start[1]:.1f} '
        f'{end[0] - bend:.1f},{end[1]:.1f} {end[0]:.1f},{end[1]:.1f}" '
        'fill="none" stroke="#9aa3ad" stroke-width="1.4" marker-end="url(#a)"/>'
    )


def write_svg(path: str, boxes, edges, graph, palette) -> None:
    placed = _place(boxes, graph)
    width = max((x + w for x, _, w, _ in placed.values()), default=0) + MARGIN
    height = max((y + h for _, y, _, h in placed.values()), default=0) + MARGIN
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="{FONT}">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9aa3ad"/></marker></defs>',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fcfcfb"/>',
    ]
    out += [_edge(placed[a], placed[b]) for a, b in edges if a in placed and b in placed]
    for name, (lines, kind) in boxes.items():
        if name not in placed:
            continue
        x, y, w, h = placed[name]
        fill, stroke = palette[kind]
        dash = ' stroke-dasharray="5 3"' if kind == "state" else ""
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="9" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )
        for i, text in enumerate(lines):
            weight = "600" if i == 0 else "400"
            colour = "#0b0b0b" if i == 0 else "#52514e"
            out.append(
                f'<text x="{x + w / 2:.1f}" y="{y + PAD + LINE * i + 11:.1f}" text-anchor="middle" '
                f'font-size="{11 if i == 0 else 10}" font-weight="{weight}" fill="{colour}">'
                f"{html.escape(text)}</text>"
            )
    out.append("</svg>")
    with open(path, "w") as handle:
        handle.write("\n".join(out))
