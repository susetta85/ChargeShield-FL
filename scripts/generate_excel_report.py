#!/usr/bin/env python3
"""
scripts/generate_excel_report.py
ChargeShield-FL — Genera report Excel da tutti i risultati in experiments/

Legge tutti i file experiment_*.json e produce:
  Sheet "Raw Data"   — ogni esperimento su una riga
  Sheet "Heat Map"   — matrice AUC-ROC: righe=round, colonne=epsilon
  Sheet "Per Rounds" — AUC-ROC medio, min, max per numero di round
  Sheet "Per Epsilon"— AUC-ROC medio, min, max per valore di epsilon

Usage:
  python scripts/generate_excel_report.py
  python scripts/generate_excel_report.py --output experiments/my_report.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Side
    )
    from openpyxl.utils import get_column_letter
except ImportError:
    print("ERROR: openpyxl non trovato. Installa con: pip install openpyxl")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ── Palette colori ChargeShield ────────────────────────────────────────────────
COLOR_HEADER_BG  = "1F4E79"   # blu scuro
COLOR_HEADER_FG  = "FFFFFF"   # bianco
COLOR_SUBHDR_BG  = "2E75B6"   # blu medio
COLOR_ROW_ALT    = "D6E4F0"   # azzurro chiaro (righe alternate)
COLOR_ROW_PLAIN  = "FFFFFF"
COLOR_ACCENT     = "FF6B35"   # arancione accent (AUC-ROC sopra 0.5)
COLOR_GOOD       = "70AD47"   # verde (LOW risk)
COLOR_WARN       = "FFC000"   # giallo (MEDIUM risk)
COLOR_BAD        = "FF0000"   # rosso (HIGH risk)
COLOR_RANDOM     = "BDD7EE"   # celeste (≈ random, AUC ≈ 0.5)

FONT_NAME = "Arial"

# ── Helpers stile ──────────────────────────────────────────────────────────────

def _font(bold=False, color="000000", size=10):
    return Font(name=FONT_NAME, bold=bold, color=color, size=size)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _border():
    s = Side(style="thin", color="BFBFBF")
    return Border(left=s, right=s, top=s, bottom=s)

def _center():
    return Alignment(horizontal="center", vertical="center")

def _right():
    return Alignment(horizontal="right", vertical="center")

def _header_cell(cell, text, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG, size=10):
    cell.value = text
    cell.font = _font(bold=True, color=fg, size=size)
    cell.fill = _fill(bg)
    cell.alignment = _center()
    cell.border = _border()

def _data_cell(cell, value, fmt=None, alt_row=False, bold=False):
    cell.value = value
    cell.font = _font(bold=bold)
    cell.fill = _fill(COLOR_ROW_ALT if alt_row else COLOR_ROW_PLAIN)
    cell.border = _border()
    cell.alignment = _center()
    if fmt:
        cell.number_format = fmt

def _set_col_width(ws, col_letter, width):
    ws.column_dimensions[col_letter].width = width

# ── Carica dati ────────────────────────────────────────────────────────────────

def load_experiments(experiments_dir: Path | None = None) -> list[dict]:
    """Legge tutti i file experiment_*.json in experiments/ e li ordina."""
    src = experiments_dir if experiments_dir is not None else EXPERIMENTS_DIR
    records = []
    for path in sorted(src.glob("experiment_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg   = data.get("config", {})
            summ  = data.get("summary", {})
            records.append({
                "timestamp":   data.get("timestamp", path.stem.replace("experiment_", "")),
                "file":        path.name,
                "rounds":      int(cfg.get("fl_rounds", 0)),
                "epsilon":     float(cfg.get("epsilon", 0)),
                "delta":       float(cfg.get("delta", 1e-5)),
                "proximal_mu": float(cfg.get("proximal_mu", 0)),
                "auc_roc":     float(summ["mean_auc_roc"]) if summ.get("mean_auc_roc") is not None else None,
                "auc_max":     float(summ["max_auc_roc"])  if summ.get("max_auc_roc")  is not None else None,
                "auc_min":     float(summ["min_auc_roc"])  if summ.get("min_auc_roc")  is not None else None,
                "privacy_risk":summ.get("privacy_risk", ""),
                # IDS: conta gli alert totali nei round
                "total_alerts": sum(
                    len(v.get("ids", {}).get("alerts", []))
                    for v in data.get("per_round", {}).values()
                ),
                "byzantine_rounds": sum(
                    1 for v in data.get("per_round", {}).values()
                    if v.get("ids", {}).get("byzantine_detected")
                ),
            })
        except Exception as e:
            print(f"WARN: impossibile leggere {path.name}: {e}")
    return records


# ── Sheet 1: Raw Data ──────────────────────────────────────────────────────────

def build_raw_data(ws, records: list[dict]) -> None:
    ws.title = "Raw Data"

    # Titolo
    ws.merge_cells("A1:K1")
    title = ws["A1"]
    title.value = "ChargeShield-FL — Experiment Results (Full Sweep)"
    title.font = _font(bold=True, color=COLOR_HEADER_FG, size=12)
    title.fill = _fill(COLOR_HEADER_BG)
    title.alignment = _center()

    headers = [
        "Timestamp", "FL Rounds", "Epsilon (ε)", "Delta (δ)", "Proximal μ",
        "AUC-ROC (mean)", "AUC-ROC (max)", "AUC-ROC (min)",
        "Privacy Risk", "IDS Alerts", "Byzantine Rounds",
    ]
    for col, h in enumerate(headers, 1):
        _header_cell(ws.cell(2, col), h, bg=COLOR_SUBHDR_BG)

    for row_idx, rec in enumerate(records, 3):
        alt = (row_idx % 2 == 0)
        _data_cell(ws.cell(row_idx, 1),  rec["timestamp"],    alt_row=alt)
        _data_cell(ws.cell(row_idx, 2),  rec["rounds"],       alt_row=alt)
        _data_cell(ws.cell(row_idx, 3),  rec["epsilon"],      fmt="0.0#", alt_row=alt)
        _data_cell(ws.cell(row_idx, 4),  rec["delta"],        fmt="0.00E+00", alt_row=alt)
        _data_cell(ws.cell(row_idx, 5),  rec["proximal_mu"],  fmt="0.00", alt_row=alt)

        # AUC-ROC colorato
        for col, key in zip([6, 7, 8], ["auc_roc", "auc_max", "auc_min"]):
            c = ws.cell(row_idx, col)
            val = rec[key]
            _data_cell(c, val, fmt="0.0000", alt_row=alt)
            if val is not None:
                if val > 0.55:
                    c.font = _font(bold=True, color=COLOR_BAD)
                elif val > 0.50:
                    c.font = _font(bold=True, color=COLOR_WARN)

        # Privacy risk con colore
        risk_cell = ws.cell(row_idx, 9)
        risk = rec["privacy_risk"]
        _data_cell(risk_cell, risk, alt_row=alt, bold=True)
        if risk == "HIGH":
            risk_cell.font = _font(bold=True, color=COLOR_BAD)
        elif risk == "MEDIUM":
            risk_cell.font = _font(bold=True, color=COLOR_WARN)
        else:
            risk_cell.font = _font(bold=True, color=COLOR_GOOD)

        _data_cell(ws.cell(row_idx, 10), rec["total_alerts"],    alt_row=alt)
        _data_cell(ws.cell(row_idx, 11), rec["byzantine_rounds"], alt_row=alt)

    # Larghezze colonne
    widths = [20, 12, 12, 14, 12, 16, 14, 14, 14, 12, 18]
    for i, w in enumerate(widths, 1):
        _set_col_width(ws, get_column_letter(i), w)

    ws.freeze_panes = "A3"
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 18


# ── Sheet 2: Heat Map ──────────────────────────────────────────────────────────

def build_heat_map(ws, records: list[dict]) -> None:
    ws.title = "Heat Map"

    # Raccoglie valori unici ordinati
    rounds_list  = sorted(set(r["rounds"] for r in records))
    epsilon_list = sorted(set(r["epsilon"] for r in records))

    # Indice (rounds, epsilon) → auc_roc
    data_map: dict[tuple, float | None] = {}
    for rec in records:
        key = (rec["rounds"], rec["epsilon"])
        # Se duplicato, tieni il più recente (già ordinati per timestamp)
        data_map[key] = rec["auc_roc"]

    # Titolo
    n_cols = len(epsilon_list) + 2
    ws.merge_cells(f"A1:{get_column_letter(n_cols)}1")
    title = ws["A1"]
    title.value = "AUC-ROC Heat Map — FedMIA vs ε (Differential Privacy Budget)"
    title.font = _font(bold=True, color=COLOR_HEADER_FG, size=12)
    title.fill = _fill(COLOR_HEADER_BG)
    title.alignment = _center()

    # Sottotitolo
    ws.merge_cells(f"A2:{get_column_letter(n_cols)}2")
    sub = ws["A2"]
    sub.value = (
        "AUC-ROC ≈ 0.50 → MIA non migliore del random (DP efficace)  |  "
        "AUC-ROC > 0.55 → MIA parzialmente efficace  |  "
        "Dataset: ACN-Data JPL 2019+2020 (13,073 sessioni)"
    )
    sub.font = _font(bold=False, color="404040", size=9)
    sub.fill = _fill("EBF3FB")
    sub.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[2].height = 28

    # Header riga 3: etichetta + epsilon columns
    _header_cell(ws.cell(3, 1), "FL Rounds \\ ε →", bg=COLOR_HEADER_BG)
    for col, eps in enumerate(epsilon_list, 2):
        _header_cell(ws.cell(3, col), f"ε = {eps}", bg=COLOR_SUBHDR_BG)
    _header_cell(ws.cell(3, len(epsilon_list) + 2), "Row Avg", bg=COLOR_SUBHDR_BG)

    # Righe dati
    for row_idx, rnd in enumerate(rounds_list, 4):
        alt = (row_idx % 2 == 0)
        _header_cell(ws.cell(row_idx, 1), f"{rnd} rounds", bg=COLOR_SUBHDR_BG)

        row_values = []
        for col_idx, eps in enumerate(epsilon_list, 2):
            val = data_map.get((rnd, eps))
            c = ws.cell(row_idx, col_idx)
            c.value = val
            c.font = _font(bold=val is not None and val > 0.51)
            c.fill = _fill(COLOR_ROW_ALT if alt else COLOR_ROW_PLAIN)
            c.border = _border()
            c.alignment = _center()
            c.number_format = "0.0000"
            if val is not None:
                row_values.append(val)
                # Colora AUC: verde→giallo→rosso
                if val > 0.60:
                    c.fill = _fill("FFCCCC")   # rosso chiaro
                    c.font = _font(bold=True, color=COLOR_BAD)
                elif val > 0.52:
                    c.fill = _fill("FFE5B4")   # arancione chiaro
                    c.font = _font(bold=True, color="8B4513")
                else:
                    c.fill = _fill("D5E8D4")   # verde chiaro (DP efficace)
                    c.font = _font(bold=False, color="2D6A2D")

        # Media di riga
        avg_col = len(epsilon_list) + 2
        avg_cell = ws.cell(row_idx, avg_col)
        if row_values:
            avg = sum(row_values) / len(row_values)
            avg_cell.value = avg
            avg_cell.number_format = "0.0000"
            avg_cell.font = _font(bold=True)
        else:
            avg_cell.value = "N/A"
        avg_cell.fill = _fill("E2EFDA")
        avg_cell.border = _border()
        avg_cell.alignment = _center()

    # Riga media di colonna
    avg_row = len(rounds_list) + 4
    _header_cell(ws.cell(avg_row, 1), "Col Avg", bg=COLOR_SUBHDR_BG)
    for col_idx, eps in enumerate(epsilon_list, 2):
        col_vals = [
            data_map.get((rnd, eps))
            for rnd in rounds_list
            if data_map.get((rnd, eps)) is not None
        ]
        c = ws.cell(avg_row, col_idx)
        if col_vals:
            avg = sum(col_vals) / len(col_vals)
            c.value = avg
            c.number_format = "0.0000"
            c.font = _font(bold=True)
        else:
            c.value = "N/A"
        c.fill = _fill("E2EFDA")
        c.border = _border()
        c.alignment = _center()

    # Legenda
    legend_row = avg_row + 2
    ws.cell(legend_row, 1).value = "Legenda:"
    ws.cell(legend_row, 1).font = _font(bold=True)
    legends = [
        ("D5E8D4", "2D6A2D", "AUC ≤ 0.52 — DP efficace, MIA ≈ random"),
        ("FFE5B4", "8B4513", "0.52 < AUC ≤ 0.60 — leakage parziale"),
        ("FFCCCC", COLOR_BAD, "AUC > 0.60 — MIA efficace, rischio HIGH"),
    ]
    for i, (bg, fg, label) in enumerate(legends):
        c = ws.cell(legend_row + 1 + i, 2)
        c.value = label
        c.fill = _fill(bg)
        c.font = _font(color=fg, bold=True)
        c.border = _border()
        c.alignment = Alignment(horizontal="left")
        ws.merge_cells(
            start_row=legend_row + 1 + i, start_column=2,
            end_row=legend_row + 1 + i, end_column=5
        )

    # Larghezze
    _set_col_width(ws, "A", 16)
    for col_idx in range(2, len(epsilon_list) + 3):
        _set_col_width(ws, get_column_letter(col_idx), 14)

    ws.freeze_panes = "B4"
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[3].height = 18


# ── Sheet 3: Per Rounds ────────────────────────────────────────────────────────

def build_per_rounds(ws, records: list[dict]) -> None:
    ws.title = "Per Rounds"

    ws.merge_cells("A1:F1")
    t = ws["A1"]
    t.value = "AUC-ROC Statistics by Number of FL Rounds"
    t.font = _font(bold=True, color=COLOR_HEADER_FG, size=12)
    t.fill = _fill(COLOR_HEADER_BG)
    t.alignment = _center()

    headers = ["FL Rounds", "N Experiments", "AUC-ROC Mean", "AUC-ROC Min", "AUC-ROC Max", "Std Dev"]
    for col, h in enumerate(headers, 1):
        _header_cell(ws.cell(2, col), h, bg=COLOR_SUBHDR_BG)

    from statistics import mean, stdev

    rounds_list = sorted(set(r["rounds"] for r in records))
    for row_idx, rnd in enumerate(rounds_list, 3):
        alt = (row_idx % 2 == 0)
        group = [r["auc_roc"] for r in records if r["rounds"] == rnd and r["auc_roc"] is not None]
        _data_cell(ws.cell(row_idx, 1), rnd,           alt_row=alt, bold=True)
        _data_cell(ws.cell(row_idx, 2), len(group),    alt_row=alt)
        _data_cell(ws.cell(row_idx, 3), mean(group) if group else None, fmt="0.0000", alt_row=alt)
        _data_cell(ws.cell(row_idx, 4), min(group)  if group else None, fmt="0.0000", alt_row=alt)
        _data_cell(ws.cell(row_idx, 5), max(group)  if group else None, fmt="0.0000", alt_row=alt)
        _data_cell(ws.cell(row_idx, 6), stdev(group) if len(group) > 1 else 0.0, fmt="0.0000", alt_row=alt)

    for col, w in zip("ABCDEF", [14, 16, 16, 14, 14, 12]):
        _set_col_width(ws, col, w)


# ── Sheet 4: Per Epsilon ───────────────────────────────────────────────────────

def build_per_epsilon(ws, records: list[dict]) -> None:
    ws.title = "Per Epsilon"

    ws.merge_cells("A1:F1")
    t = ws["A1"]
    t.value = "AUC-ROC Statistics by DP Budget (ε) — Privacy/Utility Trade-off"
    t.font = _font(bold=True, color=COLOR_HEADER_FG, size=12)
    t.fill = _fill(COLOR_HEADER_BG)
    t.alignment = _center()

    headers = ["Epsilon (ε)", "N Experiments", "AUC-ROC Mean", "AUC-ROC Min", "AUC-ROC Max", "Interpretation"]
    for col, h in enumerate(headers, 1):
        _header_cell(ws.cell(2, col), h, bg=COLOR_SUBHDR_BG)

    from statistics import mean

    epsilon_list = sorted(set(r["epsilon"] for r in records))
    interpretations = {
        0.1: "Strong DP — high noise, MIA likely ineffective",
        0.5: "Moderate-strong DP",
        1.0: "Standard DP budget",
        2.0: "Moderate DP — lower noise",
        5.0: "Weak DP — low noise, higher MIA risk",
    }
    for row_idx, eps in enumerate(epsilon_list, 3):
        alt = (row_idx % 2 == 0)
        group = [r["auc_roc"] for r in records if r["epsilon"] == eps and r["auc_roc"] is not None]
        avg = mean(group) if group else None

        _data_cell(ws.cell(row_idx, 1), eps,            fmt="0.0#",  alt_row=alt, bold=True)
        _data_cell(ws.cell(row_idx, 2), len(group),     alt_row=alt)
        auc_cell = ws.cell(row_idx, 3)
        _data_cell(auc_cell, avg, fmt="0.0000", alt_row=alt)
        if avg is not None and avg > 0.55:
            auc_cell.font = _font(bold=True, color=COLOR_BAD)
        elif avg is not None and avg > 0.51:
            auc_cell.font = _font(bold=True, color=COLOR_WARN)
        else:
            auc_cell.font = _font(bold=True, color=COLOR_GOOD)

        _data_cell(ws.cell(row_idx, 4), min(group) if group else None, fmt="0.0000", alt_row=alt)
        _data_cell(ws.cell(row_idx, 5), max(group) if group else None, fmt="0.0000", alt_row=alt)

        interp_cell = ws.cell(row_idx, 6)
        interp_cell.value = interpretations.get(eps, "")
        interp_cell.font = _font()
        interp_cell.fill = _fill(COLOR_ROW_ALT if alt else COLOR_ROW_PLAIN)
        interp_cell.border = _border()
        interp_cell.alignment = Alignment(horizontal="left", vertical="center")

    for col, w in zip("ABCDEF", [14, 16, 16, 14, 14, 42]):
        _set_col_width(ws, col, w)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera report Excel sweep ChargeShield-FL")
    parser.add_argument(
        "--output", type=Path,
        default=EXPERIMENTS_DIR / "ChargeShield_FL_Results.xlsx",
    )
    args = parser.parse_args()

    records = load_experiments()
    if not records:
        print("ERROR: Nessun file experiment_*.json trovato in experiments/")
        sys.exit(1)

    print(f"Caricati {len(records)} esperimenti.")

    wb = Workbook()
    wb.remove(wb.active)  # rimuovi sheet vuoto di default

    ws_raw   = wb.create_sheet("Raw Data")
    ws_heat  = wb.create_sheet("Heat Map")
    ws_round = wb.create_sheet("Per Rounds")
    ws_eps   = wb.create_sheet("Per Epsilon")

    build_raw_data(ws_raw, records)
    build_heat_map(ws_heat, records)
    build_per_rounds(ws_round, records)
    build_per_epsilon(ws_eps, records)

    # Proprietà workbook
    wb.properties.title   = "ChargeShield-FL Experiment Results"
    wb.properties.subject = "FedMIA vs Differential Privacy — DSN 2027"
    wb.properties.creator = "ChargeShield-FL Framework"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    print(f"Report salvato: {args.output}")


if __name__ == "__main__":
    main()
