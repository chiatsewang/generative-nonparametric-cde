import argparse
import glob
import os
import zipfile
from xml.sax.saxutils import escape

import numpy as np

# skip_rules.mk is the single source of method/model applicability rules.
# Its model-group definitions are literal lists, and *_APPLY_MODELS may use
# at most one reference to one of those predefined groups.
MODEL_GROUPS = ("ALL_MODELS", "UNIVARIATE_MODELS", "MULTIVARIATE_MODELS")


def load_skip_rules():
    from pathlib import Path

    settings_path = Path(__file__).resolve().parent.parent / "skip_rules.mk"
    settings = {}

    for line in settings_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition(":=")
        if not sep:
            raise ValueError(f"Invalid skip_rules.mk definition: {line}")
        name = name.strip()
        if name in settings:
            raise ValueError(f"Duplicate skip_rules.mk variable: {name}")
        settings[name] = value.strip()

    for group in (*MODEL_GROUPS, "ALL_ALGOS"):
        if "$" in settings[group]:
            raise ValueError(
                f"{group} must be defined without variable references"
            )

    algorithms = settings["ALL_ALGOS"].split()
    models = settings["ALL_MODELS"].split()
    applicable = {}

    for algo in algorithms:
        key = f"{algo.upper()}_APPLY_MODELS"
        value = settings[key]
        if "$" in value:
            if not (value.startswith("$(") and value.endswith(")")):
                raise ValueError(
                    f"Invalid model-group reference in {key}: {value}"
                )
            group = value[2:-1]
            if group not in MODEL_GROUPS:
                raise ValueError(
                    f"Unsupported model-group reference in {key}: {group}"
                )
            value = settings[group]
        applicable[algo] = set(value.split())
        if not applicable[algo] <= set(models):
            raise ValueError(f"Unknown data model in {key}")

    return algorithms, models, applicable


ALGORITHMS, DATA_MODELS, APPLY_MODELS = load_skip_rules()

METHOD_LABELS = {
    "ddpm": "DDPM",
    "gcds": "GCDS",
    "hy": "HY",
    "flex": "Flex",
    "deepcde": "DeepCDE",
    "srm": "SRM",
    "ctmtf": "CTMTF",
    "drf": "DRF",
}

METRICS = [
    ("mu_mse", "Mu MSE"),
    ("mu_medse", "Mu MedSE"),
    ("sigma_mse", "Sigma MSE"),
    ("sigma_medse", "Sigma MedSE"),
    ("w_mean", "W1"),
    ("escore_mean", "EScore"),
    ("reescore", "rEEScore"),
]


# -----------------------------------------------------------------------------
# Argument handling and aggregation
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate repeated simulation results into one Excel workbook"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Optional algorithm name (e.g., ddpm, hy)",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="Optional synthetic data model (e.g., U1, M2)",
    )
    return parser.parse_args()


def normalize_args(args):
    algorithm = args.algorithm.lower() if args.algorithm is not None else None
    data_name = args.data_name.upper() if args.data_name is not None else None

    if algorithm is not None and algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{args.algorithm}'. "
            f"Choose from: {', '.join(ALGORITHMS)}"
        )
    if data_name is not None and data_name not in DATA_MODELS:
        raise ValueError(
            f"Unknown data model '{args.data_name}'. "
            f"Choose from: {', '.join(DATA_MODELS)}"
        )

    return algorithm, data_name


def summarize(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def collect_case(algorithm, data_name):
    applicable = data_name in APPLY_MODELS[algorithm]
    record = {
        "algorithm": algorithm,
        "method": METHOD_LABELS[algorithm],
        "data_name": data_name,
        "applicable": applicable,
        "status": "not applicable" if not applicable else "missing",
        "n_seeds": None if not applicable else 0,
        "summaries": {key: (None, None) for key, _ in METRICS},
    }

    if not applicable:
        return record

    pattern = os.path.join(
        "workspaces", algorithm, data_name, "seed_*", "eval.npz"
    )
    eval_files = sorted(glob.glob(pattern))

    if not eval_files:
        print(f"[Warning] {algorithm}-{data_name}: no eval.npz files found")
        return record

    metric_values = {key: [] for key, _ in METRICS}
    n_failed = 0

    for path in eval_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                values = {}
                for key, _ in METRICS:
                    values[key] = float(np.asarray(data[key]).squeeze())
        except Exception as exc:
            n_failed += 1
            print(f"[Warning] Failed to read {path}: {exc}")
            continue

        for key, value in values.items():
            metric_values[key].append(value)

    n_valid = len(metric_values[METRICS[0][0]])
    record["n_seeds"] = n_valid

    if n_valid == 0:
        print(f"[Warning] {algorithm}-{data_name}: no valid eval.npz files")
        return record

    record["status"] = "ok" if n_failed == 0 else "read error"
    record["summaries"] = {
        key: summarize(metric_values[key]) for key, _ in METRICS
    }
    return record


def output_path(algorithm, data_name):
    if algorithm is None and data_name is None:
        filename = "all.xlsx"
    elif algorithm is not None and data_name is None:
        filename = f"{algorithm}.xlsx"
    elif algorithm is None and data_name is not None:
        filename = f"{data_name}.xlsx"
    else:
        filename = f"{algorithm}_{data_name}.xlsx"

    return os.path.join("workspaces", filename)


# -----------------------------------------------------------------------------
# Minimal XLSX writer
#
# The project currently has no spreadsheet-writing package in its dependencies.
# These helpers write the small workbook needed here directly as Office Open XML,
# so conclude_simulation.py does not introduce an additional Python dependency.
# -----------------------------------------------------------------------------


def column_letter(index):
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def cell_xml(row, col, value, style=0):
    ref = f"{column_letter(col)}{row}"
    style_attr = f' s="{style}"' if style else ""

    if value is None:
        return f'<c r="{ref}"{style_attr}/>'

    if isinstance(
        value, (int, float, np.integer, np.floating)
    ) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'

    text = escape(str(value))
    return (
        f'<c r="{ref}" t="inlineStr"{style_attr}>'
        f"<is><t>{text}</t></is></c>"
    )


def sheet_xml(rows, widths, freeze_top_row=True):
    n_rows = max(len(rows), 1)
    n_cols = max((len(row) for row in rows), default=1)
    dimension = f"A1:{column_letter(n_cols)}{n_rows}"

    if freeze_top_row:
        sheet_views = (
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" '
            'state="frozen"/>'
            "</sheetView></sheetViews>"
        )
    else:
        sheet_views = (
            '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        )

    cols = ["<cols>"]
    for i, width in enumerate(widths, start=1):
        cols.append(
            f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
        )
    cols.append("</cols>")

    body = ["<sheetData>"]
    for row_idx, row in enumerate(rows, start=1):
        body.append(f'<row r="{row_idx}">')
        for col_idx, item in enumerate(row, start=1):
            value, style = item
            body.append(cell_xml(row_idx, col_idx, value, style))
        body.append("</row>")
    body.append("</sheetData>")

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        f"{sheet_views}" + "".join(cols) + "".join(body) + "</worksheet>"
    )


def build_summary_rows(records, selected_algorithms):
    headers = ["Model", "Method"] + [label for _, label in METRICS]
    rows = [[(value, 1) for value in headers]]

    first_row_by_model = {}
    for record in records:
        model = record["data_name"]
        show_model = model not in first_row_by_model
        first_row_by_model[model] = True

        row = [
            (model if show_model else None, 4 if show_model else 2),
            (record["method"], 2),
        ]

        for key, _ in METRICS:
            mean_value, sd_value = record["summaries"][key]
            if mean_value is None:
                display = None
            else:
                display = f"{mean_value:.4f} ({sd_value:.4f})"
            row.append((display, 2))

        rows.append(row)

    # If only one algorithm is requested, each model has only one row; showing the
    # model label on every row is already handled by the logic above.
    _ = selected_algorithms
    return rows


def build_details_rows(records):
    headers = ["Model", "Method", "Status", "N Seeds"]
    for _, label in METRICS:
        headers.extend([f"{label} Mean", f"{label} SD"])

    rows = [[(value, 1) for value in headers]]

    for record in records:
        row = [
            (record["data_name"], 2),
            (record["method"], 2),
            (record["status"], 2),
            (record["n_seeds"], 2),
        ]
        for key, _ in METRICS:
            mean_value, sd_value = record["summaries"][key]
            row.extend([(mean_value, 3), (sd_value, 3)])
        rows.append(row)

    return rows


def write_xlsx(path, records, selected_algorithms):
    summary_rows = build_summary_rows(records, selected_algorithms)
    details_rows = build_details_rows(records)

    summary_widths = [10, 14] + [20] * len(METRICS)
    details_widths = [10, 14, 18, 10] + [16] * (2 * len(METRICS))

    summary_sheet = sheet_xml(summary_rows, summary_widths)
    details_sheet = sheet_xml(details_rows, details_widths)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="summary" sheetId="1" r:id="rId1"/>
    <sheet name="details" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1">
    <numFmt numFmtId="164" formatCode="0.0000"/>
  </numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD9D9D9"/></left>
      <right style="thin"><color rgb="FFD9D9D9"/></right>
      <top style="thin"><color rgb="FFD9D9D9"/></top>
      <bottom style="thin"><color rgb="FFD9D9D9"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", root_rels)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/styles.xml", styles)
        xlsx.writestr("xl/worksheets/sheet1.xml", summary_sheet)
        xlsx.writestr("xl/worksheets/sheet2.xml", details_sheet)


def main():
    args = parse_args()
    algorithm, data_name = normalize_args(args)

    selected_algorithms = [algorithm] if algorithm is not None else ALGORITHMS
    selected_models = [data_name] if data_name is not None else DATA_MODELS

    records = []
    for model in selected_models:
        for algo in selected_algorithms:
            records.append(collect_case(algo, model))

    path = output_path(algorithm, data_name)
    write_xlsx(path, records, selected_algorithms)

    n_ok = sum(record["status"] == "ok" for record in records)
    n_not_applicable = sum(
        record["status"] == "not applicable" for record in records
    )
    n_missing = sum(record["status"] == "missing" for record in records)
    n_read_error = sum(record["status"] == "read error" for record in records)

    print(
        "[Summary] "
        f"ok={n_ok}, missing={n_missing}, "
        f"not_applicable={n_not_applicable}, read_error={n_read_error}"
    )
    print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
