"""Clarify accuracy units and final-head decoding while retaining vector artwork.

Requires pypdf, pdfplumber, reportlab, and LiberationSerif-Regular.ttf.
The retained inputs are the publication assets before the typography revision.
"""
import argparse
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, TextStringObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
FONT = "FigureSerif"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_runs(path):
    reader = PdfReader(path)
    page = reader.pages[0]
    stream = ContentStream(page.get_contents(), reader)
    with pdfplumber.open(path) as pdf:
        chars = pdf.pages[0].chars
    fonts = page["/Resources"]["/Font"]
    font = None
    size = None
    position = 0
    runs = []
    for index, (args, op) in enumerate(stream.operations):
        if op == b"Tf":
            font, size = fonts[args[0]], float(args[1])
        if op == b"TJ":
            raise ValueError("Unexpected text array in retained source")
        if op != b"Tj":
            continue
        raw = args[0].get_original_bytes()
        if raw.startswith(b"\xfe\xff"):
            raw = raw[2:]
        count = len(raw) // 2 if font["/Subtype"] == "/Type0" else len(raw)
        if not count:
            continue
        group = chars[position:position + count]
        position += count
        matrix = group[0]["matrix"]
        factor = math.hypot(matrix[0], matrix[1])
        runs.append(dict(text="".join(c["text"] for c in group),
                         x=matrix[4], y=matrix[5], size=size * factor,
                         angle=math.degrees(math.atan2(matrix[1], matrix[0])),
                         width=sum(c["adv"] for c in group) * factor,
                         color=group[0]["non_stroking_color"], operation=index))
    assert position == len(chars)
    return reader, stream, runs


def draw(c, text, x, y, size, color, angle=0, align="left"):
    c.saveState()
    c.setFillColorRGB(*color)
    c.translate(x, y)
    c.rotate(angle)
    c.setFont(FONT, size)
    {"left": c.drawString, "center": c.drawCentredString,
     "right": c.drawRightString}[align](0, 0, text)
    c.restoreState()


def legend(c, labels, colors, y, size, width, kinds, dashed=None):
    handles, gap, itemgap = 20, 6, 18
    widths = [handles + gap + pdfmetrics.stringWidth(t, FONT, size) for t in labels]
    total = sum(widths) + itemgap * (len(labels) - 1)
    assert total < width - 20, (labels, total, width)
    x = (width - total) / 2
    for n, (label, color, itemwidth, kind) in enumerate(zip(labels, colors, widths, kinds)):
        c.setStrokeColorRGB(*color)
        c.setFillColorRGB(*color)
        c.setLineWidth(1.35)
        c.setDash([2, 3] if dashed and dashed[n] else [])
        if kind == "point":
            c.circle(x + handles / 2, y + size * .28, 2.8, stroke=0, fill=1)
        else:
            c.line(x, y + size * .28, x + handles, y + size * .28)
            if kind == "linepoint":
                c.circle(x + handles / 2, y + size * .28, 2.3, stroke=0, fill=1)
        draw(c, label, x + handles + gap, y, size, (.10, .10, .10))
        x += itemwidth + itemgap
    c.setDash([])


def revise(name, include_fraction, font_path):
    original = HERE / "legibility_originals" / name
    output = HERE.parent / name
    reader, stream, runs = extract_runs(original)
    label_changes = {
        'accuracy of the task-aligned family alone': 'accuracy of the task-aligned family alone (%)',
        'accuracy of the full model': 'accuracy of the full model (%)',
        'accuracy after the intervention': 'accuracy after the intervention (%)',
        'accuracy decoding the block directly': 'accuracy with the final head (%)',
        'decode the block directly': 'final head on block residual',
        'decode its task-aligned part': 'final head on task-aligned part',
    }
    conversions = []
    for run in runs:
        if run['text'] in label_changes:
            old = run['text']; run['text'] = label_changes[old]
            conversions.append(dict(old=old,new=run['text']))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    size = 7.0 * width / (396 * include_fraction)
    geometry_before = [(args, op) for args, op in stream.operations if op != b"Tj"]
    for run in runs:
        stream.operations[run["operation"]] = ([TextStringObject("")], b"Tj")
    assert geometry_before == [(args, op) for args, op in stream.operations if op != b"Tj"]
    page.replace_contents(stream)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(width, height), invariant=1)
    # Reflow only the legend within its existing band below the axis titles.
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, width, 37, fill=1, stroke=0)
    projection = name == "figure5_modes.pdf"
    x_ticks = set(range(0, 5)) | set(range(10, 17)) if projection else set(range(0, 5)) | set(range(10, 14))
    y_ticks = set(range(5, 10)) | set(range(17, 22)) if projection else set(range(5, 10)) | set(range(14, 19))
    legend_ids = set(range(22, 27)) if projection else set(range(19, 24))
    for i, run in enumerate(runs):
        if i in legend_ids:
            continue
        text, x, y, angle = run["text"], run["x"], run["y"], run["angle"]
        color = run["color"]
        align = "left"
        if i in x_ticks:
            x += run["width"] / 2
            align = "center"
        elif i in y_ticks:
            x += run["width"]
            y -= (size - run["size"]) * .27
            align = "right"
        elif abs(angle - 90) < .1:
            y += run["width"] / 2
            if not projection:
                x = 12 if i == 26 else 461
            align = "center"
        elif (projection and i in [27, 28]) or (not projection and i in [24, 25]):
            x += run["width"] / 2
            align = "center"
        elif projection:
            if i in [34, 35]:
                continue
            if i == 31:
                x, y = 107, 76
                c.setFillColorRGB(1, 1, 1)
                c.rect(x - 2, y - 2, pdfmetrics.stringWidth(text, FONT, size) + 4, 13, fill=1, stroke=0)
            elif i == 32:
                x += run["width"]
                align = "right"
            elif i == 33:
                x, y = 125, 162
            elif i == 36:
                x, y, align = 511, 158, "center"
            elif i == 37:
                x, y, align = 511, 140, "center"
            elif i == 39:
                x, y = 110, 113
            elif i == 40:
                x += run["width"]
                align = "right"
        else:
            if i == 28:
                x, y, align = 281, 204, "right"
            elif i == 29:
                x, y, align = 281, 188, "right"
        draw(c, text, x, y, size, color, angle, align)
    if projection:
        lines = ["the family alone is never", "worse than the model", "containing it, at 95 of the", "98 checkpoints"]
        for line, y in zip(lines, [271, 253, 235, 217]):
            draw(c, line, 74, y, size, runs[34]["color"])
        legend(c, [r["text"] for r in runs[22:25]],
               [(.3725, .4196, .4706), (.7098, .1961, .2275), (.1216, .3725, .6588)],
               24, size, width, ["point"] * 3)
        legend(c, [r["text"] for r in runs[25:27]],
               [(.7098, .1961, .2275), (.1216, .3725, .6588)],
               5, size, width, ["line"] * 2)
    else:
        legend(c, [r["text"] for r in runs[19:22]],
               [(.7098, .1961, .2275), (.7882, .6353, .1529), (.4314, .4824, .5333)],
               24, size, width, ["linepoint", "linepoint", "linepoint"])
        legend(c, [r["text"] for r in runs[22:24]],
               [(.2275, .2275, .2275), (.1216, .3725, .6588)],
               5, size, width, ["linepoint", "linepoint"], dashed=[False, True])
    c.save()
    page.merge_page(PdfReader(buffer).pages[0])
    writer.add_metadata({"/Title": name.removesuffix(".pdf").replace("_", " "),
                         "/Subject": "Enlarged typography; original experimental vector geometry preserved"})
    with output.open("wb") as f:
        writer.write(f)
    actual = PdfReader(output).pages[0].extract_text()
    before_tokens = Counter(re.findall(r"\S+", " ".join(r["text"] for r in runs)))
    after_tokens = Counter(re.findall(r"\S+", actual))
    assert before_tokens == after_tokens, (name, before_tokens-after_tokens, after_tokens-before_tokens)
    return dict(input=original.relative_to(HERE.parent).as_posix(), output=name,
                input_sha256=sha(original), output_sha256=sha(output), font_sha256=sha(font_path),
                replaced_text_runs=len(runs), source_font_size=size,
                target_printed_font_size=7, include_fraction=include_fraction,
                all_text_tokens_preserved_after_registered_label_changes=True, label_changes=conversions, original_nontext_operations_preserved=True,
                legend_reflow_band=[0, 0, width, 37])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont(FONT, str(args.font)))
    records = [revise("figure5_modes.pdf", .85, args.font), revise("figure6_depth.pdf", 1.0, args.font)]
    (HERE / "projection_depth_axis_revision.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
