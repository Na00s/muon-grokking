"""Replace two annotation labels while preserving the original vector artwork.

Requires pypdf and reportlab. Pass a LiberationSerif-Regular.ttf font file.
The original PDF is retained beside this script for reproducibility.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, TextStringObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    original = here / 'figure5_modes_original.pdf'
    output = here.parent / 'figure5_modes.pdf'
    reader = PdfReader(original)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    labels = {'circuit failure': 'projection failure',
              'circuit masking': 'projection masking'}
    placements = {}

    def locate(text, cm, tm, font, size):
        if text in labels:
            assert font['/BaseFont'] == '/LiberationSerif'
            assert cm[1] == cm[2] == tm[1] == tm[2] == 0
            placements[text] = {'x': tm[4] * cm[0] + cm[4],
                                'y': tm[5] * cm[3] + cm[5],
                                'size': size * cm[0]}

    before = page.extract_text(visitor_text=locate)
    assert set(placements) == set(labels)
    stream = ContentStream(page.get_contents(), writer)
    removed = []
    changed_indices = []
    for index, (operands, operator) in enumerate(stream.operations):
        if operator != b'Tj':
            continue
        raw = operands[0].get_original_bytes()
        if raw.startswith(b'\xfe\xff'):
            raw = raw[2:]
        # The original embedded font maps these ASCII glyphs to CID = code - 29.
        for old in labels:
            encoded = b''.join((ord(c) - 29).to_bytes(2, 'big') for c in old)
            if raw == encoded:
                stream.operations[index] = ([TextStringObject('')], operator)
                removed.append(old)
                changed_indices.append(index)
    assert sorted(removed) == sorted(labels), removed
    page.replace_contents(stream)

    pdfmetrics.registerFont(TTFont('OriginalFigureSerif', str(args.font)))
    buffer = io.BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(float(page.mediabox.width),
                                              float(page.mediabox.height)), invariant=1)
    overlay.setFillColorRGB(0.1647, 0.1647, 0.1647)
    for old, new in labels.items():
        p = placements[old]
        overlay.setFont('OriginalFigureSerif', p['size'])
        if old == 'circuit masking':
            # Keep the label's right edge at the existing horizontal leader.
            width = pdfmetrics.stringWidth(old, 'OriginalFigureSerif', p['size'])
            overlay.drawRightString(p['x'] + width, p['y'], new)
        else:
            overlay.drawString(p['x'], p['y'], new)
    overlay.save()
    page.merge_page(PdfReader(buffer).pages[0])
    writer.add_metadata({'/Title': 'Projection failure and projection masking',
                         '/Subject': 'Original data and vector geometry; two annotation labels updated'})
    with output.open('wb') as f:
        writer.write(f)
    after = PdfReader(output).pages[0].extract_text()
    assert all(old not in after and new in after for old, new in labels.items())
    remaining_before, remaining_after = before, after
    for old, new in labels.items():
        remaining_before = remaining_before.replace(old, '')
        remaining_after = remaining_after.replace(new, '')
    assert ''.join(remaining_before.split()) == ''.join(remaining_after.split())
    record = {'original_sha256': sha(original), 'output_sha256': sha(output),
              'font_sha256': sha(args.font), 'replacements': labels,
              'text_show_operations_replaced': changed_indices,
              'other_original_operations_unchanged': True,
              'all_other_extracted_text_unchanged': True,
              'placements': placements}
    (here / 'projection_label_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
