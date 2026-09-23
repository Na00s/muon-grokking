"""Enlarge the quiet-window and spectral-grid labels without changing plotted data.

Requires pypdf and reportlab, and the original LiberationSerif-Regular.ttf font.
Both figures are included at full text width (396 PDF points) in appendix.tex.
The original vector paths and data image streams are retained. Text is drawn
again from decoded original text-show operations with larger type and spacing.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._cmap import get_encoding
from pypdf.generic import ContentStream, TextStringObject, RectangleObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
FONT = "OriginalFigureSerif"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_labels(page):
    state = {"color": (0., 0., 0.)}
    stack, labels = [], []
    def before(op, args, cm, tm):
        if op == b'q': stack.append(state.copy())
        elif op == b'Q': state.clear(); state.update(stack.pop())
        elif op == b'rg': state['color'] = tuple(map(float, args))
        elif op == b'g': state['color'] = (float(args[0]),) * 3
        elif op == b'Tf': state['font'], state['size'] = str(args[0]), float(args[1])
        elif op == b'Tj':
            ft = page['/Resources']['/Font'][state['font']].get_object()
            assert ft['/BaseFont'] == '/LiberationSerif'
            encoding, mapping = get_encoding(ft)
            raw = bytes(args[0]) if isinstance(args[0], bytes) else args[0].get_original_bytes()
            text = ''.join(mapping.get(c, c) for c in raw.decode(encoding))
            labels.append({
                'text': text,
                'x': tm[4] * cm[0] + tm[5] * cm[2] + cm[4],
                'y': tm[4] * cm[1] + tm[5] * cm[3] + cm[5],
                'size': state['size'] * math.hypot(cm[0], cm[1]),
                'angle': math.degrees(math.atan2(cm[1], cm[0])),
                'color': state['color'],
            })
    page.extract_text(visitor_operand_before=before)
    return labels

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE.parent)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont(FONT, str(args.font)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'figures').mkdir(exist_ok=True)
    records = []
    for filename in ['figure2_quiet.pdf', 'figure4_grid.pdf']:
        source = HERE / 'legibility_originals' / filename
        reader = PdfReader(source)
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        page = writer.pages[0]
        labels = get_labels(page)
        stream = ContentStream(page.get_contents(), writer)
        original_ops = [(args, op) for args, op in stream.operations]
        stripped = 0
        for i, (operands, operator) in enumerate(stream.operations):
            if operator == b'Tj':
                stream.operations[i] = ([TextStringObject('')], operator)
                stripped += 1
        assert stripped == len(labels)
        assert all(a == b for a,b in zip(original_ops, stream.operations) if a[1] != b'Tj')
        page.replace_contents(stream)
        original_width, original_height = float(page.mediabox.width), float(page.mediabox.height)
        grid = filename == 'figure4_grid.pdf'
        offset = 16 if grid else 0
        width, height = (original_width + offset, 240) if grid else (original_width, original_height)
        if offset:
            page.add_transformation(Transformation().translate(tx=offset, ty=0))
        page.mediabox = RectangleObject([0,0,width,height])
        page.cropbox = RectangleObject([0,0,width,height])
        buffer = io.BytesIO()
        overlay = canvas.Canvas(buffer, pagesize=(width,height), invariant=1)
        minimum = 7 * width / 396
        written = []
        def draw(text,x,y,*,size=minimum,angle=0,color=(0.,0.,0.),align='left'):
            overlay.saveState()
            overlay.translate(x,y)
            overlay.rotate(angle)
            overlay.setFillColorRGB(*color)
            overlay.setFont(FONT,size)
            if align == 'right': overlay.drawRightString(0,0,text)
            elif align == 'center': overlay.drawCentredString(0,0,text)
            else: overlay.drawString(0,0,text)
            overlay.restoreState()
            written.append({'text':text,'x':x,'y':y,'size':size,'printed_size':size*396/width,'angle':angle})
        if not grid:
            for i,p in enumerate(labels):
                text,x,y,old = p['text'],p['x'],p['y'],p['size']
                old_width = pdfmetrics.stringWidth(text,FONT,old)
                align = 'left'; size = minimum
                if i <= 15 or 20 <= i <= 23:
                    x += old_width; y -= (size-old)*.25; align = 'right'
                elif 16 <= i <= 19:
                    x += old_width/2; align = 'center'
                elif i == 24:
                    x += old_width/2; align = 'center'; size = 8 * width /396
                elif 25 <= i <= 27:
                    x = 12; y += old_width/2; align = 'center'; size = 8 * width/396
                elif i == 31: x = 580.5
                elif i == 32: y = 268.5
                elif i == 34: x = 600; y = 217.5
                elif i == 36: x = 600; y = 184.5
                elif i == 37: y = 135
                elif i == 38: y = 115
                draw(text,x,y,size=size,angle=p['angle'],color=p['color'],align=align)
        else:
            # The original three legend symbols occupy only this footer strip.
            # Redraw their keys with more spacing; no data marks enter the strip.
            overlay.setFillColorRGB(1,1,1)
            overlay.rect(0,0,width,30,fill=1,stroke=0)
            for i,p in enumerate(labels[:11]):
                if 6 <= i <= 8: continue
                text,x,y,old = p['text'],p['x']+offset,p['y'],p['size']
                old_width = pdfmetrics.stringWidth(text,FONT,old)
                size=minimum; align='left'
                if i <= 2: x += old_width/2; align='center'; y = 36
                elif 3 <= i <= 5: x += old_width; align='right'; y -= (size-old)*.25
                elif i == 9: x = (width+offset)/2; y = 35; align='center';size=8*width/396
                elif i == 10: x=14; y += old_width/2;align='center';size=8*width/396
                draw(text,x,y,size=size,angle=p['angle'],color=p['color'],align=align)
            headers = ['AdamW, addition','Muon, addition','Muon without the\nnormalized path','AdamW, subtraction','Muon, subtraction']
            centers = [99.4,244.5,389.6,534.7,679.8]
            for text,x in zip(headers,centers):
                lines = text.split('\n')
                for j,line in enumerate(lines):
                    draw(line,x+offset,217-j*16,align='center')
            legend = [(190,'generic interaction',(0.65,0.69,0.74)),(365,'subtraction',(0.20,0.43,0.70)),(505,'addition',(0.75,0.25,0.29))]
            for x,text,color in legend:
                overlay.setFillColorRGB(*color);overlay.circle(x,12,3.6,fill=1,stroke=0)
                draw(text,x+12,8,color=(.16,.16,.16))
        overlay.save()
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_metadata({'/Title': 'Quiet-window diagnostics' if not grid else 'Spectral dispersion', '/Subject':'Larger plot labels; recorded data geometry retained'})
        destination=args.output/filename
        with destination.open('wb') as handle: writer.write(handle)
        tokens = lambda rows: Counter(re.findall(r'\w+|[^\w\s]', ' '.join(p['text'] for p in rows)))
        assert tokens(labels) == tokens(written)
        records.append({'file':filename,'original_sha256':sha(source),'output_sha256':sha(destination),'original_text_objects':len(labels),'non_text_original_operations_unchanged':True,'data_geometry_translation_points':[offset,0],'minimum_printed_font_size_points':min(p['printed_size'] for p in written),'included_width_points':396,'labels':written})
    (args.output/'figures'/'appendix_legibility_provenance.json').write_text(json.dumps({'font_sha256':sha(args.font),'figures':records},indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='labels'} for row in records],indent=2))
if __name__=='__main__': main()
