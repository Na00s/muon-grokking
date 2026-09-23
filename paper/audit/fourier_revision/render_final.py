"""Render the final PDF and contact sheets for a subsequent visual review."""
from pathlib import Path
import hashlib
import json
import subprocess
from PIL import Image, ImageDraw
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
pdf = HERE / 'build/main.pdf'
digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
folder = HERE / 'final_preview' / digest[:12]
folder.mkdir(parents=True, exist_ok=True)
poppler = '/Users/alipro/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
subprocess.run([poppler, '-scale-to', '1500', '-png', str(pdf), str(folder / 'page')], check=True)
pages = sorted(folder.glob('page-*.png'))
assert len(pages) == len(PdfReader(pdf).pages)
sheets = []
for start in range(0, len(pages), 4):
    sheet = Image.new('RGB', (1540, 2050), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for local, path in enumerate(pages[start:start+4]):
        with Image.open(path) as source:
            thumbnail = source.convert('RGB')
            thumbnail.thumbnail((750, 970), Image.Resampling.LANCZOS)
        x = 10 + (local % 2) * 770
        y = 32 + (local // 2) * 1020
        draw.text((x, y - 22), f'Page {start + local + 1}', fill='black')
        sheet.paste(thumbnail, (x, y))
    name = folder / f'sheet-{start//4+1:02d}.png'
    sheet.save(name)
    sheets.append(str(name))
index = dict(pdf_sha256=digest, page_count=len(pages), pages=[str(p) for p in pages], sheets=sheets,
             status='Rendered; visual inspection is still required.')
(folder / 'render_manifest.json').write_text(json.dumps(index, indent=2) + '\n')
(HERE / 'latest_render.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps(index, indent=2))
