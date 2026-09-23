"""Package the four requested wording corrections after visual review."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/wording_revision'
REPO = ROOT / 'work/muon-grokking'
PAPER = REPO / 'paper'
SRC = PAPER / 'source'
OUT = ROOT / 'outputs/audited_submission'
BASE = 'c533e94ae301d1f28d2499d7f9902d3a5d8d3d75'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(source, destination):
    files = [p for p in sorted(source.rglob('*')) if p.is_file()
             and not any(x in p.parts for x in ['__pycache__', 'plot_cache', 'mpl_cache'])]
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(source))
    return files


def main():
    pdf = WORK / 'build/main.pdf'
    independent = json.loads((WORK / 'independent_audit.json').read_text())
    pixels = json.loads((WORK / 'figure_pixel_audit.json').read_text())
    assert independent['all_passed'], independent
    assert sha(pdf) in json.dumps(independent), 'Independent audit targets another PDF'
    assert pixels['pixels_outside_label_regions_identical']
    log = (WORK / 'build/main.log').read_text()
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    assert r'\newlabel{sec:main-end}{{9}{9}' in (WORK / 'build/main.aux').read_text()
    old = json.loads((PAPER / 'audit/fourier_revision/final_verification.json').read_text())
    assert old['all_passed']
    # Confirm that the experiment files and all appendix sources are untouched.
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE], cwd=REPO, text=True).splitlines()
    assert all(p.startswith('paper/') for p in changed), changed
    source_changes = [p for p in changed if p.startswith('paper/source/')]
    assert set(source_changes) <= {'paper/source/main.tex', 'paper/source/README.md',
                                   'paper/source/figure5_modes.pdf'}, source_changes
    before = PdfReader(PAPER / 'Muon_Grokking_Revised.pdf')
    after = PdfReader(pdf)
    assert len(before.pages) == len(after.pages) == 28
    for n in [0, 1, *range(10, 28)]:
        assert before.pages[n].extract_text() == after.pages[n].extract_text(), n + 1
    # The source archive is self-contained and is compiled from a fresh extraction.
    source_zip = PAPER / 'ICLR_Submission_Muon_Grokking_Revised_Source.zip'
    files = bundle(SRC, source_zip)
    fresh = WORK / 'source_rebuild_anonymous'
    assert not fresh.exists(), 'Preserve previous rebuild; choose a fresh location'
    (fresh / 'source').mkdir(parents=True)
    (fresh / 'build').mkdir()
    with zipfile.ZipFile(source_zip) as archive:
        assert len(archive.namelist()) == len(files)
        for name in archive.namelist():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(SRC / name)
        archive.extractall(fresh / 'source')
    env = os.environ.copy()
    env['TECTONIC_CACHE_DIR'] = str(ROOT / 'work/submission_revision/tectonic_cache')
    run = subprocess.run(['tectonic', '--only-cached', '--untrusted', '--keep-logs',
                          '--outdir', str(fresh / 'build'), 'main.tex'],
                         cwd=fresh / 'source', env=env, text=True, capture_output=True, check=True)
    (WORK / 'source_archive_compile.log').write_text(run.stdout + run.stderr)
    rebuilt = PdfReader(fresh / 'build/main.pdf')
    assert [p.extract_text() for p in rebuilt.pages] == [p.extract_text() for p in after.pages]
    assert 'Overfull' not in (fresh / 'build/main.log').read_text()
    record = dict(all_passed=True, scope='Four requested wording and artwork corrections',
                  baseline_commit=BASE, pdf_sha256=sha(pdf), pages=28, scientific_main_pages=9,
                  source_archive_sha256=sha(source_zip), source_archive_files=len(files),
                  source_archive_rebuild_text_identical=True, experiment_sources_unchanged=True,
                  other_pages_text_identical=[1, 2, *range(11, 29)],
                  visual_review=dict(pages=list(range(3, 11)), figure4_full_size=True,
                                     result='No clipping, overlap, missing labels, or broken layout'),
                  independent_audit=independent, figure_pixel_audit=pixels,
                  previous_experimental_audit=dict(path='audit/fourier_revision/final_verification.json',
                                                  sha256=sha(PAPER / 'audit/fourier_revision/final_verification.json'),
                                                  all_passed=True),
                  source_sha256={p.relative_to(SRC).as_posix(): sha(p) for p in files})
    audit = PAPER / 'audit/wording_revision'
    audit.mkdir(parents=True, exist_ok=True)
    (WORK / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    for name in ['verification.json', 'independent_audit.json', 'figure_pixel_audit.json',
                 'package_revision.py', 'source_archive_compile.log']:
        shutil.copy2(WORK / name, audit / name)
    shutil.copy2(WORK / 'verification.json', PAPER / 'verification.json')
    shutil.copy2(pdf, PAPER / 'Muon_Grokking_Revised.pdf')
    patch = subprocess.check_output(['git', 'diff', BASE, '--', 'paper/source/main.tex',
                                     'paper/source/README.md', 'paper/revision_addendum.md'], cwd=REPO)
    (PAPER / 'wording_revision.patch').write_bytes(patch)
    shutil.copytree(SRC, OUT / 'source', dirs_exist_ok=True)
    shutil.copytree(PAPER / 'audit', OUT / 'audit', dirs_exist_ok=True)
    for name in ['verification.json', 'revision_addendum.md', 'wording_revision.patch']:
        shutil.copy2(PAPER / name, OUT / name)
    shutil.copy2(pdf, OUT / 'Muon_Grokking_Audited.pdf')
    shutil.copy2(source_zip, OUT / 'ICLR_Submission_Muon_Grokking_Audited_Source.zip')
    shutil.copy2(WORK / 'verification.json', OUT / 'package_verification.json')
    bundle(OUT / 'audit', OUT / 'Muon_Grokking_Audit_Records.zip')
    print(json.dumps({k: record[k] for k in ['all_passed', 'pdf_sha256', 'source_archive_sha256',
                                            'pages', 'scientific_main_pages', 'source_archive_files']}, indent=2))


if __name__ == '__main__':
    main()
