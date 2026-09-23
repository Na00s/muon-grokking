"""Check the complete release package after final manuscript verification."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/muon-grokking'
SUPP = REPO / 'followup_studies'
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/audited_submission'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    final = json.loads((HERE / 'final_verification.json').read_text())
    assert final['all_passed']
    pdf = OUT / 'Muon_Grokking_Audited.pdf'
    assert sha(pdf) == final['pdf_sha256']
    assert sha(REPO / 'paper/Muon_Grokking_Revised.pdf') == final['pdf_sha256']

    # Use Git's publication inventory so ignored binary/cache files stay outside
    # the text manifest. The binary descriptors hash their separate archives.
    listed = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'followup_studies'],
        cwd=REPO,
    ).decode().split('\0')
    records = []
    for name in sorted(set(listed) - {''}):
        path = REPO / name
        if path == SUPP / 'manifest.json' or not path.is_file():
            continue
        assert path.suffix not in {'.pt', '.pth', '.npz', '.npy'}, name
        records.append(dict(path=path.relative_to(SUPP).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    binary_manifests = ['ARTIFACTS.json', 'CAUSAL_ARTIFACTS.json', 'FOURIER_HORIZON_ARTIFACTS.json']
    manifest = dict(file_count=len(records), total_bytes=sum(r['bytes'] for r in records),
                    files=records, binary_manifests=binary_manifests)
    (SUPP / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(any(r['path'] == name for r in records) for name in binary_manifests)
    assert any(r['path'].startswith('work/long_horizon/runs/') for r in records)
    assert any(r['path'].startswith('work/fourier_control/') for r in records)

    descriptor = json.loads((SUPP / 'FOURIER_HORIZON_ARTIFACTS.json').read_text())
    archive = ROOT / 'work/publish' / descriptor['asset_name']
    python = '/opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python'
    commands = [
        [python, str(SUPP / 'download_artifacts.py'), '--manifest', 'FOURIER_HORIZON_ARTIFACTS.json', '--verify-only'],
        [python, str(SUPP / 'download_artifacts.py'), '--manifest', 'FOURIER_HORIZON_ARTIFACTS.json', '--archive', str(archive), '--verify-archive'],
    ]
    logs = []
    for command in commands:
        run = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
        logs.append(run.stdout)

    source_zip = OUT / 'ICLR_Submission_Muon_Grokking_Audited_Source.zip'
    assert sha(source_zip) == sha(REPO / 'paper/ICLR_Submission_Muon_Grokking_Revised_Source.zip')
    check_root = ROOT / 'work/publish/source_check'
    source = check_root / 'source'
    build = check_root / 'build'
    assert not source.exists(), 'Preserve any existing reproduction; choose a fresh check directory.'
    source.mkdir(parents=True)
    build.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        for item in z.infolist():
            relative = Path(item.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert hashlib.sha256(z.read(item)).hexdigest() == sha(REPO / 'paper/source' / relative)
        z.extractall(source)
    environment = os.environ.copy()
    environment['TECTONIC_CACHE_DIR'] = str(ROOT / 'work/submission_revision/tectonic_cache')
    run = subprocess.run(['tectonic', '--only-cached', '--untrusted', '--keep-logs',
                          '--outdir', str(build), str(source / 'main.tex')],
                         cwd=source, env=environment, text=True, capture_output=True, check=True)
    (check_root / 'compile.log').write_text(run.stdout + run.stderr)
    assert 'Overfull' not in (build / 'main.log').read_text()
    actual = [p.extract_text() for p in PdfReader(build / 'main.pdf').pages]
    expected = [p.extract_text() for p in PdfReader(pdf).pages]
    assert actual == expected, 'Fresh source archive build differs from reviewed PDF text.'
    result = dict(all_passed=True, published_text_files=len(records),
                  binary_files=descriptor['file_count'], archive_sha256=sha(archive),
                  source_archive_sha256=sha(source_zip), source_archive_files=len(names),
                  fresh_source_compile_text_identical=True, pages=len(actual),
                  reviewed_pdf_sha256=sha(pdf), binary_verification_logs=logs)
    (HERE / 'package_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'package_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    for base in [REPO / 'paper/audit/fourier_revision', OUT / 'audit/fourier_revision']:
        base.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / 'package_verification.json', base / 'package_verification.json')
        shutil.copy2(Path(__file__), base / 'verify_packaged_revision.py')
        shutil.copy2(check_root / 'compile.log', base / 'source_archive_compile.log')
    with zipfile.ZipFile(OUT / 'Muon_Grokking_Audit_Records.zip', 'w', zipfile.ZIP_DEFLATED) as audit_zip:
        for path in sorted((OUT / 'audit').rglob('*')):
            if path.is_file() and not any(part in path.parts for part in ['__pycache__', 'plot_cache', 'mpl_cache']):
                audit_zip.write(path, path.relative_to(OUT / 'audit'))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
