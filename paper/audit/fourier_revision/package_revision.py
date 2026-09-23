"""Publish the completed Fourier controls and long-horizon extensions locally.

This script only runs once the prescribed training horizons and audits pass.
Binary evidence is bundled separately for a GitHub release. Existing evidence
and its original audit are retained.
"""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT/'work/muon-grokking'
SUPP = REPO/'followup_studies'
PAPER = REPO/'paper'
SRC = PAPER/'source'
WORK = ROOT/'work/fourier_revision'
OUT = ROOT/'outputs/audited_submission'
TAG = 'fourier-horizon-study-2026-09-22'
ASSET = ROOT/'work/publish/muon-grokking-fourier-horizon-artifacts-2026-09-22.tar.gz'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def copy_tree(source, dest):
    for p in sorted(source.rglob('*')):
        if not p.is_file() or any(x in p.parts for x in ['__pycache__', 'plot_cache', 'mpl_cache']):
            continue
        q = dest/p.relative_to(source)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)

def bundle(directory, destination):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and not any(x in p.parts for x in ['__pycache__','plot_cache','mpl_cache']):
                archive.write(p, p.relative_to(directory))

def main():
    long = ROOT/'work/long_horizon'
    fourier = ROOT/'work/fourier_control'
    completion = json.loads((long/'completion.json').read_text())
    assert completion['status'] == 'completed'
    for p in [fourier/'results/verification.json', fourier/'results/frequency_verification.json']:
        assert json.loads(p.read_text())['all_passed'], p
    assert (long/'summary.json').exists()
    assert (WORK/'control_audit/independent_verification.json').exists()
    assert (WORK/'final_verification.json').exists()
    verify = json.loads((WORK/'final_verification.json').read_text())
    assert verify['all_passed'] and verify['visual_review_complete']
    assert all(c['passed'] for c in verify['checks'])
    assert sha(WORK/'build/main.pdf')==verify['pdf_sha256'], 'Reviewed PDF changed after verification'
    source_hashes={p.relative_to(SRC).as_posix():sha(p) for p in sorted(SRC.rglob('*'))
                   if p.is_file() and '__pycache__' not in p.parts and 'plot_cache' not in p.parts}
    assert source_hashes==verify['source_sha256'], 'Manuscript source changed after verification'
    for name,path in [('fourier_primary',fourier/'results/verification.json'),
                      ('fourier_secondary',fourier/'results/frequency_verification.json'),
                      ('independent_primary',WORK/'control_audit/independent_verification.json'),
                      ('independent_secondary',WORK/'control_audit/secondary_direct_verification.json')]:
        assert sha(path)==verify['new_control_audits'][name]['sha256'], f'Audit changed after final verification: {name}'
    for key,name in [('sha256','verification.json'),('summary_sha256','summary.json'),
                     ('runner_sha256','run_long_horizon.py'),('protocol_sha256','protocol.json'),
                     ('checkpoint_manifest_sha256','checkpoint_manifest.json')]:
        assert sha(long/name)==verify['long_horizon_audit'][key], f'Long-horizon evidence changed after verification: {name}'
    reproduction=json.loads((WORK/'publication_reproduction/verification.json').read_text())
    assert reproduction['all_passed'] and reproduction['same_memorizer_results'] and reproduction['figure_pixels_identical']
    # Verify the actual experiment artifacts against the completed audit before copying.
    for item in json.loads((long/'checkpoint_manifest.json').read_text())['artifacts']:
        path=long/item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'], path
    for group in ['source_sha256','input_sha256','output_sha256']:
        for name,digest in json.loads((fourier/'results/publication_manifest.json').read_text())[group].items():
            assert sha(fourier/name)==digest, f'Fourier publication manifest mismatch: {name}'
    ASSET.parent.mkdir(parents=True,exist_ok=True)
    for name in ['fourier_control', 'long_horizon']:
        copy_tree(ROOT/'work'/name, SUPP/'work'/name)

    records = []
    for name in ['fourier_control','long_horizon']:
        for original in sorted((ROOT/'work'/name).rglob('*')):
            if original.is_file() and original.suffix in ['.pt','.pth','.npz','.npy']:
                p=SUPP/'work'/name/original.relative_to(ROOT/'work'/name)
                assert sha(p)==sha(original), p
                records.append(dict(path=p.relative_to(SUPP).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
    assert records
    with tarfile.open(ASSET, 'w:gz', compresslevel=6) as archive:
        for item in records:
            archive.add(SUPP/item['path'], arcname=item['path'], recursive=False)
    descriptor = dict(schema_version=1, repository='Na00s/muon-grokking', release_tag=TAG,
        release_url=f'https://github.com/Na00s/muon-grokking/releases/tag/{TAG}',
        asset_name=ASSET.name,
        download_url=f'https://github.com/Na00s/muon-grokking/releases/download/{TAG}/{ASSET.name}',
        archive_bytes=ASSET.stat().st_size, archive_sha256=sha(ASSET), file_count=len(records),
        uncompressed_bytes=sum(r['bytes'] for r in records), files=records,
        scope='Every saved binary from the Fourier-control and six completed long-horizon extensions, including resume verification states.',
        provenance='Published experiment scripts, protocols, results, and binaries preserve their source bytes. Absolute provenance paths identify the original runtime; reproduction commands use the released relative layout.')
    (SUPP/'FOURIER_HORIZON_ARTIFACTS.json').write_text(json.dumps(descriptor,indent=2)+'\n')

    audit = PAPER/'audit/fourier_revision'
    copy_tree(WORK/'control_audit', audit/'control_audit')
    for name in ['interpretation_changes.json','final_verification.json','visual_review.json','novelty_positioning.md',
                 'incorporate_completed_horizons.py','write_completed_docs.py','verify_revision.py','package_revision.py','render_final.py']:
        if (WORK/name).exists():
            shutil.copy2(WORK/name,audit/name)
    for name in ['runner_verification.json','verification.json','failure_verification.json','summary.json','protocol.json']:
        if (long/name).exists():
            shutil.copy2(long/name, audit/('long_horizon_'+name))
    for name in ['verification.json','frequency_verification.json','summary.json','manifest.json','frequency_manifest.json','publication_manifest.json']:
        shutil.copy2(fourier/'results'/name,audit/('fourier_'+name))
    copy_tree(WORK/'publication_reproduction', audit/'publication_reproduction')
    shutil.copy2(WORK/'revision_addendum.md',PAPER/'revision_addendum.md')
    shutil.copy2(WORK/'build/main.pdf', PAPER/'Muon_Grokking_Revised.pdf')
    shutil.copy2(WORK/'final_verification.json', PAPER/'verification.json')
    for baseline, name in [(ROOT/'work/submission_revision/original','targeted_revision.patch'),
                           (ROOT/'work/submission_audit/input','audit_revision.patch'),
                           (WORK/'input','fourier_revision.patch')]:
        delta=[]
        for p in sorted(SRC.rglob('*')):
            if p.is_file() and p.suffix in ['.tex','.bib','.py','.md']:
                previous=baseline/p.relative_to(SRC)
                a=previous.read_text().splitlines(True) if previous.exists() else []
                delta.extend(difflib.unified_diff(a,p.read_text().splitlines(True),
                    fromfile='a/'+p.relative_to(SRC).as_posix(),tofile='b/'+p.relative_to(SRC).as_posix()))
        (PAPER/name).write_text(''.join(delta))
    bundle(SRC, PAPER/'ICLR_Submission_Muon_Grokking_Revised_Source.zip')

    OUT.mkdir(parents=True,exist_ok=True)
    copy_tree(SRC,OUT/'source')
    copy_tree(PAPER/'audit',OUT/'audit')
    for name in ['audit_report.md','revision_notes.md','revision_addendum.md','targeted_revision.patch','audit_revision.patch','fourier_revision.patch','verification.json']:
        shutil.copy2(PAPER/name,OUT/name)
    shutil.copy2(PAPER/'Muon_Grokking_Revised.pdf',OUT/'Muon_Grokking_Audited.pdf')
    shutil.copy2(PAPER/'ICLR_Submission_Muon_Grokking_Revised_Source.zip',OUT/'ICLR_Submission_Muon_Grokking_Audited_Source.zip')
    for p in SRC.glob('*.png'):
        if p.name.startswith(('followup_', 'fourier_')):
            (OUT/'figures').mkdir(exist_ok=True)
            (PAPER/'figures').mkdir(exist_ok=True)
            shutil.copy2(p,OUT/'figures'/p.name)
            shutil.copy2(p,PAPER/'figures'/p.name)
    bundle(OUT/'audit',OUT/'Muon_Grokking_Audit_Records.zip')
    print(json.dumps(dict(output=str(OUT),binary_files=len(records),archive_bytes=ASSET.stat().st_size,
                         pdf_sha256=sha(OUT/'Muon_Grokking_Audited.pdf')),indent=2))

if __name__=='__main__':
    main()
