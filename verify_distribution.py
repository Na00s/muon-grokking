"""Verify the supplied overlay bytes against its distribution correspondence."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
manifest=json.loads((B/'packaging_manifest.json').read_text())
checks=[]
for row in manifest['files']:
 p=B/'followup_studies/work'/row['study']/row['path']
 assert p.is_file(),str(p.relative_to(B))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['supplied_sha256'],str(p.relative_to(B))
 assert p.stat().st_size==row['supplied_bytes']
 checks.append(str(p.relative_to(B)))
for row in manifest['portable_script_correspondence']:
 p=B/row['anonymization_only_archive']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['archive_sha256'],str(p.relative_to(B))
 checks.append(str(p.relative_to(B)))
for row in manifest['generated_distribution_files']:
 p=B/row['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
 assert p.stat().st_size==row['bytes']
 checks.append(row['path'])
print(json.dumps({'status':'passed','verified_files':len(checks),'scope':'Supplied file hashes and sizes. Historical protocols retain their pre-anonymization hashes.'},indent=2))
