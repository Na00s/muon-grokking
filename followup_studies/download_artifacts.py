"""Restore the complete experiment binaries from the matching GitHub release."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def destination(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    target=(ROOT/Path(*path.parts)).resolve()
    if not target.is_relative_to(ROOT):raise ValueError(f'Path escapes study directory: {name}')
    return target


def verify_files(manifest):
    failures=[]
    for item in manifest['files']:
        p=destination(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            failures.append(item['path'])
    if failures:raise RuntimeError('Missing or changed files:\n'+'\n'.join(failures))
    print(f"Verified all {manifest['file_count']} binary files.")


def obtain_archive(manifest):
    cache=ROOT/'.download-cache';cache.mkdir(exist_ok=True)
    archive=cache/manifest['asset_name']
    if archive.is_file() and digest(archive)==manifest['archive_sha256']:return archive
    temporary=archive.with_suffix(archive.suffix+'.partial')
    request=urllib.request.Request(manifest['download_url'],headers={'User-Agent':'muon-grokking-reproduction'})
    print(f"Downloading {manifest['archive_bytes']/1e9:.2f} GB from {manifest['release_url']}",flush=True)
    try:
        with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response,out,4*1024*1024)
        if temporary.stat().st_size!=manifest['archive_bytes'] or digest(temporary)!=manifest['archive_sha256']:
            raise RuntimeError('Archive size or SHA-256 differs from the selected artifact manifest')
        temporary.replace(archive)
    finally:
        if temporary.exists():temporary.unlink()
    return archive


def restore(archive,manifest,verify_archive=False):
    if archive.stat().st_size!=manifest['archive_bytes'] or digest(archive)!=manifest['archive_sha256']:
        raise RuntimeError('Archive size or SHA-256 differs from the selected artifact manifest')
    expected={x['path']:x for x in manifest['files']};seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise RuntimeError(f'Unexpected or duplicate archive entry: {member.name}')
            item=expected[member.name];target=destination(member.name)
            if member.size!=item['bytes']:raise RuntimeError(f'Size mismatch: {member.name}')
            seen.add(member.name)
            if target.exists() and (not target.is_file() or digest(target)!=item['sha256']):
                raise FileExistsError(f'Preserving changed local file: {target}')
            source=tar.extractfile(member)
            checksum=hashlib.sha256()
            temporary=None
            out=None
            if not verify_archive and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.download.tmp')
                out=temporary.open('wb')
            try:
                with source:
                    for block in iter(lambda:source.read(4*1024*1024),b''):
                        checksum.update(block)
                        if out:out.write(block)
                if out:out.close()
                if checksum.hexdigest()!=item['sha256']:
                    raise RuntimeError(f'Content SHA-256 mismatch: {member.name}')
                if temporary:temporary.replace(target)
            finally:
                if out and not out.closed:out.close()
                if temporary and temporary.exists():temporary.unlink()
    if seen!=set(expected):raise RuntimeError('Archive is missing declared files')
    print(f"Verified {'archive entries' if verify_archive else 'restored files'}: {len(seen)}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('ARTIFACTS.json'),help='Artifact manifest relative to the study directory, or an absolute path')
    parser.add_argument('--archive',type=Path,help='Use a previously downloaded archive')
    parser.add_argument('--verify-only',action='store_true',help='Check every local binary against the selected artifact manifest')
    parser.add_argument('--verify-archive',action='store_true',help='Verify archive entries without writing files')
    args=parser.parse_args()
    manifest_path=args.manifest if args.manifest.is_absolute() else ROOT/args.manifest
    manifest=json.loads(manifest_path.read_text())
    if args.verify_only:
        verify_files(manifest);return
    archive=args.archive if args.archive else obtain_archive(manifest)
    restore(archive,manifest,verify_archive=args.verify_archive)


if __name__=='__main__':main()
