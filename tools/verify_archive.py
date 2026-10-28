"""Read-only integrity checks for local or explicitly relocated historical assets."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def locations():
    local=ROOT/'archive/manifest_20261024.json'
    if local.is_file():
        return dict(manifest=local,external_root=None,local_roots=(),transfer_manifest=None,additional_manifests=())
    pointer=json.loads((ROOT/'config/storage.json').read_text(encoding='utf-8-sig'))
    external=Path(pointer['external_root']).resolve()
    if not external.is_absolute() or external==ROOT.resolve() or external.is_relative_to(ROOT.resolve()):
        raise ValueError('Historical archive must be outside the project')
    return dict(manifest=external/Path(pointer['legacy_manifest'].replace('\\','/')),
                external_root=external,local_roots=tuple(pointer.get('legacy_local_roots',[])),
                transfer_manifest=Path(pointer['transfer_manifest']),
                additional_manifests=tuple(pointer.get('additional_manifests',[])))

def safe_path(base,relative):
    path=(base/Path(relative.replace('\\','/'))).resolve()
    if not path.is_relative_to(base.resolve()):raise ValueError('Archive path escapes declared storage')
    return path

def matches(path,digest):
    if not path.is_file():return False
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()==digest

def verify(include_transferred=False):
    location=locations()
    manifest=json.loads(location['manifest'].read_text(encoding='utf-8-sig'))
    failures=[]
    for entry in manifest['entries']:
        relative=Path(entry['stored'].replace('\\','/'))
        base=(location['external_root'] if location['external_root'] and
              relative.parts[0] not in location['local_roots'] else ROOT)
        if not matches(safe_path(base,entry['stored']),entry['sha256']):failures.append(entry['stored'])
    additional_checked=0
    for relative_manifest in location['additional_manifests']:
        supplement_path=safe_path(location['external_root'],relative_manifest)
        supplement=json.loads(supplement_path.read_text(encoding='utf-8-sig'))
        if not supplement.get('completed'):failures.append(relative_manifest+':not_completed')
        for entry in supplement['files']:
            if not matches(safe_path(location['external_root'],entry['relative']),entry['sha256']):
                failures.append(entry['relative'])
            additional_checked+=1
    checked=0
    if include_transferred and location['transfer_manifest']:
        transfer=json.loads(location['transfer_manifest'].read_text(encoding='utf-8-sig'))
        if not transfer.get('completed'):failures.append('transfer_not_completed')
        for entry in transfer['files']:
            path=safe_path(location['external_root'],entry['relative'])
            if not matches(path,entry['sha256']):failures.append(entry['relative'])
            checked+=1
    return dict(files=len(manifest['entries']),additional_files_checked=additional_checked,transferred_files_checked=checked,
                storage_root=str(location['external_root'] or ROOT),all_preserved=not failures,failures=failures)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true',help='Also hash all transferred datasets/checkpoints (large).')
    arguments=parser.parse_args()
    result=verify(arguments.all);print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['all_preserved'] else 1)
