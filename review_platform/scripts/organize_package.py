"""Organize a Reviewer Package by declared segment duration and sampling group.

Preserves native waveform bytes, task IDs, peaks, and instructions. Old packages
are never overwritten; the original descriptor and backup identity remain byte-for-byte unchanged.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile

def encode(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def organize(source,output,groups=None,core_duration=None):
 source,output=Path(source),Path(output)
 if source.resolve()==output.resolve() or output.exists():raise ValueError('Choose a new output; existing packages are preserved.')
 classifications={}
 if groups:
  with Path(groups).open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):
    key=row['task_id']
    if key in classifications:raise ValueError('Duplicate task classification: '+key)
    classifications[key]=row
 records=[]
 with zipfile.ZipFile(source) as outer:
  books=[n for n in outer.namelist() if n.endswith('.ecgreview')]
  if len(books)!=1:raise ValueError('Expected exactly one .ecgreview descriptor in Reviewer Package.')
  name=books[0];prefix=name[:-len(Path(name).name)]
  with zipfile.ZipFile(io.BytesIO(outer.read(name))) as book:
   manifest=json.loads(book.read('manifest.json'));items=json.loads(book.read('items.json'));original_descriptor=outer.read(name)
  if not items or len({x['id'] for x in items})!=len(items):raise ValueError('Invalid or duplicate task IDs.')
  unknown=set(classifications)-{x['id'] for x in items}
  if unknown:raise ValueError('Classification IDs not in this package: '+','.join(sorted(unknown)))
  payloads={}
  for item in items:
   meta=classifications.get(item['id'],{});duration=float(meta.get('duration_sec') or core_duration or item.get('durationGroupSec') or round(item['durationSec'],1))
   group=meta.get('sampling_group') or item.get('samplingGroup') or 'Unclassified'
   if group not in ['Random','Difficult','Unclassified'] or not 0<duration<=3600:raise ValueError('Invalid group for '+item['id'])
   original=item['file'];data=outer.read(prefix+original)
   if len(data)!=item['fileBytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Waveform integrity failed: '+item['id'])
   label=f'{duration:g}s';new=f'{label}/{group}/waveforms/{Path(original).name}'
   if new in payloads:raise ValueError('Duplicate output waveform path.')
   payloads[new]=data
   records.append(dict(task_id=item['id'],duration_sec=duration,sampling_group=group,original_file=original,new_file=new,waveform_sha256=item['sha256']))
  old_id=manifest['packageId'];package_hash=hashlib.sha256(original_descriptor).hexdigest()
  browse=dict(format='ecg-browse-index',version=1,packageHash=package_hash,items=[dict(id=r['task_id'],file=r['new_file'],durationGroupSec=r['duration_sec'],samplingGroup=r['sampling_group']) for r in records])
  root=old_id+'_grouped_Reviewer_Package/';output.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as out:
   out.writestr(root+old_id+'.ecgreview',original_descriptor)
   out.writestr(root+'browse_index.json',encode(browse))
   for path,data in payloads.items():out.writestr(root+path,data)
   out.writestr(root+'OPEN_PACKAGE.txt','Open the Cloudflare review website and drop this ZIP. Browse segment duration, then Random / Difficult.\nUnclassified means the source did not establish a sampling group; it does not mean easy.\nWaveform bytes, native sample positions and task IDs are unchanged.\nOriginal descriptor bytes and package ID/hash are preserved. Existing progress backups remain compatible on the same website and reviewer name.\n')
 with zipfile.ZipFile(output) as z:
  assert z.testzip() is None
  for row in records:assert hashlib.sha256(z.read(root+row['new_file'])).hexdigest()==row['waveform_sha256']
 receipt=dict(source=str(source),output=str(output),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),original_package_id=old_id,new_package_id=old_id,descriptor_sha256=package_hash,descriptor_byte_identical=True,backup_identity_preserved=True,tasks=len(items),waveforms_byte_identical=True,groups=records)
 output.with_suffix('.verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--groups-csv',type=Path,help='task_id,duration_sec,sampling_group from verified source classification');p.add_argument('--core-duration',type=float,help='Declared core segment duration; waveform and context remain intact')
 a=p.parse_args();r=organize(a.source,a.output,a.groups_csv,a.core_duration);print(json.dumps({k:v for k,v in r.items() if k!='groups'},ensure_ascii=False,indent=2))
