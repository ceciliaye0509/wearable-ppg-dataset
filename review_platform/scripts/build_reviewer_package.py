"""Convert a selected existing ECG packet to an isolated, allowlisted local review package.

Never reads annotations.sqlite3 or changes source data. Research mapping is kept outside ZIP.
"""
import argparse
import gzip
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def build(source, output, package_id, title, kind='auto', blind=False):
    assert ID.fullmatch(package_id), 'Invalid package ID'
    assert not output.exists(), 'Output already exists; never silently replace a distributed package'
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rt') as f:
        original = json.load(f)
    clips = original['clips']
    items, mapping, files = [], [], {}
    for i, clip in enumerate(clips, 1):
        task_id = f'task_{i:04d}'
        if clip.get('typeTargets'):
            raise ValueError('This tool only accepts R-peak position review packets')
        mode = 'rpeaks'
        t, x = clip['t'], clip['x']
        assert len(t) == len(x) and len(t) >= 2
        assert all(t[j] > t[j-1] for j in range(1, len(t))), 'Do not silently sort timestamps'
        times = [round(v-t[0], 6) for v in t]
        waveform = encoded({'t': times, 'x': x})
        assert len(waveform) <= 10*1024*1024
        raw = gzip.compress(waveform, mtime=0)
        file = f'waveforms/{task_id}.json.gz'
        files[file] = raw
        # Never copy predictions, confidence, classifications, other reviewers' answers or absolute dates.
        peaks = [] if blind else sorted(set(clip.get('suggested', [])))
        items.append({'id': task_id, 'participant': str(clip.get('participant', 'UNKNOWN')),
                      'title': f'任务 {i:02d}', 'mode': mode, 'file': file,
                      'sha256': hashlib.sha256(raw).hexdigest(), 'fileBytes': len(raw),
                      'sampleCount': len(times), 'durationSec': times[-1], 'targets': [],
                      'contextPeaks': [],
                      'initialPeaks': peaks,
                      'initialPeakSource': 'none' if blind else ('mixed' if any(v == 'automatic_context' for v in clip.get('contextPeakProvenance', {}).values()) else ('prior_human' if clip.get('reviewSource') else 'algorithm')),
                      'contextPeakSources': {},
                      'instructions': '逐页检查真实ECG，修订R位置并标记不可读区间。不应为了让RR规则而移动清晰QRS。'})
        mapping.append({'task_id': task_id, 'original_clip_id': clip['id'],
                        'original_window_start': clip.get('windowStart'),
                        'original_clip_start': clip.get('clipStart'),
                        'target_mapping': []})
    manifest = {'format': 'ecg-review', 'version': 1, 'packageId': package_id,
                'title': title, 'createdAt': datetime.now(timezone.utc).isoformat(),
                'reviewMode': 'blind' if blind else 'assisted',
                'instructions': '独立审核。标注仅保存在本机，请定期备份；完成后将结果ZIP交给研究团队。'}
    import io
    book = io.BytesIO()
    with zipfile.ZipFile(book, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('manifest.json', encoded(manifest))
        z.writestr('items.json', encoded(items))
    assert len(book.getvalue()) <= 25*1024*1024
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        root = package_id + '_Reviewer_Package/'
        z.writestr(root + package_id + '.ecgreview', book.getvalue())
        for name, data in files.items():
            z.writestr(root+name, data)
        z.writestr(root+'开始复核.txt', '1. 解压整个任务包。\n2. 打开团队的审核网址，输入分配的审核员编号。\n3. 选择此文件夹，波形与标注不会上传。\n4. 定期下载.ecgprogress备份；完成后导出结果ZIP交给团队。\n带参考峰的任务属于非盲审核。\n')
    output.chmod(0o600)
    private_mapping = output.with_suffix('.research_mapping.json')
    private_mapping.write_text(json.dumps({'package_sha256': hashlib.sha256(book.getvalue()).hexdigest(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'mapping': mapping}, ensure_ascii=False, indent=2))
    private_mapping.chmod(0o600)
    print(json.dumps({'package': str(output), 'tasks': len(items), 'targets': sum(len(t['targets']) for t in items), 'package_sha256': hashlib.sha256(book.getvalue()).hexdigest(), 'research_mapping_in_reviewer_zip': False, 'source_modified': False}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package-id', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--kind', choices=['auto', 'rpeaks'], default='auto')
    p.add_argument('--blind', action='store_true')
    a=p.parse_args()
    build(a.source.resolve(), a.output.resolve(), a.package_id, a.title, a.kind, a.blind)
