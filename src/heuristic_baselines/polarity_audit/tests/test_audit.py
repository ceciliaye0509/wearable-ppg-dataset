import csv
from contextlib import closing
import gzip
import hashlib
from email.parser import BytesParser
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_audit import stratified_sample
from analyze_audit import summarize
from serve_audit import COOKIE, initialize_database, make_handler, read_ratings
from peak_assistance import suggest_peaks


class FakeSocket:
    def __init__(self, request):
        self.request, self.output = request, bytearray()
    def makefile(self, *args):
        return io.BytesIO(self.request)
    def sendall(self, value):
        self.output.extend(value)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='polarity-audit-test-')
        self.db = Path(self.folder.name) / 'audit.sqlite3'
        initialize_database(self.db, 'fixed-packet-digest')
        self.packet = {'tasks': [dict(id='P2_R001_B00006', participant='P2', t=[0,.5,1], x=[1,2,1],
                                      focusStartSec=0, focusEndSec=1, secret_prediction='negative')],
                       'protocol': {'strata': [dict(participant='P2', population_minutes=10, sampled_minutes=1, inverse_probability_weight=10)]}}
        self.handler = make_handler(self.packet, self.db, 'audit-access-code', b'long-session-secret-for-audit-test')
        status, headers, _ = self.request('POST', '/api/login', {'reviewer':'QA', 'accessCode':'audit-access-code'})
        self.assertEqual(status, 200)
        self.cookie = headers['Set-Cookie'].split(';')[0]

    def tearDown(self):
        self.folder.cleanup()

    def request(self, method, path, body=None, extra=None):
        payload = json.dumps(body).encode() if body is not None else b''
        headers = {'Host':'127.0.0.1:8766', 'Content-Length':str(len(payload))}
        if hasattr(self, 'cookie'):
            headers['Cookie'] = self.cookie
        if body is not None:
            headers['Content-Type'] = 'application/json'
        headers.update(extra or {})
        raw = f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{k}: {v}\r\n' for k,v in headers.items()) + '\r\n'
        socket = FakeSocket(raw.encode() + payload)
        self.handler(socket, ('127.0.0.1',1234), SimpleNamespace(server_name='test',server_port=8766))
        head, body_bytes = bytes(socket.output).split(b'\r\n\r\n',1)
        first, header_bytes = head.split(b'\r\n',1)
        return int(first.split()[1]), BytesParser().parsebytes(header_bytes + b'\r\n\r\n'), body_bytes

    def answer(self, version=0, **changes):
        result = dict(qrs_readability='readable', direction_determinability='no', direction='undetermined',
                      confidence='high', notes='', status='complete', version=version)
        result.update(changes)
        return result

    def test_two_assessments_are_independent_and_predictions_are_hidden(self):
        status, _, payload = self.request('GET','/api/task/P2_R001_B00006')
        self.assertEqual(status,200)
        self.assertNotIn('secret_prediction',json.loads(payload))
        status, _, payload = self.request('POST','/api/task/P2_R001_B00006/review',self.answer())
        self.assertEqual(status,200)
        self.assertEqual(json.loads(payload)['direction_determinability'],'no')
        self.assertEqual(read_ratings(self.db)['P2_R001_B00006']['qrs_readability'],'readable')
        _, _, raw = self.request('GET','/api/export.csv')
        row = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))[0]
        self.assertEqual(row['qrs_readability'],'readable')
        self.assertEqual(row['direction_determinability'],'no')

    def test_incomplete_or_conflicting_review_cannot_overwrite(self):
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review', self.answer(direction_determinability=''))[0],400)
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review', self.answer())[0],200)
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review', self.answer(qrs_readability='unreadable'))[0],409)
        self.assertEqual(read_ratings(self.db)['P2_R001_B00006']['qrs_readability'],'readable')

    def test_revision_history_and_packet_binding_preserve_existing_ratings(self):
        self.request('POST','/api/task/P2_R001_B00006/review', self.answer())
        self.request('POST','/api/task/P2_R001_B00006/review', self.answer(1,qrs_readability='partially_readable'))
        with closing(sqlite3.connect(self.db)) as db:
            prior=json.loads(db.execute('SELECT snapshot_json FROM revisions').fetchone()[0])
        self.assertEqual(json.loads(prior['answer_json'])['qrs_readability'],'readable')
        initialize_database(self.db,'fixed-packet-digest')
        with self.assertRaises(ValueError): initialize_database(self.db,'different-packet')
        self.assertEqual(read_ratings(self.db)['P2_R001_B00006']['version'],2)

    def test_separate_cookie_and_authentication(self):
        self.assertTrue(self.cookie.startswith(COOKIE+'='))
        self.assertEqual(self.request('GET','/api/tasks',extra={'Cookie':''})[0],401)
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review',self.answer(),extra={'Origin':'https://other.example'})[0],403)

    def test_peak_marks_roundtrip_export_and_legacy_save_preservation(self):
        answer = self.answer(peak_indices=[1], peak_origin='mixed', peaks_reviewed=True, assistance_shown=True)
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review',answer)[0],200)
        _, _, payload = self.request('GET','/api/task/P2_R001_B00006')
        task = json.loads(payload)
        self.assertEqual(task['rating']['peak_indices'],[1])
        self.assertEqual(task['assistance']['peak_indices'],[])
        self.assertNotIn('raw_polarity',task['assistance'])
        # A pre-upgrade browser saves only the two assessments: existing peaks survive.
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review',self.answer(1))[0],200)
        saved = read_ratings(self.db)['P2_R001_B00006']
        self.assertEqual(saved['peak_indices'],[1])
        self.assertTrue(saved['peaks_reviewed'])
        self.assertTrue(saved['assistance_shown'])
        _, _, raw = self.request('GET','/api/export.csv')
        row = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))[0]
        self.assertEqual(json.loads(row['peak_times_sec']),[.5])
        self.assertEqual(json.loads(row['peak_indices']),[1])
        self.assertTrue(row['assistance_source_sha256'])

    def test_invalid_peaks_cannot_overwrite_ratings(self):
        for peaks in [[-1],[3],[2],[True],[.5],[1,1],'bad']:
            body = self.answer(peak_indices=peaks, peak_origin='manual', peaks_reviewed=False, assistance_shown=True)
            self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review',body)[0],400)
        self.assertEqual(read_ratings(self.db),{})

    def test_showing_assistance_never_fills_answers_or_stores_reviews(self):
        self.request('GET','/api/task/P2_R001_B00006')
        self.assertEqual(read_ratings(self.db),{})
        body = self.answer(qrs_readability='', direction_determinability='', peak_indices=[1],
                           peak_origin='algorithm', peaks_reviewed=False, assistance_shown=True)
        self.assertEqual(self.request('POST','/api/task/P2_R001_B00006/review',body)[0],400)

    def test_assistance_is_sign_equivariant_and_within_focus(self):
        t = np.arange(-2,22,1/130)
        x = sum(900*np.exp(-((t-second)/.018)**2) for second in np.arange(-1,22,.8))
        task = dict(t=t.tolist(),x=x.tolist(),focusStartSec=0,focusEndSec=20)
        positive = suggest_peaks(task)
        negative = suggest_peaks(dict(task,x=(-x).tolist()))
        self.assertGreater(len(positive['peak_indices']),20)
        self.assertEqual(positive['peak_indices'],negative['peak_indices'])
        self.assertTrue(all(0 <= t[i] < 20 for i in positive['peak_indices']))


class SamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_stratified_without_replacement(self):
        rows=[dict(participant=p,run_index=0,block_index=i,raw_polarity='ambiguous') for p,n in [('P2',50),('P5',2)] for i in range(n)]
        a,strata=stratified_sample(rows,8,20261023)
        b,_=stratified_sample(list(reversed(rows)),8,20261023)
        self.assertEqual(set(a),set(b))
        self.assertEqual(len(a),10)
        self.assertEqual(strata[0]['inverse_probability_weight'],50/8)
        self.assertEqual(strata[1]['inverse_probability_weight'],1)

    def test_no_prevalence_claim_before_all_planned_reviews_complete(self):
        sample=[dict(id='a',participant='P2',inverse_probability_weight=10,new_raw_polarity='positive'),dict(id='b',participant='P2',inverse_probability_weight=10,new_raw_polarity='positive')]
        strata=[dict(participant='P2',population_minutes=20,sampled_minutes=2)]
        first=dict(status='complete',qrs_readability='readable',direction_determinability='yes',direction='positive')
        self.assertIsNone(summarize(sample,strata,{'a':first})['population_estimates'])
        full=summarize(sample,strata,{'a':first,'b':dict(first,qrs_readability='unreadable')})
        self.assertEqual(full['population_estimates']['qrs_readability']['readable']['proportion'],.5)

    def test_safety_controls_are_not_pooled_into_old_alert_prevalence(self):
        sample=[dict(id='a',participant='P2',inverse_probability_weight=10,new_raw_polarity='positive'),dict(id='b',participant='P2',inverse_probability_weight=10,new_raw_polarity='positive'),dict(id='c',participant='P2',inverse_probability_weight=100,new_raw_polarity='ambiguous',sampling_stratum='new_uncertainty')]
        strata=[dict(participant='P2',population_minutes=20,sampled_minutes=2)]
        rating=dict(status='complete',qrs_readability='readable',direction_determinability='yes',direction='positive')
        result=summarize(sample,strata,{'a':rating,'b':rating,'c':dict(rating,qrs_readability='unreadable')})
        self.assertEqual(result['population_estimates']['qrs_readability']['readable']['proportion'],1)
        self.assertEqual(result['safety_completed_minutes'],1)


class GeneratedPacketChecks(unittest.TestCase):
    def test_actual_packet_and_known_p2_regression(self):
        folder=Path(__file__).resolve().parents[2]/'outputs/beatwise_polarity_audit_v2'
        if not (folder/'audit_packet.json.gz').is_file():
            self.skipTest('Generated audit packet is not available')
        with gzip.open(folder/'audit_packet.json.gz','rt') as f: packet=json.load(f)
        with (folder/'sampling_manifest.csv').open() as f: sample=list(csv.DictReader(f))
        self.assertEqual({t['id'] for t in packet['tasks']},{r['id'] for r in sample})
        self.assertEqual(len({t['id'] for t in packet['tasks']}),len(packet['tasks']))
        primary=[r for r in sample if r['sampling_stratum']=='old_ambiguous']
        self.assertAlmostEqual(sum(float(r['inverse_probability_weight']) for r in primary),packet['protocol']['population_minutes'])
        source=Path(__file__).resolve().parents[1]/'beatwise.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),packet['protocol']['method_source_sha256'])
        for task in packet['tasks']:
            t,x=np.asarray(task['t']),np.asarray(task['x'])
            self.assertEqual(t.shape,x.shape)
            self.assertTrue(np.isfinite(x).all() and np.isfinite(t).all() and np.all(np.diff(t)>0))
            self.assertNotIn('new_raw_polarity',task)
        known=json.loads((folder/'known_p2_example.json').read_text())
        self.assertEqual(known['old']['raw_polarity'],'ambiguous')
        self.assertEqual(known['candidate']['raw_polarity'],'positive')
        self.assertEqual(known['candidate']['event_count'],61)
        self.assertEqual(known['candidate']['negative_votes'],0)


if __name__ == '__main__': unittest.main()
