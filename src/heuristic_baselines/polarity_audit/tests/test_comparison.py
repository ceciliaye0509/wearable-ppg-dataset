import csv
from contextlib import closing
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from email.parser import BytesParser
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from build_comparison import sample_disagreements, overlaps_reserved, trace_old, trace_new
from comparison_review import validate_comparison
from serve_audit import initialize_database,make_handler,read_ratings,save_rating
from test_audit import FakeSocket


class PairedSamplingTests(unittest.TestCase):
    def rows(self):
        return [dict(participant=p,run_index='0',block_index=str(i),start_ms=str(60000*i),end_ms=str(60000*(i+1)),
                     old_raw_polarity='ambiguous',raw_polarity='positive') for p,n in [('P2',20),('P3',2)] for i in range(n)]

    def test_stratified_random_unique_reproducible_and_no_reserved_context(self):
        rows=self.rows();reserved={'P2':[(60000,360000)]}
        a,strata=sample_disagreements(rows,reserved,8,20261028)
        b,_=sample_disagreements(list(reversed(rows)),reserved,8,20261028)
        self.assertEqual(a,b);self.assertEqual(len(a),10)
        self.assertEqual(len({(r['participant'],r['block_index']) for r in a}),10)
        self.assertFalse(any(overlaps_reserved(r,reserved) for r in a))
        self.assertEqual(sum(r['method_a']=='old' for r in a if r['participant']=='P2'),4)
        self.assertEqual(strata[0]['eligible_minutes'],15)
        self.assertEqual(strata[1]['sampled_minutes'],2)

    def test_matching_and_short_blocks_not_sampled_empty_participant_retained(self):
        rows=self.rows()+[dict(participant='P1',run_index='0',block_index='0',start_ms='0',end_ms='60000',old_raw_polarity='positive',raw_polarity='positive')]
        rows[0]['end_ms']='59000'
        selected,strata=sample_disagreements(rows,{},8)
        self.assertNotIn(('P2','0'),[(r['participant'],r['block_index']) for r in selected])
        self.assertEqual(strata[0]['sampled_minutes'],0)
        self.assertEqual(strata[0]['differing_blocks'],0)

    def test_vote_tracing_matches_frozen_functions_and_sign_reversal(self):
        t=np.arange(0,60,1/130);x=sum(900*np.exp(-((t-v)/.018)**2)-300*np.exp(-((t-v-.04)/.018)**2) for v in np.arange(.5,60,.8));ms=t*1000
        for method in [trace_old,trace_new]:
            result=method(x,ms,130);negative=method(-x,ms,130)
            self.assertGreater(result['event_count'],50)
            self.assertEqual(result['event_count'],len(result['votes']))
            self.assertEqual(result['positive_votes'],negative['negative_votes'])
            self.assertEqual([v['index'] for v in result['votes']],[v['index'] for v in negative['votes']])


class ComparisonApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.old_db=root/'old.sqlite3';self.pair_db=root/'paired.sqlite3'
        initialize_database(self.old_db,'old-packet');initialize_database(self.pair_db,'pair-packet')
        save_rating(self.old_db,'original-minute',dict(status='complete',peak_indices=[1],notes='preserved'),0,'Previous reviewer')
        self.before=read_ratings(self.old_db)
        side=dict(direction='positive',score=1.,event_count=1,positive_votes=1,negative_votes=0,uncertain_beats=0,votes=[dict(index=1,sign=1)],decision='stable_positive')
        task=dict(id='pair-demo',participant='P2',t=[0,.5,1],x=[0,900,0],focusStartSec=0,focusEndSec=1,
                  sides={'A':dict(side,method_id='new'),'B':dict(side,method_id='old',direction='ambiguous')})
        original=dict(tasks=[],protocol=dict(strata=[]))
        paired=dict(tasks=[task],protocol=dict(strata=[dict(participant='P2',sampled_minutes=1,eligible_minutes=5,differing_blocks=5)]))
        self.handler=make_handler(original,self.old_db,'test-access-code',b'test-long-private-session-key',comparison_packet=paired,comparison_db=self.pair_db)
        _,headers,_=self.request('POST','/api/login',dict(reviewer='QA',accessCode='test-access-code'))
        self.cookie=headers['Set-Cookie'].split(';')[0]

    def tearDown(self):self.temp.cleanup()

    def request(self,method,path,body=None,extra=None):
        payload=json.dumps(body).encode() if body is not None else b''
        headers={'Host':'127.0.0.1:8766','Content-Length':str(len(payload))}
        if hasattr(self,'cookie'):headers['Cookie']=self.cookie
        if body is not None:headers['Content-Type']='application/json'
        headers.update(extra or {})
        raw=f'{method} {path} HTTP/1.1\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n'
        socket=FakeSocket(raw.encode()+payload)
        self.handler(socket,('127.0.0.1',1234),SimpleNamespace(server_name='test',server_port=8766))
        head,data=bytes(socket.output).split(b'\r\n\r\n',1);first,header=head.split(b'\r\n',1)
        return int(first.split()[1]),BytesParser().parsebytes(header+b'\r\n\r\n'),data

    def answer(self,**changes):return dict(choice='A',status='complete',confidence='high',notes='',version=0,**changes)

    def test_public_api_has_both_votes_but_no_method_identity_or_auto_rating(self):
        status,_,data=self.request('GET','/api/comparison/task/pair-demo')
        self.assertEqual(status,200);task=json.loads(data)
        self.assertEqual(set(task['sides']),{'A','B'});self.assertNotIn('method_id',data.decode())
        self.assertIsNone(task['rating']);self.assertEqual(read_ratings(self.pair_db),{})
        self.assertEqual(task['sides']['A']['votes'],[dict(index=1,sign=1)])

    def test_single_choice_roundtrip_history_conflict_and_original_preservation(self):
        status,_,data=self.request('POST','/api/comparison/task/pair-demo/review',self.answer())
        self.assertEqual(status,200);self.assertEqual(json.loads(data)['choice'],'A')
        self.assertEqual(self.request('POST','/api/comparison/task/pair-demo/review',self.answer())[0],409)
        status,_,_=self.request('POST','/api/comparison/task/pair-demo/review',dict(choice='neither',status='complete',confidence='low',notes='Both wrong',version=1))
        self.assertEqual(status,200);self.assertEqual(read_ratings(self.pair_db)['pair-demo']['choice'],'neither')
        with closing(sqlite3.connect(self.pair_db)) as db:self.assertEqual(db.execute('select count(*) from revisions').fetchone()[0],1)
        self.assertEqual(read_ratings(self.old_db),self.before)

    def test_export_no_identity_and_html_safe(self):
        self.request('POST','/api/comparison/task/pair-demo/review',dict(choice='B',status='complete',confidence='medium',notes='=1+1',version=0))
        status,headers,data=self.request('GET','/api/comparison/export.csv')
        self.assertEqual(status,200);row=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))[0]
        self.assertEqual(row['choice'],'B');self.assertTrue(row['notes'].startswith("'"));self.assertNotIn('method_id',row)

    def test_invalid_inputs_auth_and_legacy_route(self):
        for choice in ['old','new','AB',[],True]:
            body=dict(choice=choice,status='complete',confidence='high',notes='',version=0)
            self.assertEqual(self.request('POST','/api/comparison/task/pair-demo/review',body)[0],400)
        self.assertEqual(self.request('POST','/api/comparison/task/pair-demo/review',dict(choice='',status='complete',confidence='high',notes='',version=0))[0],400)
        self.assertEqual(self.request('GET','/api/comparison/tasks',extra={'Cookie':''})[0],401)
        self.assertEqual(self.request('POST','/api/comparison/task/pair-demo/review',self.answer(),extra={'Origin':'https://bad.example'})[0],403)
        self.assertIn(b'Which result fits',self.request('GET','/')[2])
        self.assertIn(b'Can QRS timing',self.request('GET','/legacy/')[2])
        self.assertEqual(read_ratings(self.old_db),self.before)
        with self.assertRaises(ValueError):validate_comparison(dict(choice='A',status=[],version=0))


class ComparisonPacketTests(unittest.TestCase):
    def test_generated_packet_sampling_waveforms_and_reserved_contexts(self):
        folder=HERE.parent/'outputs/polarity_pairwise_disagreements_v1'
        if not (folder/'comparison_packet.json.gz').exists():self.skipTest('Packet not generated')
        with gzip.open(folder/'comparison_packet.json.gz','rt') as f:packet=json.load(f)
        with (folder/'sampling_manifest_private.csv').open() as f:sample=list(csv.DictReader(f))
        reserved={}
        with (HERE.parent/'outputs/ecg_review_supplement_blind60_v1/review_manifest.csv').open() as f:
            for r in csv.DictReader(f):reserved.setdefault(r['participant'],[]).append((float(r['window_start_unix_ms']),float(r['window_start_unix_ms'])+300000))
        self.assertEqual(len(packet['tasks']),sum(s['sampled_minutes'] for s in packet['protocol']['strata']))
        self.assertEqual(len({t['id'] for t in packet['tasks']}),len(sample))
        self.assertFalse(any(overlaps_reserved(r,reserved) for r in sample))
        self.assertTrue(all(r['old_raw_polarity']!=r['raw_polarity'] for r in sample))
        for stratum in packet['protocol']['strata']:
            self.assertEqual(stratum['sampled_minutes'],min(8,stratum['eligible_minutes']))
        self.assertEqual(hashlib.sha256((HERE/'beatwise.py').read_bytes()).hexdigest(),packet['protocol']['beatwise_source_sha256'])
        for task in packet['tasks']:
            self.assertTrue(np.isfinite(task['x']).all());self.assertTrue(np.all(np.diff(task['t'])>0))
            self.assertGreaterEqual(task['focusEndSec'],59.5)
            self.assertNotEqual(task['sides']['A']['direction'],task['sides']['B']['direction'])
            for side in task['sides'].values():
                self.assertEqual(len(side['votes']),side['event_count'])
                self.assertTrue(all(0<=v['index']<len(task['t']) and v['sign'] in [-1,0,1] for v in side['votes']))


if __name__=='__main__':unittest.main()
