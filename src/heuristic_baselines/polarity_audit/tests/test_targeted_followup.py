import copy
import csv
from email.parser import BytesParser
import gzip
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from build_targeted_followup import select_followup,summarise_evidence,DEFAULT_OUT
from serve_targeted_followup import COOKIE,make_handler,public_task
from serve_audit import initialize_database,read_ratings,save_rating
from test_audit import FakeSocket


class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(id=f'P{i}_R000_B00001',participant=f'P{i}',technical_pass='True',old_v1='negative',v3_minute='negative',rod300='negative') for i in range(1,7)]

    def test_existing_direction_and_decisive_preference_are_not_repeated(self):
        labeled={'P1_R000_B00001':dict(status='complete',direction_determinability='yes',direction='negative')}
        prefs={'pair_P2_R000_B00001':dict(status='complete',choice='A'),
               'pair_P3_R000_B00001':dict(status='complete',choice='unsure')}
        changes={r['id'] for r in self.rows[:3]}
        priority,optional,skipped=select_followup(self.rows,{r['id'] for r in self.rows},changes,labeled,prefs)
        self.assertEqual([r['id'] for r in priority],['P3_R000_B00001'])
        self.assertEqual({r['evidence'] for r in skipped},{'explicit_direction','paired_preference_only'})
        self.assertEqual(len(optional),3)
        self.assertEqual(len(summarise_evidence(self.rows,labeled)),3)
        self.assertTrue(all(m['reference_minutes']==1 for m in summarise_evidence(self.rows,labeled)))

    def test_opposite_predictions_are_prioritized_and_sampling_is_reproducible(self):
        self.rows[0]['rod300']='positive'
        a=select_followup(self.rows,{r['id'] for r in self.rows},set(),{}, {})
        b=select_followup(list(reversed(self.rows)),{r['id'] for r in self.rows},set(),{}, {})
        self.assertEqual(a,b);self.assertEqual(len(a[0]),1)
        self.assertEqual(a[0][0]['selection_reason'],'opposite_signed_predictions')


class TargetApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.db=root/'new.sqlite3';self.old=root/'old.sqlite3'
        initialize_database(self.db,'new-packet');initialize_database(self.old,'old-packet')
        save_rating(self.old,'prior',dict(status='complete',direction='negative',peak_indices=[1,3]),0,'Qiyu')
        self.before=read_ratings(self.old)
        side=dict(direction='negative',support='supported',decision='stable_negative',flip=True)
        self.task=dict(id='followup_demo',participant='P4',queue='priority',t=[0,.5,1],x=[0,-100,0],focusStartSec=0,focusEndSec=1,
            sides={key:dict(side,method_id=name) for key,name in zip('ABC',['old_v1','v3_minute','rod300'])},private_reason='secret')
        self.packet=dict(tasks=[self.task],protocol=dict(participants=[],evidence={},priority_minutes=1,optional_minutes=0))
        self.handler=make_handler(self.packet,self.db,'test-followup-code',b'new-secret-for-followup')
        status,headers,_=self.request('POST','/api/login',dict(reviewer='QA',accessCode='test-followup-code'))
        self.assertEqual(status,200);self.cookie=headers['Set-Cookie'].split(';')[0]

    def tearDown(self):self.temp.cleanup()

    def request(self,method,path,body=None,extra=None):
        raw=json.dumps(body).encode() if body is not None else b''
        headers={'Host':'127.0.0.1:8767','Content-Length':str(len(raw))}
        if hasattr(self,'cookie'):headers['Cookie']=self.cookie
        if body is not None:headers['Content-Type']='application/json'
        headers.update(extra or {})
        head=f'{method} {path} HTTP/1.1\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n'
        sock=FakeSocket(head.encode()+raw);self.handler(sock,('127.0.0.1',1234),SimpleNamespace(server_name='test',server_port=8767))
        head,payload=bytes(sock.output).split(b'\r\n\r\n',1);first,rest=head.split(b'\r\n',1)
        return int(first.split()[1]),BytesParser().parsebytes(rest+b'\r\n\r\n'),payload

    def answer(self,**changes):return dict(choice='BC',confidence='high',status='complete',notes='',version=0,**changes)

    def test_loading_is_read_only_and_identity_is_hidden(self):
        _,_,raw=self.request('GET','/api/task/followup_demo');v=json.loads(raw)
        self.assertIsNone(v['rating']);self.assertNotIn('private_reason',v)
        self.assertNotIn('method_id',raw.decode());self.assertNotIn('rod300',raw.decode())
        self.assertEqual(v['x'],[0,-100,0]);self.assertEqual(read_ratings(self.db),{})
        self.assertEqual(read_ratings(self.old),self.before)

    def test_save_tie_revision_conflict_and_old_peaks_preserved(self):
        status,_,raw=self.request('POST','/api/task/followup_demo/review',self.answer())
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['choice'],'BC')
        self.assertEqual(self.request('POST','/api/task/followup_demo/review',self.answer())[0],409)
        second=self.answer();second.update(choice='same',version=1)
        self.assertEqual(self.request('POST','/api/task/followup_demo/review',second)[0],200)
        with sqlite3.connect(self.db) as con:self.assertEqual(con.execute('select count(*) from revisions').fetchone()[0],1)
        self.assertEqual(read_ratings(self.old),self.before)

    def test_auth_and_origin_and_incomplete_choice(self):
        self.assertTrue(self.cookie.startswith(COOKIE+'='));self.assertNotIn('ecg_polarity_audit_session',self.cookie)
        self.assertEqual(self.request('GET','/api/tasks',extra={'Cookie':''})[0],401)
        self.assertEqual(self.request('POST','/api/task/followup_demo/review',self.answer(),{'Origin':'https://elsewhere.test'})[0],403)
        bad=self.answer();bad['choice']=''
        self.assertEqual(self.request('POST','/api/task/followup_demo/review',bad)[0],400)
        self.assertEqual(read_ratings(self.db),{})

    def test_export_keeps_abc_and_escapes_spreadsheet_formulas(self):
        body=self.answer();body['notes']='=1+1'
        self.request('POST','/api/task/followup_demo/review',body)
        _,_,raw=self.request('GET','/api/export.csv')
        rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        self.assertEqual(rows[0]['choice'],'BC');self.assertTrue(rows[0]['notes'].startswith("'"))
        self.assertNotIn('method_id',raw.decode());self.assertNotIn('rodrigues',raw.decode())


class BuiltFollowupChecks(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_OUT/'followup_packet.json.gz').exists(),'Packet not built yet')
    def test_packet_contains_only_new_work_and_retains_prior_records_as_evidence(self):
        with gzip.open(DEFAULT_OUT/'followup_packet.json.gz','rt') as f:packet=json.load(f)
        tasks=packet['tasks'];self.assertEqual(sum(t['queue']=='priority' for t in tasks),11)
        self.assertEqual(sum(t['queue']=='optional' for t in tasks),3)
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))
        self.assertTrue(all(set(t['sides'])=={'A','B','C'} and 'rating' not in t for t in tasks))
        self.assertFalse(any(t['id'].endswith('P4_R006_B00265') for t in tasks))
        evidence=json.loads((DEFAULT_OUT/'reused_evidence.json').read_text())
        prior=set(evidence['direction_records']);prefs={k.removeprefix('pair_') for k,v in evidence['paired_preferences'].items() if v.get('status')=='complete' and v.get('choice') in ['A','B','same']}
        self.assertFalse({t['id'].removeprefix('followup_') for t in tasks}&(prior|prefs))
        self.assertEqual(packet['protocol']['evidence']['explicit_direction_records'],36)
        self.assertEqual([m['same_direction'] for m in packet['protocol']['evidence']['metrics']],[10,27,28])


if __name__=='__main__':unittest.main()
