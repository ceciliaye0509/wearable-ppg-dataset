from contextlib import closing
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
from serve_direction_truth import COOKIE,REVIEW_MODE,compare_labels,make_handler,load_sources,summary
from build_direction_truth import DEFAULT_OUT
from build_direction_random20 import DEFAULT_OUT as RANDOM_OUT,select_random,migrate_labels,sha,eligibility_reasons
from serve_audit import initialize_database,read_ratings,save_rating
from test_audit import FakeSocket


def fixture():
    tasks=[dict(id=key,participant='P4',queue=queue,t=[0,.5,1],x=[0,100,0],focusStartSec=0,focusEndSec=1) for key,queue in [('pos','priority'),('neg','priority'),('balanced','optional'),('unreadable','optional'),('unsure','optional')]]
    predictions={key:{m:dict(direction=direction,flip=direction=='negative') for m,direction in zip(['old_v1','v3_minute','rod300'],values)}
                 for key,values in [('pos',['positive','ambiguous','negative']),('neg',['ambiguous','negative','negative']),('balanced',['positive','ambiguous','negative']),('unreadable',['positive','negative','ambiguous']),('unsure',['positive','negative','ambiguous'])]}
    return dict(version=1,tasks=tasks,protocol=dict(participants=[])),predictions


class DirectionScoringTests(unittest.TestCase):
    def test_random_and_difficult_labels_never_form_a_pooled_score(self):
        packet,predictions=fixture()
        packet['protocol']['version']='raw_direction_truth_v2'
        for task in packet['tasks']:task['queue']='main' if task['id']=='pos' else 'difficult'
        ratings={key:dict(direction=label,status='complete',review_mode=REVIEW_MODE) for key,label in [('pos','positive'),('neg','negative')]}
        result=compare_labels(packet,predictions,ratings)
        self.assertNotIn('methods',result);self.assertTrue(result['pooled_score_prohibited'])
        main,hard,optional=result['cohorts']
        self.assertEqual(main['signed_reference_minutes'],1);self.assertEqual(hard['signed_reference_minutes'],1)
        self.assertEqual(main['methods'][0]['correct_and_decided'],1)
        self.assertEqual(hard['methods'][0]['abstained'],1)
        self.assertEqual(optional['completed_new_labels'],0)

    def test_correct_wrong_abstention_and_unknowns_are_separate(self):
        packet,predictions=fixture()
        ratings={key:dict(direction=key if key in ['positive','negative'] else label,status='complete',review_mode=REVIEW_MODE,confidence='low') for key,label in [('pos','positive'),('neg','negative'),('balanced','balanced'),('unreadable','unreadable'),('unsure','unsure')]}
        result=compare_labels(packet,predictions,ratings)
        self.assertEqual(result['completed_new_labels'],5);self.assertEqual(result['signed_reference_minutes'],2)
        old,v3,rod=result['methods']
        self.assertEqual((old['correct_and_decided'],old['opposite_direction'],old['abstained']),(1,0,1))
        self.assertEqual((v3['correct_and_decided'],v3['opposite_direction'],v3['abstained']),(1,0,1))
        self.assertEqual((rod['correct_and_decided'],rod['opposite_direction'],rod['abstained']),(1,1,0))
        self.assertEqual(old['direction_accuracy_when_decided'],1.)
        self.assertEqual(rod['coverage'],1.);self.assertEqual(rod['direction_accuracy_when_decided'],.5)
        self.assertEqual(rod['proposed_reversals_on_positive'],1)
        self.assertEqual(rod['signed_output_on_balanced_or_mixed'],1)

    def test_legacy_preferences_and_old_direction_records_never_count(self):
        packet,predictions=fixture()
        result=compare_labels(packet,predictions,{'pos':dict(direction='positive',status='complete'),
            'neg':dict(choice='B',status='complete',review_mode='assisted_targeted_three_method_preference'),
            'outside':dict(direction='negative',status='complete',review_mode=REVIEW_MODE)})
        self.assertEqual(result['completed_new_labels'],0);self.assertFalse(result['old_records_used'])
        self.assertTrue(all(m['coverage'] is None and m['direction_accuracy_when_decided'] is None for m in result['methods']))


class DirectionTruthApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name);self.db=root/'fresh.sqlite3';self.old=root/'old.sqlite3'
        initialize_database(self.db,'fresh-reference');initialize_database(self.old,'old-reference')
        save_rating(self.old,'old',dict(choice='A',direction='negative',peak_indices=[1,2],status='complete'),0,'Qiyu');self.before=read_ratings(self.old)
        self.packet,self.predictions=fixture();self.handler=make_handler(self.packet,self.predictions,self.db,'direction-test-code',b'new-direction-test-secret')
        status,headers,_=self.request('POST','/api/login',dict(reviewer='QA',accessCode='direction-test-code'))
        self.assertEqual(status,200);self.cookie=headers['Set-Cookie'].split(';')[0]

    def tearDown(self):self.temp.cleanup()

    def request(self,method,path,body=None,extra=None):
        payload=json.dumps(body).encode() if body is not None else b''
        headers={'Host':'127.0.0.1:8767','Content-Length':str(len(payload))}
        if hasattr(self,'cookie'):headers['Cookie']=self.cookie
        if body is not None:headers['Content-Type']='application/json'
        headers.update(extra or {});request=f'{method} {path} HTTP/1.1\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n'
        sock=FakeSocket(request.encode()+payload);self.handler(sock,('127.0.0.1',1234),SimpleNamespace(server_name='test',server_port=8767))
        head,raw=bytes(sock.output).split(b'\r\n\r\n',1);first,rest=head.split(b'\r\n',1)
        return int(first.split()[1]),BytesParser().parsebytes(rest+b'\r\n\r\n'),raw

    def answer(self,direction='positive',version=0):return dict(direction=direction,confidence='high',notes='',status='complete',version=version)

    def test_predictions_prior_answers_and_algorithm_names_are_not_sent(self):
        status,_,raw=self.request('GET','/api/task/pos');self.assertEqual(status,200);task=json.loads(raw)
        self.assertIsNone(task['rating']);self.assertEqual(set(task),{'id','participant','queue','t','x','focusStartSec','focusEndSec','rating'})
        self.assertNotIn('rod300',raw.decode());self.assertNotIn('flip',raw.decode());self.assertEqual(read_ratings(self.db),{})
        self.assertTrue(self.cookie.startswith(COOKIE+'='))

    def test_results_stay_hidden_until_priority_labels_complete(self):
        self.assertEqual(self.request('GET','/api/results')[0],409)
        self.request('POST','/api/task/pos/review',self.answer())
        self.assertEqual(self.request('GET','/api/results')[0],409)
        self.request('POST','/api/task/neg/review',self.answer('negative'))
        status,_,raw=self.request('GET','/api/results');self.assertEqual(status,200)
        result=json.loads(raw);self.assertEqual(result['signed_reference_minutes'],2);self.assertFalse(result['old_records_used'])
        self.assertEqual(read_ratings(self.old),self.before)

    def test_finished_cohort_does_not_reveal_an_unfinished_cohort(self):
        self.packet['protocol'].update(version='raw_direction_truth_v2',main_minutes=1,difficult_minutes=1,optional_minutes=3)
        self.packet['tasks'][0]['queue']='main';self.packet['tasks'][1]['queue']='difficult'
        self.assertEqual(self.request('GET','/api/results')[0],409)
        self.request('POST','/api/task/pos/review',self.answer())
        status,_,raw=self.request('GET','/api/results');self.assertEqual(status,200)
        result=json.loads(raw);self.assertEqual([r['id'] for r in result['cohorts']],['main'])
        _,_,raw=self.request('GET','/api/tasks');data=json.loads(raw)
        self.assertEqual([c['results_ready'] for c in data['cohorts']],[True,False,False])

    def test_fresh_label_save_revision_restore_and_stale_save_rejection(self):
        self.assertEqual(self.request('POST','/api/task/pos/review',self.answer())[0],200)
        self.assertEqual(self.request('POST','/api/task/pos/review',self.answer('negative'))[0],409)
        self.assertEqual(self.request('POST','/api/task/pos/review',self.answer('balanced',1))[0],200)
        _,_,raw=self.request('GET','/api/task/pos');self.assertEqual(json.loads(raw)['rating']['direction'],'balanced')
        with closing(sqlite3.connect(self.db)) as db:self.assertEqual(db.execute('select count(*) from revisions').fetchone()[0],1)
        self.assertEqual(read_ratings(self.old),self.before)

    def test_invalid_labels_auth_private_files_and_export(self):
        for label in ['','A',[],None]:self.assertEqual(self.request('POST','/api/task/pos/review',self.answer(label))[0],400)
        self.assertEqual(self.request('GET','/api/tasks',extra={'Cookie':''})[0],401)
        self.assertEqual(self.request('GET','/private_predictions.json')[0],404)
        self.assertEqual(self.request('POST','/api/task/pos/review',self.answer(),{'Origin':'https://elsewhere.test'})[0],403)
        answer=self.answer('negative');answer.update(choice='A',method_id='old_v1',notes='=1+1')
        self.request('POST','/api/task/pos/review',answer)
        _,headers,raw=self.request('GET','/api/export.csv');rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        self.assertEqual(rows[0]['direction'],'negative');self.assertNotIn('choice',rows[0]);self.assertEqual(rows[0]['review_mode'],REVIEW_MODE)
        self.assertTrue(rows[0]['notes'].startswith("'"));self.assertIn('direction_reference_labels.csv',headers['Content-Disposition'])


class BuiltDirectionTruthTests(unittest.TestCase):
    @unittest.skipUnless((RANDOM_OUT/'reference_packet.json.gz').exists(),'Random packet not built yet')
    def test_every_participant_has_eight_random_minutes_with_difficult_tasks_separate(self):
        packet,predictions=load_sources(RANDOM_OUT/'reference_packet.json.gz',RANDOM_OUT/'private_predictions.json')
        from collections import Counter
        main=[t for t in packet['tasks'] if t['queue']=='main']
        self.assertEqual(Counter(t['participant'] for t in main),{f'P{i}':8 for i in range(1,21)})
        self.assertTrue(all(t['focusEndSec']==60 and t['focusStartSec']==0 for t in main))
        self.assertEqual(sum(t['queue']=='difficult' for t in packet['tasks']),12)
        self.assertEqual(sum(t['queue']=='optional' for t in packet['tasks']),3)
        selection=json.loads((RANDOM_OUT/'frozen_selection.json').read_text())
        self.assertFalse(selection['algorithm_outputs_used']);self.assertFalse(selection['human_answers_used'])
        self.assertEqual({'truth_'+r['id'] for r in selection['selected']},{t['id'] for t in main})
        self.assertEqual(len({t['id'] for t in packet['tasks']}),175)
        self.assertEqual(summary(packet,{})['main_minutes'],160)
        self.assertTrue(all(predictions[t['id']][m]['flip'] is None for t in main for m in ['old_v1','v3_minute','rod300']))
        for p,digest in packet['protocol']['protected_hashes'].items():self.assertEqual(sha(Path(p)),digest)

    @unittest.skipUnless((DEFAULT_OUT/'reference_packet.json.gz').exists(),'Reference packet not built yet')
    def test_restored_minute_empty_answers_and_private_frozen_predictions(self):
        packet,predictions=load_sources(DEFAULT_OUT/'reference_packet.json.gz',DEFAULT_OUT/'private_predictions.json')
        self.assertEqual(len(packet['tasks']),15);self.assertEqual(packet['protocol']['priority_minutes'],12)
        self.assertIn('truth_P4_R006_B00265',{t['id'] for t in packet['tasks']})
        self.assertTrue(all('rating' not in t and 'sides' not in t and 'direction' not in t for t in packet['tasks']))
        self.assertFalse(packet['protocol']['old_records_used_for_scoring']);self.assertFalse(packet['protocol']['old_records_prefilled'])
        archives=json.loads((DEFAULT_OUT/'archive_manifest.json').read_text())
        self.assertEqual([a['ratings'] for a in archives['archived']],[36,31,11])
        for item in archives['archived']:
            self.assertTrue(Path(item['backup']).exists());self.assertFalse(item['used_in_new_scoring'])


class SamplingAndMigrationTests(unittest.TestCase):
    def test_normal_sample_grid_offset_is_not_excluded_as_a_partial_minute(self):
        import numpy as np
        t=np.arange(4.,60000.,8.);x=np.zeros(len(t))
        row=dict(start_ms=4.,end_ms=60000.,sample_count=len(t))
        self.assertEqual(eligibility_reasons(t,x,row),[])
        row['start_ms']=1000.
        self.assertEqual(eligibility_reasons(t,x,row),['partial_minute_under_59.5s'])

    def test_repeatable_sampling_does_not_depend_on_predictions_or_input_order(self):
        rows=[dict(participant='P1',run_index=0,block_index=i) for i in range(100)]
        original=[r['block_index'] for r in select_random(rows,'P1')]
        for r in rows:r.update(direction='negative' if r['block_index']%2 else 'positive',human_answer='unused')
        self.assertEqual([r['block_index'] for r in select_random(list(reversed(rows)),'P1')],original)
        self.assertEqual(len(set(original)),8)

    def test_migration_preserves_answers_versions_and_revision_history_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'old.json.gz';new=root/'new.json.gz'
            task=fixture()[0]['tasks'][0]
            with gzip.open(old,'wt') as f:json.dump(dict(tasks=[task]),f)
            with gzip.open(new,'wt') as f:json.dump(dict(tasks=[dict(task,queue='difficult')]),f)
            source=root/'source.sqlite3';dest=root/'dest.sqlite3'
            initialize_database(source,sha(old))
            first=dict(direction='negative',status='in_progress',review_mode=REVIEW_MODE,notes='keep this')
            save_rating(source,task['id'],first,0,'Qiyu')
            save_rating(source,task['id'],dict(first,direction='positive',status='complete'),1,'Qiyu')
            before=read_ratings(source)
            report=migrate_labels(old,new,source,dest)
            self.assertEqual(read_ratings(dest),before);self.assertEqual(read_ratings(source),before)
            self.assertEqual(report['migrated_labels'],1);self.assertEqual(report['migrated_revisions'],1)
            with closing(sqlite3.connect(source)) as s,closing(sqlite3.connect(dest)) as d:
                self.assertEqual(s.execute('SELECT * FROM revisions').fetchall(),d.execute('SELECT * FROM revisions').fetchall())
            with self.assertRaises(ValueError):migrate_labels(old,new,source,dest)


if __name__=='__main__':unittest.main()
