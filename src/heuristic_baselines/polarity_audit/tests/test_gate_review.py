from contextlib import closing
import csv
import gzip
import io
import json
from pathlib import Path
import sqlite3
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from direction_gate_review import validate_interval_answer,aggregate_quarters,DEFAULT_OUT,REVIEW_MODE
from serve_direction_truth import make_handler,load_sources
from serve_audit import initialize_database,read_ratings,save_rating
import test_direction_truth as truth_tests
from build_direction_random20 import DEFAULT_OUT as MAIN_OUT


class IntervalValidationTests(unittest.TestCase):
    def answer(self,**extra):
        return dict(tile_readability=['readable']*12,direction='negative',status='complete',version=0,**extra)

    def test_no_prefilled_readability_or_implicit_completion(self):
        body=self.answer();body['tile_readability']=['']*12
        with self.assertRaises(ValueError):validate_interval_answer(body)
        body['status']='in_progress';body['direction']=''
        answer,_=validate_interval_answer(body)
        self.assertEqual(answer['tile_readability'],['']*12)
        self.assertEqual(answer['readable_intervals_sec'],[])

    def test_readable_intervals_are_derived_without_changing_peak_or_minute_labels(self):
        body=self.answer();body.update(tile_readability=['readable']*2+['unreadable']*10,
            peak_indices=[1,2],source_direction='positive',readable_intervals_sec=[[0,60]])
        answer,_=validate_interval_answer(body)
        self.assertEqual(answer['readable_intervals_sec'],[[0,5],[5,10]])
        self.assertNotIn('peak_indices',answer);self.assertNotIn('source_direction',answer)
        self.assertEqual(answer['review_mode'],REVIEW_MODE)

    def test_signed_label_requires_readable_interval_and_complete_coverage(self):
        for tiles,direction in [(['unreadable']*12,'positive'),(['readable']*12,'unreadable'),(['readable']*11,'positive')]:
            body=self.answer();body.update(tile_readability=tiles,direction=direction)
            with self.assertRaises(ValueError):validate_interval_answer(body)
        body=self.answer();body.update(tile_readability=['unreadable']*12,direction='unreadable')
        self.assertEqual(validate_interval_answer(body)[0]['direction'],'unreadable')

    def test_local_candidate_needs_45_seconds_and_no_opposite_quarter(self):
        def result(signs):return aggregate_quarters([dict(direction=s) for s in signs])
        self.assertEqual(result(['positive']*3+['severe_artifact']),'positive')
        self.assertEqual(result(['positive']*2+['severe_artifact']*2),'ambiguous')
        self.assertEqual(result(['positive']*3+['negative']),'ambiguous')


class GateApiTests(truth_tests.DirectionTruthApiTests):
    # Reuse the existing signed-in fake socket harness and the main-app checks.
    def setUp(self):
        super().setUp()
        self.gate_db=Path(self.temp.name)/'intervals.sqlite3';initialize_database(self.gate_db,'interval-reference')
        task=dict(self.packet['tasks'][0],id='gate_pos',source_id='pos',queue='gate_review',minute_span_uv=6000,
            tiles=[dict(index=i,start_sec=5*i,end_sec=5*(i+1),span_uv=1000,over_threshold=False) for i in range(12)])
        self.gate_packet=dict(tasks=[task],protocol={})
        self.handler=make_handler(self.packet,self.predictions,self.db,'direction-test-code',b'new-direction-test-secret',
                                 gate_packet=self.gate_packet,gate_db_path=self.gate_db)

    def interval_answer(self):
        return dict(tile_readability=['readable']*12,direction='negative',status='complete',version=0,confidence='medium',notes='')

    def test_interval_save_export_conflict_and_history_leave_main_store_identical(self):
        self.request('POST','/api/task/pos/review',self.answer())
        main_before=read_ratings(self.db)
        answer=self.interval_answer()
        self.assertEqual(self.request('POST','/gate-api/task/gate_pos/review',answer)[0],200)
        self.assertEqual(self.request('POST','/gate-api/task/gate_pos/review',answer)[0],409)
        answer.update(version=1,direction='positive')
        self.assertEqual(self.request('POST','/gate-api/task/gate_pos/review',answer)[0],200)
        self.assertEqual(read_ratings(self.db),main_before)
        self.assertEqual(read_ratings(self.gate_db)['gate_pos']['direction'],'positive')
        with closing(sqlite3.connect(self.gate_db)) as db:self.assertEqual(db.execute('select count(*) from revisions').fetchone()[0],1)
        _,headers,raw=self.request('GET','/gate-api/export.csv')
        self.assertIn('direction_gate_interval_labels.csv',headers['Content-Disposition'])
        row=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))[0]
        self.assertEqual(row['source_id'],'pos');self.assertEqual(json.loads(row['tile_readability']),['readable']*12)

    def test_gate_routes_are_private_and_do_not_send_algorithm_or_prior_answers(self):
        self.assertEqual(self.request('GET','/gate-api/tasks',extra={'Cookie':''})[0],401)
        self.assertEqual(self.request('GET','/gate-api/private_diagnostics.json')[0],404)
        status,_,raw=self.request('GET','/gate-api/task/gate_pos')
        task=json.loads(raw);self.assertEqual(status,200);self.assertIsNone(task['rating'])
        self.assertNotIn('reference',task);self.assertNotIn('gate_bypass_diagnostic',task)
        self.assertNotIn('local_quarters_candidate',task);self.assertNotIn('direction',task)
        self.assertEqual(self.request('POST','/gate-api/task/gate_pos/review',self.interval_answer(),{'Origin':'https://elsewhere.test'})[0],403)
        self.assertEqual(read_ratings(self.gate_db),{})


class BuiltGateReviewTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_OUT/'review_packet.json.gz').exists(),'Gate packet not built')
    def test_nine_target_waveforms_match_original_and_controls_cover_all_160(self):
        main,_=load_sources(MAIN_OUT/'reference_packet.json.gz',MAIN_OUT/'private_predictions.json')
        original={t['id']:t for t in main['tasks']}
        packet=json.load(gzip.open(DEFAULT_OUT/'review_packet.json.gz','rt'))
        self.assertEqual(len(packet['tasks']),9)
        for task in packet['tasks']:
            src=original[task['source_id']]
            self.assertEqual(task['t'],src['t']);self.assertEqual(task['x'],src['x'])
            self.assertEqual(len(task['tiles']),12);self.assertNotIn('rating',task)
        diagnostics=json.loads((DEFAULT_OUT/'private_diagnostics.json').read_text())
        self.assertEqual(len(diagnostics['results']),160)
        self.assertTrue(all(len(r['quarters'])==4 and len(r['amplitude_tiles'])==12 for r in diagnostics['results']))
        self.assertEqual(diagnostics['metrics']['original']['correct'],105)
        self.assertEqual(diagnostics['metrics']['original']['signed_outputs_on_unreadable'],0)


if __name__=='__main__':unittest.main()
