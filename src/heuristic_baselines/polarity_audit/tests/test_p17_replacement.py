from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from build_p17_replacement import DEFAULT_OUT,PREVIOUS,BAD_IDS,choose,overlaps
from serve_direction_truth import load_sources,summary,compare_labels,REVIEW_MODE,REPAIR_VERSION
import test_direction_truth as truth_tests


class RepairScoringTests(unittest.TestCase):
    def fixture(self):
        packet,predictions=truth_tests.fixture()
        for task in packet['tasks']:
            task['in_corrected_main']=task['id']=='pos'
            task['superseded_in_corrected_main']=task['id']=='neg'
            if task['queue']=='priority':task['queue']='main'
        new=dict(packet['tasks'][0],id='replacement',participant='P17',queue='replacement',
                 in_corrected_main=True,superseded_in_corrected_main=False,replaces_id='neg')
        packet['tasks'].append(new)
        packet['protocol'].update(version=REPAIR_VERSION,main_minutes=2,corrected_main_minutes=2,
            replacement_minutes=1,difficult_minutes=0,optional_minutes=3,corrected_scope='Test repaired cohort')
        predictions['replacement']={m:dict(direction='positive',flip=None) for m in ['old_v1','v3_minute','rod300']}
        ratings={key:dict(direction='positive',status='complete',review_mode=REVIEW_MODE) for key in ['pos','neg']}
        return packet,predictions,ratings

    def test_corrected_score_excludes_superseded_label_and_waits_for_replacement(self):
        packet,predictions,ratings=self.fixture()
        groups={r['id']:r for r in summary(packet,ratings)['cohorts']}
        self.assertEqual(groups['corrected']['completed'],1);self.assertFalse(groups['corrected']['results_ready'])
        self.assertEqual(groups['main']['completed'],2);self.assertTrue(groups['main']['results_ready'])
        self.assertEqual(set(groups['corrected']['task_ids']),{'pos','replacement'})
        ratings['replacement']=dict(direction='positive',status='complete',review_mode=REVIEW_MODE)
        groups={r['id']:r for r in compare_labels(packet,predictions,ratings)['cohorts']}
        self.assertTrue(groups['corrected']['results_ready'])
        self.assertEqual(groups['corrected']['signed_reference_minutes'],2)
        self.assertEqual(groups['corrected']['methods'][0]['correct_and_decided'],2)
        self.assertEqual(groups['main']['completed_new_labels'],2)

    def test_sampling_is_repeatable_and_does_not_read_prediction_fields(self):
        rows=[dict(run_index=1,block_index=i) for i in range(30)]
        ids=[r['block_index'] for r in choose(rows)]
        for r in rows:r.update(direction='arbitrary',human_label='not consulted')
        self.assertEqual([r['block_index'] for r in choose(list(reversed(rows)))],ids)
        self.assertEqual(len(set(ids)),4)
        self.assertFalse(overlaps((0,60),(60,120)));self.assertTrue(overlaps((0,60),(59,120)))


class RepairApiTests(truth_tests.DirectionTruthApiTests):
    def test_corrected_results_hidden_until_new_labels_saved_and_export_membership_explicit(self):
        from serve_direction_truth import make_handler
        packet,predictions,_=RepairScoringTests().fixture()
        self.handler=make_handler(packet,predictions,self.db,'direction-test-code',b'new-direction-test-secret')
        for task_id in ['pos','neg']:self.request('POST',f'/api/task/{task_id}/review',self.answer())
        status,_,raw=self.request('GET','/api/results');self.assertEqual(status,200)
        self.assertNotIn('corrected',[r['id'] for r in json.loads(raw)['cohorts']])
        _,_,raw=self.request('GET','/api/task/replacement');task=json.loads(raw)
        self.assertIsNone(task['rating']);self.assertNotIn('predictions',task)
        answer=self.answer();answer.update(in_corrected_main=False,superseded_in_corrected_main=True)
        self.assertEqual(self.request('POST','/api/task/replacement/review',answer)[0],200)
        _,_,raw=self.request('GET','/api/results')
        self.assertIn('corrected',[r['id'] for r in json.loads(raw)['cohorts']])
        _,_,raw=self.request('GET','/api/export.csv');rows={r['id']:r for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}
        self.assertEqual(rows['replacement']['in_corrected_main'],'True')
        self.assertEqual(rows['replacement']['replaces_id'],'neg')
        self.assertEqual(rows['neg']['superseded_in_corrected_main'],'True')
        self.assertEqual(rows['neg']['in_corrected_main'],'False')


class BuiltReplacementTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_OUT/'reference_packet.json.gz').exists(),'Replacement set not built')
    def test_replacements_are_distinct_outside_P19_and_all_old_waveforms_retained(self):
        packet,predictions=load_sources(DEFAULT_OUT/'reference_packet.json.gz',DEFAULT_OUT/'private_predictions.json')
        old=json.load(gzip.open(PREVIOUS/'reference_packet.json.gz','rt'))
        old_by_id={t['id']:t for t in old['tasks']};new_by_id={t['id']:t for t in packet['tasks']}
        self.assertEqual(len(packet['tasks']),179)
        for key,t in old_by_id.items():
            for field in ['participant','queue','t','x','focusStartSec','focusEndSec']:
                self.assertEqual(t[field],new_by_id[key][field])
        corrected=[t for t in packet['tasks'] if t['in_corrected_main']]
        self.assertEqual(Counter(t['participant'] for t in corrected),{f'P{i}':8 for i in range(1,21)})
        self.assertFalse(set(BAD_IDS)&{t['id'] for t in corrected})
        selection=json.loads((DEFAULT_OUT/'replacement_selection.json').read_text())
        self.assertFalse(selection['algorithm_outputs_used_for_selection']);self.assertFalse(selection['human_answers_used_for_selection'])
        source=Path(__file__).resolve().parents[2]/'outputs/ecg_polarity_correction_all20_v1/polarity_blocks.csv'
        with source.open() as stream:
            rows={f"{r['participant']}_R{int(r['run_index']):03d}_B{int(r['block_index']):05d}":r for r in csv.DictReader(stream)}
        p19=[rows[t['id'].removeprefix('truth_')] for t in old['tasks'] if t['participant']=='P19']
        for r in selection['selected']:
            interval=(float(r['end_ms'])-60000.,float(r['end_ms']))
            self.assertFalse(overlaps(interval,selection['excluded_entire_source_interval_ms']))
            self.assertTrue(all(not overlaps(interval,(float(z['end_ms'])-60000.,float(z['end_ms']))) for z in p19))
            self.assertNotIn('truth_'+r['id'],old_by_id)
        fresh=[t for t in packet['tasks'] if t['queue']=='replacement']
        self.assertEqual(len(fresh),4);self.assertTrue(all('rating' not in t for t in fresh))
        self.assertTrue(all(set(predictions[t['id']])=={'old_v1','v3_minute','rod300'} for t in fresh))


if __name__=='__main__':unittest.main()
