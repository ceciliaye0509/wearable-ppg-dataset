import csv,gzip,io,json
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from flip_acceptance import validate,compare,MODE
from build_flip_acceptance import DEFAULT_OUT
from serve_direction_truth import make_handler
from serve_audit import initialize_database,read_ratings
import test_direction_truth as truth

class AcceptanceLogicTests(unittest.TestCase):
 def test_all_three_minutes_required_and_no_prefilled_answer(self):
  body=dict(minute_directions=['','',''],status='complete',version=0)
  with self.assertRaises(ValueError):validate(body)
  body['status']='in_progress';self.assertEqual(validate(body)[0]['minute_directions'],['','',''])
 def test_wrong_or_unreadable_new_flip_and_unclear_context_prevent_promotion(self):
  packet=dict(tasks=[dict(id='x')]);pred=dict(x=dict(v3_flip=[False]*3,combined_flip=[True]*3))
  for labels in [['negative']*3,['negative','positive','negative'],['negative','unreadable','negative']]:
   r=compare(packet,pred,dict(x=dict(minute_directions=labels,status='complete',review_mode=MODE)))
   self.assertEqual(r['acceptance_pass'],labels==['negative']*3)
  with self.assertRaises(ValueError):compare(packet,pred,{})
  pred['x']['v3_flip']=[True]*3
  self.assertFalse(compare(packet,pred,dict(x=dict(minute_directions=['negative']*3,status='complete',review_mode=MODE)))['acceptance_pass'])
 def test_agreement_control_failure_also_prevents_promotion(self):
  packet=dict(tasks=[dict(id='new'),dict(id='control')])
  pred=dict(new=dict(v3_flip=[False]*3,combined_flip=[True]*3),control=dict(v3_flip=[True]*3,combined_flip=[True]*3))
  ratings=dict(new=dict(minute_directions=['negative']*3,status='complete',review_mode=MODE),control=dict(minute_directions=['positive']*3,status='complete',review_mode=MODE))
  self.assertFalse(compare(packet,pred,ratings)['acceptance_pass'])
 def test_selected_windows_unseen_nonoverlapping_and_frozen(self):
  p=json.load(gzip.open(DEFAULT_OUT/'review_packet.json.gz','rt'));pr=json.loads((DEFAULT_OUT/'private_predictions.json').read_text())
  seen=set(p['protocol']['excluded_seen_ids']);ids=[v for r in pr.values() for v in r['minute_ids']]
  self.assertFalse(set(ids)&seen);self.assertEqual(len(ids),len(set(ids)))
  self.assertEqual(len(p['tasks']),11);self.assertTrue(all('rating' not in t and t['focusEndSec']==180 for t in p['tasks']))

class AcceptanceApiTests(truth.DirectionTruthApiTests):
 def test_separate_store_privacy_completion_and_csv(self):
  db=Path(self.temp.name)/'accept.sqlite3';initialize_database(db,'accept-packet')
  task=dict(self.packet['tasks'][0],id='accept_x',queue='flip_acceptance',focusEndSec=180)
  packet=dict(tasks=[task]);pred=dict(accept_x=dict(v3_flip=[False]*3,combined_flip=[True]*3))
  self.handler=make_handler(self.packet,self.predictions,self.db,'direction-test-code',b'new-direction-test-secret',acceptance_packet=packet,acceptance_predictions=pred,acceptance_db_path=db)
  self.assertEqual(self.request('GET','/accept-api/results')[0],409)
  status,_,raw=self.request('GET','/accept-api/task/accept_x');self.assertEqual(status,200)
  self.assertNotIn('combined_flip',raw.decode());self.assertIsNone(json.loads(raw)['rating'])
  self.assertEqual(self.request('GET','/accept-api/private_predictions.json')[0],404)
  self.assertEqual(self.request('GET','/accept-api/tasks',extra={'Cookie':''})[0],401)
  body=dict(minute_directions=['negative']*3,status='complete',version=0,confidence='high',notes='=1+1')
  self.assertEqual(self.request('POST','/accept-api/task/accept_x/review',body,{'Origin':'https://other.test'})[0],403)
  self.assertEqual(self.request('POST','/accept-api/task/accept_x/review',body)[0],200)
  self.assertEqual(self.request('POST','/accept-api/task/accept_x/review',body)[0],409)
  _,_,raw=self.request('GET','/accept-api/results');self.assertTrue(json.loads(raw)['acceptance_pass'])
  self.assertEqual(read_ratings(self.db),{})
  _,headers,raw=self.request('GET','/accept-api/export.csv');self.assertIn('flip_acceptance_labels.csv',headers['Content-Disposition'])
  row=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))[0];self.assertEqual(row['minute3_direction'],'negative');self.assertTrue(row['notes'].startswith("'"))

if __name__=='__main__':unittest.main()
