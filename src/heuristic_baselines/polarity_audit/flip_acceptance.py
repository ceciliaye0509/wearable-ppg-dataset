"""Independent three-minute flip acceptance labels, isolated from all prior reviews."""
import csv,json
from io import StringIO
from urllib.parse import urlsplit
from serve_targeted_followup import make_handler
from remote_ecg_review.serve_review import csv_safe

MODE='three_minute_flip_acceptance_v1'
LABELS={'','positive','negative','balanced','mixed','unreadable','unsure'}

def validate(body):
 labels=body.get('minute_directions');status=body.get('status');version=body.get('version')
 if not isinstance(labels,list) or len(labels)!=3 or any(not isinstance(v,str) or v not in LABELS for v in labels):raise ValueError('Choose a direction for each of the three minutes.')
 if status not in {'in_progress','complete'} or type(version) is not int or version<0:raise ValueError('Invalid status or saved version.')
 if status=='complete' and '' in labels:raise ValueError('Label all three minutes before completing.')
 notes=body.get('notes','');confidence=body.get('confidence','medium')
 if not isinstance(notes,str) or len(notes)>2000 or confidence not in {'high','medium','low'}:raise ValueError('Invalid notes or confidence.')
 return dict(minute_directions=labels,status=status,confidence=confidence,notes=notes,review_mode=MODE),version

def completed(r):return r.get('review_mode')==MODE and r.get('status')=='complete' and isinstance(r.get('minute_directions'),list) and len(r['minute_directions'])==3 and all(v in LABELS-{''} for v in r['minute_directions'])
def public(task,rating=None):return {**{k:task[k] for k in ['id','participant','queue','t','x','focusStartSec','focusEndSec']},'rating':rating}
def summary(packet,ratings):return dict(tasks=[dict(id=t['id'],participant=t['participant'],queue=t['queue'],rating=ratings.get(t['id'])) for t in packet['tasks']],total=len(packet['tasks']),completed=sum(completed(ratings.get(t['id'],{})) for t in packet['tasks']))

def compare(packet,predictions,ratings):
 if not packet['tasks'] or not all(completed(ratings.get(t['id'],{})) for t in packet['tasks']):raise ValueError('Complete every interval before viewing method comparisons.')
 result={};new_correct=0;new_unsafe=0;new_context_uncertain=0
 for method in ['v3_flip','combined_flip']:
  count=correct=wrong=unverifiable=0
  for task in packet['tasks']:
   labels=ratings[task['id']]['minute_directions'];p=predictions[task['id']]
   for i,flip in enumerate(p[method]):
    if not flip:continue
    count+=1;correct+=labels[i]=='negative';wrong+=labels[i] in {'positive','balanced','mixed'};unverifiable+=labels[i] in {'unreadable','unsure'}
    if method=='combined_flip' and not p['v3_flip'][i]:
     new_correct+=labels[i]=='negative';new_unsafe+=labels[i]!='negative';new_context_uncertain+=not all(v=='negative' for v in labels)
  result[method]=dict(flipped_minutes=count,confirmed_negative=correct,wrong_direction=wrong,unverifiable=unverifiable)
 return dict(methods=result,new_confirmed_negative_minutes=new_correct,new_unsafe_minutes=new_unsafe,
  new_flips_without_fully_negative_context=new_context_uncertain,
  acceptance_pass=(new_correct>0 and new_unsafe==0 and new_context_uncertain==0
                   and result['combined_flip']['wrong_direction']==0 and result['combined_flip']['unverifiable']==0),
  scope='Small targeted independent acceptance on previously unseen waveform intervals. Not a population error estimate; results never change production corrections automatically.')

def export(packet,ratings):
 out=StringIO(newline='');fields=['id','participant','minute1_direction','minute2_direction','minute3_direction','status','confidence','notes','reviewer','updated_at','version','review_mode']
 writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
 for t in packet['tasks']:
  r=ratings.get(t['id'])
  if not r or r.get('review_mode')!=MODE:continue
  row=dict(id=t['id'],participant=t['participant'],**r)
  for i,v in enumerate(r['minute_directions']):row[f'minute{i+1}_direction']=v
  writer.writerow({k:csv_safe(row.get(k,'')) for k in fields})
 return out.getvalue().encode('utf-8-sig')

def attach(Main,packet,predictions,db,access,secret,cookie):
 assets={'/flip-acceptance':('flip_acceptance.html','text/html'),'/flip_acceptance.js':('flip_acceptance.js','text/javascript')}
 Review=make_handler(packet,db,access,secret,public_fn=public,validate_fn=validate,csv_fn=export,cookie_name=cookie,assets_override=assets,summary_fn=summary,results_fn=lambda p,r:compare(p,predictions,r),export_filename='flip_acceptance_labels.csv')
 class Handler(Main):
  def dispatch(self,method):
   old=self.path
   if old.startswith('/accept-api/'):self.path='/api/'+old[len('/accept-api/'):]
   try:getattr(Review,method)(self)
   finally:self.path=old
  def do_GET(self):
   path=urlsplit(self.path).path
   if path in assets or path.startswith('/accept-api/'):self.dispatch('do_GET')
   else:super().do_GET()
  def do_POST(self):
   if urlsplit(self.path).path.startswith('/accept-api/'):self.dispatch('do_POST')
   else:super().do_POST()
 return Handler
