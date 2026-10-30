"""Deferred flags, immutable first-pass snapshots and reviewer isolation."""
import csv,io,json,sqlite3
from test_full_window import FullWindowAPITests,clip_fixture
from serve_review import make_handler

class DeferredScreeningTests(FullWindowAPITests):
    def setUp(self):
        super().setUp()
        self.clip['deferredScreeningReview']=True
        self.clip['qualityFlags']=[dict(code='severe_source_block',label='Amplitude alert',startIndex=2,endIndex=8,rawSpanUv=6100)]
        self.clip['qualitySummary']=dict(available=True,requiresReview=True)
        packet=dict(participants=[dict(participant='P1')],clips=[self.legacy,self.clip],reviewBatches=[dict(id='existing',label='Existing',target=1),dict(id='full_window_hrv_v1',label='Review',target=1)])
        handler=make_handler(packet,self.db,'test-access-code',b'test-session-secret-32-characters',False)
        handler.log_message=lambda *args:None
        self.server.RequestHandlerClass=handler

    def test_flags_require_complete_first_pass_and_explicit_reveal(self):
        value=self.load();self.assertEqual(value['qualityFlags'],[]);self.assertTrue(value['screeningWithheld'])
        self.assertEqual(self.request('POST','/api/clip/test_clip/reveal-screening',{'version':0})[0],400)
        self.assertEqual(self.save_full(peaks=[4,6],pages=list(range(60)),ranges=[[10,20]],status='complete')[0],200)
        self.assertEqual(self.load()['qualityFlags'],[])
        _,_,raw=self.request('GET','/api/export.csv')
        self.assertTrue(all(row['quality_flags_json']=='[]' for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))))
        self.assertEqual(self.request('POST','/api/clip/test_clip/reveal-screening',{'version':0})[0],400)
        self.assertEqual(self.request('POST','/api/clip/test_clip/reveal-screening',{'version':1})[0],200)
        value=self.load();self.assertFalse(value['screeningWithheld']);self.assertEqual(len(value['qualityFlags']),1)
        self.assertEqual(value['peaks'],[4,6]);self.assertEqual(value['unreadableRanges'],[[10,20]])
        self.assertEqual(self.save_full(version=1,peaks=[5],pages=list(range(60)),status='complete')[0],200)
        self.assertEqual(self.request('POST','/api/clip/test_clip/reveal-screening',{'version':2})[0],200)
        _,_,raw=self.request('GET','/api/export.csv')
        exported=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        first=json.loads(exported[0]['first_pass_annotation_json']);self.assertEqual(json.loads(first['peaks_json']),[4,6])
        self.assertEqual(json.loads(first['unreadable_ranges_json']),[[10,20]])
        self.login('Other reviewer');self.assertTrue(self.load()['screeningWithheld']);self.assertEqual(self.load()['peaks'],[2,8])
        _,_,raw=self.request('GET','/api/full-window-export.csv')
        exported=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        self.assertTrue(exported)
        self.assertTrue(all(row['quality_flags_json']=='[]' for row in exported))

    def test_hidden_counts_do_not_disclose_assignment(self):
        _,_,raw=self.request('GET','/api/participants');value=json.loads(raw)
        self.assertEqual(value['participants'][0]['batches']['full_window_hrv_v1']['qualityFlagged'],0)
        _,_,raw=self.request('GET','/api/participant/P1/clips')
        value=next(c for c in json.loads(raw)['clips'] if c['id']=='test_clip')
        self.assertEqual(value['qualityWarningCount'],0);self.assertEqual(value['qualityWarningLabels'],[])

    # Inherited generic tests expect flags for their unblinded fixtures. Keep
    # their original fixture by running them on the unmodified parent class.
    def test_own_progress_default_export_all_export_and_legacy_preservation(self):
        self.clip.pop('deferredScreeningReview',None)
        super().test_own_progress_default_export_all_export_and_legacy_preservation()
