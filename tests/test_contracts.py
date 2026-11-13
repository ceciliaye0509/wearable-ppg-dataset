import ast,unittest
from pathlib import Path
import numpy as np
from hrv.common.windowing import _candidate_windows
from hrv.common.signals import _rawslots_to_grid
from hrv.v46.policies import bounded_resample,segment_support,canonical_source
from hrv.v46.baseline import baseline_nn
from hrv.common.metrics import summarize_nn

class Contracts(unittest.TestCase):
    def test_epoch_window_rule(self):
        ppg={d:{'timestamp':np.array([1001.,1000001.])} for d in ['Earring','Ring','Watch']}
        a,b,_=_candidate_windows(ppg,{'time_ms':np.array([500.,1000001.])},devices=list(ppg),window_sec=300,stride_sec=30)
        self.assertEqual(a[0],30000.);self.assertTrue(np.all(b-a==300000));self.assertTrue(np.all(np.diff(a)==30000))
    def test_slot_chooses_nearest_sample(self):
        v,m,t,r=_rawslots_to_grid(np.array([3.,8.,5.]),np.array([9.,12.,20.]),np.arange(3)*10.,slot_tolerance_ms=5.)
        self.assertEqual(v[1],3.);self.assertEqual(t[1],9.);self.assertFalse(m[0])
    def test_no_long_gap_interpolation(self):
        v,m,_=bounded_resample(np.array([1.,2.]),np.array([0.,300.]),np.arange(0.,301.,10.),max_gap_ms=250.)
        self.assertEqual(m.sum(),2);self.assertTrue(np.isnan(v[1:-1]).all())
    def test_new_boundaries_have_guard(self):
        t=np.r_[np.arange(0.,100000.,10.),np.arange(100300.,300000.,10.)];grid=np.arange(30000)*10.
        s,k=segment_support(t,grid,250.)
        self.assertEqual(len(s),2);self.assertFalse(k[(grid>96990)&(grid<103300)].any())
    def test_conflicting_duplicate_is_missing(self):
        t,x,b,_,_=canonical_source([0.,10.,10.,20.],[np.array([1.,2.,9.,3.])])
        self.assertTrue(b[0][1]);self.assertTrue(np.isnan(x[0][1]))
    def test_rr_invalid_interval_does_not_create_new_pair(self):
        nn,diff=baseline_nn(np.array([0.,800.,1620.,1720.,2620.,3530.]))
        self.assertEqual(len(diff),2)
    def test_segments_not_concatenated_for_rmssd(self):
        result=summarize_nn([(np.array([800.,820.]),np.array([20.])),(np.array([900.,910.]),np.array([10.]))])
        self.assertEqual(result['n_pairs'],2);self.assertEqual(result['rmssd_ms'],np.sqrt(250.))
    def test_no_cross_version_or_analysis_import(self):
        root=Path(__file__).resolve().parents[1]/'src/hrv'
        for version,other in [('hf_rawslot','v46'),('v46','hf_rawslot')]:
            for p in (root/version).glob('*.py'):
                s=p.read_text();tree=ast.parse(s)
                for n in ast.walk(tree):
                    if isinstance(n,ast.ImportFrom):self.assertFalse((n.module or '').startswith('hrv.'+other))
                self.assertNotIn('/analysis/',s)
    def test_masked_loss_excludes_protected_logits(self):
        import torch
        from types import SimpleNamespace
        from hrv.v46.loss import masked_window_loss
        x=torch.tensor([0.,20.,0.],requires_grad=True)
        s=SimpleNamespace(target=np.array([0.,1.,1.]),valid=np.array([True,False,True]))
        masked_window_loss([x],[s]).backward();self.assertEqual(x.grad[1].item(),0.)
    def test_hf_keeps_p5_and_pan_gate(self):
        p=Path(__file__).resolve().parents[1]/'src/hrv/hf_rawslot/generate.py'
        s=p.read_text();self.assertIn('boundary_keep_arr & ecg_keep_arr',s);self.assertIn('default="P2,P14,P16,P17"',s)

    def test_detector_failure_keeps_rejected_candidate(self):
        from hrv.v46.ecg import ECGReference,CentralGate
        from hrv.v46.context import PaddingGuard
        obj=ECGReference.__new__(ECGReference)
        obj.t=np.arange(0.,320001.,10.);obj.raw=np.sin(obj.t/100.)
        obj.acquisition=obj.raw;obj.corrected=obj.raw;obj.fs0=100.
        obj.central=CentralGate(obj.t,obj.raw);obj.padding=PaddingGuard(obj.t,obj.raw)
        obj.exclusions=[];obj.unreadable=[]
        obj.ref=dict(manual_rejections=[],full300_exceptions=[],patches=[])
        def fail(*args):raise ValueError('test detector failure')
        obj.detect=fail
        label=obj.label(10000.,310000.)
        self.assertFalse(label['ecg_label_qc_pass'])
        self.assertIn('detector_failure',label['ecg_label_qc_reason'])
        self.assertTrue(np.isnan(label['ecg_rmssd_corrected_ms']))

    def test_v46_targets_ignore_external_target_mode(self):
        from unittest.mock import patch
        from hrv.v46.model import peak_targets
        with patch.dict('os.environ',{'PEAK_TARGET_MODE':'gaussian'}):
            actual=peak_targets(np.array([200.]),30)
        self.assertEqual(np.flatnonzero(actual).tolist(),[8,9,10,11,12])
    def test_v46_decoder_uses_frozen_distance(self):
        from unittest.mock import patch
        from hrv.v46.model import peak_indices
        prob=np.zeros(60);prob[[10,20,40]]=[.7,.8,.9]
        with patch.dict('os.environ',{'PEAK_MIN_DISTANCE_MS':'1000','PEAK_PROMINENCE':'1.0'}):
            peaks=peak_indices(prob,.5)
        self.assertEqual(peaks.tolist(),[20,40])
    def test_duplicate_heldout_participant_is_rejected(self):
        from hrv.v46.model import validate_folds
        records=[dict(fold=i,train=['A'],val=['B'],test=['C'+str(i)]) for i in range(4)]
        records[1]['test']=['C0']
        with self.assertRaises(ValueError):validate_folds(records)

if __name__=='__main__':unittest.main()
