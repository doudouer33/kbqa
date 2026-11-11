"""Evaluation must keep invalid labels out of CE and distinguish fixed policies."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate import policy_metrics,paired_interval,ce_values

class EvaluationTests(unittest.TestCase):
    def test_invalid_labels_are_excluded_from_ce_not_answer_metrics(self):
        p=np.ones((2,15))/15;targets=np.zeros((2,15));targets[0,0]=1;valid=np.array([True,False]);f=np.zeros((2,15));f[0,0]=1
        r=policy_metrics(p,f,np.ones((2,15))*10,np.zeros((2,15)),targets,valid)
        self.assertEqual(r['count'],2);self.assertEqual(r['valid_label_count'],1);self.assertAlmostEqual(r['soft_ce'],np.log(15));self.assertAlmostEqual(r['argmax_answer_f1'],.5)
    def test_fixed_policy_infinite_ce_is_not_reported_as_finite(self):
        t=np.array([[.5,.5]+[0]*13]);p=np.array([[1.]+[0]*14]);ce,kl,bad=ce_values(t,p)
        self.assertTrue(bad[0]);self.assertGreater(ce[0],0)
        r=policy_metrics(p,np.zeros((1,15)),np.ones((1,15)),np.zeros((1,15)),t,np.array([True]))
        self.assertIsNone(r['soft_ce']);self.assertEqual(r['infinite_soft_ce_count'],1)
    def test_identical_predictions_match_target_entropy(self):
        t=np.array([[.25,.75]+[0]*13]);ce,kl,bad=ce_values(t,t)
        self.assertAlmostEqual(kl[0],0);self.assertFalse(bad[0])
    def test_paired_interval_zero_differences(self):
        r=paired_interval(np.zeros(10),repetitions=20);self.assertEqual(r['ci95_low'],0);self.assertEqual(r['ci95_high'],0)
if __name__=='__main__':unittest.main(verbosity=2)
