"""Regression tests for copied algorithms and the Train-specific gold join."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import f1_score,model_doc,prepare_gold,match_evidence
from soft_oracle import Parameters,Question,build_label
from build_data import prepare_train_gold,reproduce_reference
from common import config

class CoreTests(unittest.TestCase):
    def q(self,f1=(1.,1.,0.),ev=(1.,1.,1.),cost=(10,20,30)):
        return Question('q','bridge','Q?','A',tuple(range(1,len(f1)+1)),f1,(0,)*len(f1),{'context_tokens':cost},tuple({'sentence_recall_lower':e,'sentence_unknown':0} for e in ev),('',)*len(f1),1)
    def test_original_distribution_reproduction(self):
        self.assertEqual(reproduce_reference(config())['reproduced_count'],1000)
    def test_zero_and_empty_are_null(self):
        p=Parameters.from_dict(config()['soft_label'])
        for q,status in [(self.q(f1=(0.,0.,0.)),'all_zero_f1'),(self.q(ev=(0.,0.,0.)),'empty_candidates')]:
            r=build_label(q,p);self.assertEqual(r['label_status'],status);self.assertIsNone(r['soft_label']);self.assertIsNone(r['argmax_soft_label_k'])
    def test_cost_normalizes_by_all_k_and_epsilon_is_inclusive(self):
        p=Parameters.from_dict(config()['soft_label']);r=build_label(self.q(f1=(.9,1.,0.),ev=(1.,1.,0.)),p)
        self.assertEqual(r['valid_candidates'],(1,2));self.assertAlmostEqual(r['normalized_cost_curve'][0],1/3);self.assertAlmostEqual(sum(r['soft_label']),1)
    def test_official_f1_yes_no_and_empty(self):
        self.assertEqual(f1_score('yes','yes')[0],1);self.assertEqual(f1_score('yes indeed','yes')[0],0);self.assertEqual(f1_score('','Simba')[0],0);self.assertEqual(f1_score('The young Simba','young Simba')[0],1)
    def raw_and_sample(self):
        raw={'_id':'q','question':'Q?','answer':'yes','type':'bridge','level':'easy','context':[['A',['A supporting sentence.']],['B',['Another supporting sentence.']],['Duplicate B',['Another supporting sentence.']]],'supporting_facts':[['A',0],['B',0]]}
        sample={'question_id':'q','question_text':'Q?','type':'bridge','level':'easy','answers_objects':[{'spans':['yes']}],'contexts':[{'title':'A','paragraph_text':'A supporting sentence.','is_supporting':True},{'title':'Duplicate B','paragraph_text':'Another supporting sentence.','is_supporting':True}]}
        return raw,sample
    def test_train_dedup_does_not_change_raw_support_sentences(self):
        raw,sample=self.raw_and_sample();gold=prepare_train_gold(sample,raw);self.assertEqual([g['title'] for g in gold],['A','B']);self.assertEqual(sum(len(g['facts']) for g in gold),2)
    def test_invalid_support_index_stops(self):
        raw,sample=self.raw_and_sample();raw['supporting_facts'][0][1]=20
        with self.assertRaisesRegex(ValueError,'Bad fact index'):prepare_train_gold(sample,raw)
    def test_approved_missing_sentence_keeps_denominator(self):
        raw,sample=self.raw_and_sample();raw['supporting_facts'][0][1]=20
        gold=prepare_train_gold(sample,raw,unknown_facts={('A',20)})
        result=match_evidence(gold,[{'title':'B','paragraph_text':'Another supporting sentence.'}])
        self.assertEqual(result['sentence_total'],2);self.assertEqual(result['sentence_hit'],1)
        self.assertEqual(result['sentence_unknown'],1);self.assertEqual(result['sentence_recall_lower'],.5)
        self.assertEqual(result['sentence_recall_upper'],1)

    def test_sentence_lower_preserves_unknown(self):
        raw,sample=self.raw_and_sample();gold=prepare_train_gold(sample,raw)
        r=match_evidence(gold,[{'title':'A','paragraph_text':'Completely different version of the page.'}])
        self.assertEqual(r['sentence_hit'],0);self.assertEqual(r['sentence_unknown'],1);self.assertEqual(r['sentence_recall_lower'],0);self.assertEqual(r['sentence_recall_upper'],.5)
if __name__=='__main__':unittest.main(verbosity=2)
