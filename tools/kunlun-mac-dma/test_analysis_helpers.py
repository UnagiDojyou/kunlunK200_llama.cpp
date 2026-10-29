import unittest
from analysis_helpers import ticks_by_name,parse_standard,summary
class AnalysisTests(unittest.TestCase):
 def test_interleaved_profile_names_and_warmup(self):
  text='TICKS old [99,0]\nTICKS new [88,1]\nTICKS new [12,1]\nTICKS old [24,0]\nTICKS old [26,0]\nTICKS new [14,1]'
  groups=ticks_by_name(text)
  self.assertEqual(summary(v[0] for v in groups['old'][1:])['median'],25)
  self.assertEqual(summary(v[0] for v in groups['new'][1:])['median'],13)
  self.assertTrue(all(v[1]==1 for v in groups['new']))
 def test_raw_duration_and_reciprocal_throughput(self):
  text='log before\n{"build_commit":"x","n_prompt":512,"n_gen":0,"samples_ns":[1000000000,2000000000]}\n{"build_commit":"x","n_prompt":0,"n_gen":128,"samples_ns":[4000000000,8000000000]}\nlog after'
  rows=parse_standard('std-7b-row-u128-a1-m0-i3',text)
  self.assertEqual(rows[0]['ms']['mean'],1500)
  self.assertEqual(rows[0]['ts']['mean'],384)
  self.assertEqual(rows[1]['ts']['mean'],24)
  self.assertEqual(rows[1]['order'],3)
if __name__=='__main__':unittest.main()
