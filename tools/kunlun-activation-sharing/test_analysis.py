import unittest,importlib.util,json,statistics
from pathlib import Path
spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class AnalysisTests(unittest.TestCase):
 def test_interleaved_bench_stderr(self):
  a=dict(build_commit='test',samples_ns=[1,2,3],n_prompt=512,n_gen=0)
  b=dict(build_commit='test',samples_ns=[4,5,6],n_prompt=0,n_gen=128)
  text='GUARD START\n[\n'+json.dumps(a)+'KUNLUN stats: calls=3\n,\n'+json.dumps(b)+'KUNLUN activation graph\n]'
  self.assertEqual(module.bench_rows(text),[a,b])
 def test_even_median_and_sample_std(self):
  s=module.summary([1.,2.,4.,10.]);self.assertEqual(s['median'],3.);self.assertEqual(s['n'],4);self.assertEqual(s['stddev'],statistics.stdev([1.,2.,4.,10.]))
 def test_named_counter_and_float(self):
  self.assertEqual(module.kv_values('phase=warmup.tg quant_chunks=128 quant_ms=2.300'),dict(phase='warmup.tg',quant_chunks='128',quant_ms='2.300'))
if __name__=='__main__':unittest.main()
