import unittest,tempfile
from pathlib import Path
from measurement_cache import fingerprint
class CacheTests(unittest.TestCase):
 def test_library_flags_and_fixture_invalidate(self):
  workspace=Path(__file__).resolve().parent
  with tempfile.TemporaryDirectory(dir=workspace) as d:
   self.assertTrue(Path(d).resolve().is_relative_to(workspace))
   root=Path(d);exe=root/'program';exe.write_bytes(b'executable');library=root/'libkunlunk200.so';library.write_bytes(b'library-A')
   fixture=root/'input.ids';fixture.write_text('1 2 3')
   env={'LD_LIBRARY_PATH':str(root),'GGML_KUNLUN_Q4_ACT_SHARE':'0','K200_INPUT_IDS':str(fixture)}
   a=fingerprint([exe],env)[0];self.assertEqual(a,fingerprint([exe],env)[0])
   kernels=root/'kunlunk200/kernels';kernels.mkdir(parents=True);(kernels/'kernel.bin').write_bytes(b'GPU-A');(kernels/'kernel.entry').write_text('entry')
   ka=fingerprint([exe],env)[0];(kernels/'kernel.bin').write_bytes(b'GPU-B');self.assertNotEqual(ka,fingerprint([exe],env)[0])
   library.write_bytes(b'library-B');b=fingerprint([exe],env)[0];self.assertNotEqual(a,b)
   env['GGML_KUNLUN_Q4_ACT_SHARE']='1';c=fingerprint([exe],env)[0];self.assertNotEqual(b,c)
   fixture.write_text('1 2 4');self.assertNotEqual(c,fingerprint([exe],env)[0])
   observers=root/'observers';observers.mkdir();observer=observers/'audit.so';observer.write_bytes(b'observer-A');env['LD_PRELOAD']=str(observer)
   before=fingerprint([exe],env)[0];observer.write_bytes(b'observer-B');self.assertNotEqual(before,fingerprint([exe],env)[0])
if __name__=='__main__':unittest.main()
