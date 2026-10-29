"""Result identity includes executable, host libraries, flags and input fixtures."""
from pathlib import Path
import hashlib,json

def fingerprint(command,environment):
 command=list(map(str,command));exe=Path(command[0]);files={exe.resolve()}
 files.update(p.resolve() for p in exe.parent.glob('*.so*') if p.is_file())
 search=environment.get('LD_LIBRARY_PATH','')
 for directory in ([search] if Path(search).is_dir() else search.split(':')):
  if not directory:continue
  library=Path(directory)/'libkunlunk200.so'
  if library.is_file():
   files.add(library.resolve())
   kernel_dir=Path(environment.get('KUNLUN_KERNEL_DIR') or library.resolve().parent/'kunlunk200/kernels')
   if kernel_dir.is_dir():files.update(p.resolve() for p in kernel_dir.iterdir() if p.is_file() and p.suffix in ['.bin','.entry'])
   break
 preload=environment.get('LD_PRELOAD','')
 for value in ([preload] if Path(preload).is_file() else preload.replace(':',' ').split()):
  p=Path(value)
  if p.is_file():files.add(p.resolve())
 for key in ['K200_INPUT_IDS','K200_TEACHER_FILE','K200_PROMPT_FILE']:
  value=environment.get(key)
  if value and Path(value).is_file():files.add(Path(value).resolve())
 dependencies={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
 models={}
 for value in command:
  p=Path(value)
  if p.suffix=='.gguf' and p.is_file():
   s=p.stat();models[str(p.resolve())]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 identity=dict(command=command,environment=environment,dependencies=dependencies,models=models)
 return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),dependencies
