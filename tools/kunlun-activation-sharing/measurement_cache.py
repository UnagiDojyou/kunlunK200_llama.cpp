"""Result identity includes executable, host libraries, flags and input fixtures."""
from pathlib import Path
import hashlib,json

def fingerprint(command,environment):
 command=list(map(str,command));exe=Path(command[0]);files={exe.resolve()}
 files.update(p.resolve() for p in exe.parent.glob('*.so*') if p.is_file())
 for directory in environment.get('LD_LIBRARY_PATH','').split(':'):
  if not directory:continue
  library=Path(directory)/'libkunlunk200.so'
  if library.is_file():files.add(library.resolve());break
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
