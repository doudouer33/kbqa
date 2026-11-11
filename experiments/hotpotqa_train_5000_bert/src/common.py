"""Experiment-local IO; no RAG or historical-experiment imports."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]

def require(ok,message):
    if not ok:raise ValueError(message)
def unique(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,'Duplicate JSON key: '+k);d[k]=v
    return d
def read_json(path):return json.loads(Path(path).read_text(),object_pairs_hook=unique)
def rows(path):
    with Path(path).open() as f:
        for n,line in enumerate(f,1):
            require(bool(line.strip()),'Blank JSONL '+str(path)+':'+str(n))
            yield json.loads(line,object_pairs_hook=unique)
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def write_rows(path,records):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):d.update(chunk)
    return d.hexdigest()
def text_hash(s):return hashlib.sha256(s.encode()).hexdigest()
def config():return read_json(HERE/'config/experiment.json')
def append_json(path,obj):
    with Path(path).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()
