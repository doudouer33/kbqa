"""Single-instance durable offline build -> RTX4090 training -> evaluation."""
import fcntl,subprocess,sys,time
from common import HERE,ROOT,write_json

def main():
    with (HERE/'logs/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,script in [('building_labels','build_data.py'),('training_bert','train.py'),('evaluating','evaluate.py')]:
            if name=='building_labels' and (HERE/'data/metadata.json').exists():
                print('STAGE_SKIP',name,'prepared metadata exists; training rechecks artifact hashes',flush=True)
                continue
            write_json(HERE/'reports/status.json',{'stage':name,'updated_unix':time.time(),'data_construction_complete':name!='building_labels','training_started':name!='building_labels','evaluation_complete':False})
            print('STAGE_START',name,flush=True)
            with (HERE/'logs'/(name+'.log')).open('w') as stream:
                result=subprocess.run([sys.executable,'-u',str(HERE/'src'/script)],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            print('STAGE_EXIT',name,result.returncode,flush=True)
            if result.returncode:
                write_json(HERE/'reports/status.json',{'stage':name,'failed':True,'returncode':result.returncode,'log':str(HERE/'logs'/(name+'.log'))})
                raise SystemExit(result.returncode)
        write_json(HERE/'reports/status.json',{'stage':'complete','data_construction_complete':True,'training_started':True,'training_complete':True,'evaluation_complete':True})
        print('PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
