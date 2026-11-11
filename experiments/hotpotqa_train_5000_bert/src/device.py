"""Fail-closed RTX 4090 binding; never choose another GPU or CPU."""
import os,subprocess,csv,io,time
from common import require,HERE,write_json

def check_device(cfg):
    target=cfg['training'];uuid=target['gpu_uuid']
    fields='uuid,name,memory.total,memory.used,memory.free,driver_version,index'
    text=subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
    devices=[list(map(str.strip,r)) for r in csv.reader(io.StringIO(text))];matches=[r for r in devices if r[0]==uuid]
    require(len(matches)==1,'Required 4090 UUID unavailable')
    r=matches[0]
    write_json(HERE/'reports/latest_gpu_check.json',{'gpu_uuid':r[0],'name':r[1],'total_mib':int(r[2]),'used_mib':int(r[3]),'free_mib':int(r[4]),'checked_unix':time.time()})
    require(r[1]==target['required_gpu_name'],'Unexpected GPU model')
    require(int(r[4])>=target['minimum_free_mib'],'RTX 4090 has insufficient free memory; stop')
    # Must run before importing torch.
    # PyTorch 1.13 device_count misreports UUID visibility on this host; bind
    # the physical index resolved from the REQUIRED UUID, then recheck model.
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=r[6]
    import torch
    require(torch.cuda.is_available(),'CUDA unavailable; CPU fallback forbidden')
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==target['required_gpu_name'],'CUDA device binding mismatch')
    torch.cuda.set_device(0)
    x=torch.ones((32,32),device='cuda');require((x@x).sum().item()==32768,'CUDA compute failed')
    return {'gpu_uuid':uuid,'gpu_name':r[1],'memory_total_mib':int(r[2]),'memory_used_before_mib':int(r[3]),'memory_free_before_mib':int(r[4]),'driver':r[5],'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'logical_device':'cuda:0','torch_device_properties':str(torch.cuda.get_device_properties(0)),'pid':os.getpid()}

def memory_snapshot():
    import torch
    free,total=torch.cuda.mem_get_info(0)
    return {'allocated_mib':torch.cuda.memory_allocated(0)/2**20,'reserved_mib':torch.cuda.memory_reserved(0)/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated(0)/2**20,'free_mib':free/2**20,'total_mib':total/2**20}
