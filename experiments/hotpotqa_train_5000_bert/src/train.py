"""Full BERT fine-tuning on question text only; soft CE selection on validation.
Data features are built exclusively from queries.jsonl.question_text.
Labels/curves are joined separately as supervision/evaluation, never tokenizer input.
"""
import os,random,time,math
from common import *
from device import check_device,memory_snapshot

def main():
    cfg=config();tc=cfg['training'];os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    device_info=check_device(cfg);write_json(HERE/'reports/gpu_preflight.json',device_info);print('GPU_PREFLIGHT',device_info,flush=True)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    from transformers import AutoTokenizer,BertForSequenceClassification,get_linear_schedule_with_warmup
    torch.set_num_threads(4);seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    metadata=read_json(HERE/'data/metadata.json')
    for name,digest in metadata['artifacts_sha256'].items():require(sha256(HERE/'data'/name)==digest,'Prepared data hash differs '+name)
    splits=read_json(HERE/'data/splits.json');queries={x['question_id']:x['question_text'] for x in rows(HERE/'data/queries.jsonl')};labels={x['question_id']:x for x in rows(HERE/'data/labels.jsonl')}
    tokenizer=AutoTokenizer.from_pretrained(tc['model'],local_files_only=True,use_fast=True)
    lengths=[len(tokenizer.encode(q,add_special_tokens=True)) for q in queries.values()]
    write_json(HERE/'reports/question_token_lengths.json',{'max':max(lengths),'over_max_length':sum(n>tc['max_length'] for n in lengths),'max_length':tc['max_length'],'input_fields':['question_text']})
    class Questions(Dataset):
        def __init__(self,ids,valid_only):
            self.ids=[q for q in ids if not valid_only or labels[q]['label_status']=='valid']
            self.features=tokenizer([queries[q] for q in self.ids],padding='max_length',truncation=True,max_length=tc['max_length'],return_tensors='pt')
        def __len__(self):return len(self.ids)
        def __getitem__(self,i):
            features={k:v[i] for k,v in self.features.items()}
            target=labels[self.ids[i]]['soft_label'];return features,torch.tensor(target if target is not None else [0.]*15,dtype=torch.float32),i
    train=Questions(splits['train'],True);val=Questions(splits['validation'],True)
    require(len(train)>0 and len(val)>0,'No valid training/validation labels')
    require(not (HERE/'checkpoints/best/config.json').exists(),'Existing checkpoint; refusing overwrite')
    generator=torch.Generator();generator.manual_seed(seed)
    tl=DataLoader(train,batch_size=tc['batch_size'],shuffle=True,generator=generator,num_workers=0,pin_memory=True)
    vl=DataLoader(val,batch_size=tc['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
    model=BertForSequenceClassification.from_pretrained(tc['model'],num_labels=15,local_files_only=True)
    model.config.id2label={i:'k='+str(i+1) for i in range(15)};model.config.label2id={v:k for k,v in model.config.id2label.items()}
    require(all(p.requires_grad for p in model.parameters()),'Encoder unexpectedly frozen')
    total_parameters=sum(p.numel() for p in model.parameters());encoder_parameters=sum(p.numel() for p in model.bert.parameters())
    initial_encoder_probe=model.bert.encoder.layer[0].attention.self.query.weight.detach().cpu().clone()
    model.to('cuda:0');optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    steps=math.ceil(len(tl)/tc['gradient_accumulation'])*tc['epochs'];scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*tc['warmup_ratio']),steps)
    scaler=torch.cuda.amp.GradScaler(enabled=True);best=float('inf');history=[];start=time.time()
    mean=np.mean([labels[q]['soft_label'] for q in train.ids],axis=0)
    write_json(HERE/'data/train_mean_distribution.json',{'probabilities':mean.tolist(),'training_valid_count':len(train),'source_split':'train','source_label_status':'valid','source_ids_sha256':text_hash('\n'.join(train.ids))})
    write_json(HERE/'reports/training_configuration.json',{'config':cfg,'device':device_info,'trainable_parameters':total_parameters,'trainable_encoder_parameters':encoder_parameters,'counts':splits['counts'],'feature_fields':['question_text'],'full_parameter_finetuning':True,'initial_memory':memory_snapshot()})
    for epoch in range(1,tc['epochs']+1):
        model.train();total_loss=0.;total_n=0;optimizer.zero_grad(set_to_none=True)
        for step,(features,target,_) in enumerate(tl,1):
            features={k:v.to('cuda:0',non_blocking=True) for k,v in features.items()};target=target.to('cuda:0',non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):logits=model(**features).logits
            loss=-(target*torch.log_softmax(logits.float(),dim=-1)).sum(-1).mean()
            require(torch.isfinite(loss).item(),'Nonfinite training loss')
            scaler.scale(loss/tc['gradient_accumulation']).backward()
            if step%tc['gradient_accumulation']==0 or step==len(tl):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip'])
                previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=previous_scale:scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss+=loss.item()*len(target);total_n+=len(target)
            if step%20==0 or step==len(tl):
                log={'epoch':epoch,'step':step,'train_soft_ce_running':total_loss/total_n,'lr':scheduler.get_last_lr()[0],'seconds':time.time()-start,'memory':memory_snapshot()};append_json(HERE/'logs/batches.jsonl',log);print('TRAIN',log,flush=True)
        model.eval();val_ce=val_entropy=0.;val_n=0
        with torch.no_grad():
            for features,target,_ in vl:
                features={k:v.to('cuda:0') for k,v in features.items()};target=target.to('cuda:0')
                with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):logits=model(**features).logits
                val_ce+=(-(target*torch.log_softmax(logits.float(),-1)).sum(-1)).sum().item();val_entropy+=(-(target*torch.log(target.clamp_min(1e-30))).sum(-1)).sum().item();val_n+=len(target)
        row={'epoch':epoch,'train_soft_ce':total_loss/total_n,'validation_soft_ce':val_ce/val_n,'validation_kl':(val_ce-val_entropy)/val_n,'seconds':time.time()-start,'memory':memory_snapshot()};history.append(row);append_json(HERE/'logs/epochs.jsonl',row);print('EPOCH',row,flush=True)
        if row['validation_soft_ce']<best:
            best=row['validation_soft_ce'];model.save_pretrained(HERE/'checkpoints/best');tokenizer.save_pretrained(HERE/'checkpoints/best')
            write_json(HERE/'checkpoints/best/selection.json',dict(row,selection='minimum_validation_soft_ce',test_used_for_selection=False))
    encoder_change=(model.bert.encoder.layer[0].attention.self.query.weight.detach().cpu()-initial_encoder_probe).abs()
    require(encoder_change.max().item()>0,'Encoder weights did not update')
    best_info=read_json(HERE/'checkpoints/best/selection.json');write_json(HERE/'reports/training_summary.json',{'history':history,'best':best_info,'seconds':time.time()-start,'final_memory':memory_snapshot(),'device':device_info,'encoder_probe_mean_abs_change':encoder_change.mean().item(),'encoder_probe_max_abs_change':encoder_change.max().item()})
    # First test inference happens only after best checkpoint selection is complete.
    model=BertForSequenceClassification.from_pretrained(HERE/'checkpoints/best',local_files_only=True).to('cuda:0');model.eval()
    inference=[]
    for split in ('validation','test'):
        ds=Questions(splits[split],False);dl=DataLoader(ds,batch_size=tc['batch_size'],shuffle=False,num_workers=0)
        with torch.no_grad():
            for features,_,indices in dl:
                features={k:v.to('cuda:0') for k,v in features.items()}
                with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):logits=model(**features).logits
                probabilities=torch.softmax(logits.float(),-1).cpu().numpy()
                for i,p in zip(indices.tolist(),probabilities):inference.append({'question_id':ds.ids[i],'split':split,'probabilities':p.astype(float).tolist(),'argmax_k':int(p.argmax())+1})
    write_rows(HERE/'data/bert_predictions.jsonl',inference);print('TRAINING_COMPLETE',best_info,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write_json(HERE/'reports/training_failure.json',{'exception':repr(exc),'automatic_cpu_or_other_gpu_fallback':False});raise
