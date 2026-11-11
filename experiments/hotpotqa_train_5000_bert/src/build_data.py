"""Offline 5000x15 F1/token/sentence-evidence curves and Original Soft Labels.
No HTTP/model generation/retrieval clients are imported. Gold Train support sentence
indices and literal matching are required; paragraph recall is never substituted.
"""
import collections,json,math,random,re,time
from functools import lru_cache
from pathlib import Path
from common import *
from metrics import prepare_gold,match_evidence,model_doc,render_context,f1_score,exact_match_score,compact
from soft_oracle import Parameters,Question,build_label


def prepare_train_gold(sample, raw, unknown_facts=()):
    # Adapt only the join checks to process_hotpotqa.py: it deduplicates equal
    # paragraph bodies, keeps the last title, and clips stored bodies to 1000 words.
    # Evidence itself still uses ORIGINAL raw support sentences and unchanged matcher.
    full = {title: ''.join(sentences) for title, sentences in raw['context']}
    reverse = {''.join(sentences): title.strip() for title, sentences in raw['context']}
    supports = {full[title] for title, _ in raw['supporting_facts']}
    expected = {title: {'text': ' '.join(body.strip().split(' ')[:1000]), 'support': body in supports}
                for body, title in reverse.items()}
    actual = {r['title']: r for r in sample['contexts']}
    require(len(actual)==len(sample['contexts']) and set(actual)==set(expected), 'Train preprocessing title mismatch '+sample['question_id'])
    for title, item in actual.items():
        require(compact(item['paragraph_text'])==compact(expected[title]['text']), 'Train preprocessing text mismatch '+sample['question_id']+'/'+title)
        require(item['is_supporting']==expected[title]['support'], 'Train preprocessing support flag mismatch')
    canonical = dict(sample, contexts=[{'title': title, 'paragraph_text': ''.join(sentences),
                                       'is_supporting': title in {t for t,_ in raw['supporting_facts']}}
                                      for title,sentences in raw['context']])
    return prepare_gold(canonical, raw, unknown_facts=unknown_facts)


def make_question(r):
    return Question(r['question_id'],r['type'],r['question_text'],r.get('gold_answer',''),tuple(r.get('k_values',range(1,16))),tuple(r['f1_curve']),tuple(r.get('em_curve',[0]*15)),{'context_tokens':tuple(r['token_cost_curve'])},tuple(r['evidence_diagnostic']),tuple(r.get('predictions',['']*15)),r['original_hard_oracle_k'])


def summary(labels):
    valid=[r for r in labels if r['label_status']=='valid'];n=len(labels);states=collections.Counter(r['label_status'] for r in labels)
    result={'total':n,'valid_count':len(valid),'statuses':{s:{'count':states[s],'ratio':states[s]/n if n else None} for s in ('valid','all_zero_f1','empty_candidates')},'mean_probability':[sum(r['soft_label'][i] for r in valid)/len(valid) if valid else None for i in range(15)],'mean_entropy_nats':sum(r['entropy'] for r in valid)/len(valid) if valid else None,'argmax_k_histogram':{str(k):sum(r['argmax_soft_label_k']==k for r in valid) for k in range(1,16)},'candidate_count_histogram':dict(sorted(collections.Counter(r['valid_candidate_count'] for r in valid).items()))}
    return result


def reproduce_reference(cfg):
    reference=list(rows(ROOT/cfg['reference_labels']));params=Parameters.from_dict(cfg['soft_label'])
    for r in reference:
        require(r['parameters']==params.to_dict(),'Historical reference parameter mismatch')
        label=build_label(make_question(r),params)
        require(label['label_status']==r['label_status'],'Historical status not reproduced')
        require(list(label['valid_candidates'])==r['valid_candidates'],'Historical candidates not reproduced')
        require(label['argmax_soft_label_k']==r['argmax_soft_label_k'],'Historical argmax not reproduced')
        if r['soft_label'] is not None:
            require(max(abs(a-b) for a,b in zip(label['soft_label'],r['soft_label']))<1e-12,'Historical probability not reproduced')
    return {'reproduced_count':len(reference),'probability_tolerance':1e-12,'summary':summary(reference),'source_sha256':sha256(ROOT/cfg['reference_labels'])}


def main():
    start=time.time();cfg=config();out=HERE/'data';require(not (out/'metadata.json').exists(),'Data already built; refusing to replace')
    reference=reproduce_reference(cfg);print('Historical Original Soft Label reproduced:',reference['reproduced_count'],flush=True)
    source=ROOT/cfg['data_source'];samples=list(rows(source/'data/train_5000.jsonl'));index={r['question_id']:r for r in samples};qids=list(index)
    require(len(samples)==len(index)==cfg['expected_questions'],'Sample count/duplicates')
    raw_path=ROOT/cfg['raw_gold'];require(raw_path.is_file(),'Official Train supporting facts missing')
    print('Loading official Train supporting facts',flush=True)
    raw=json.loads(raw_path.read_text());raw_index={r['_id']:r for r in raw if r['_id'] in index};require(set(raw_index)==set(index),'Official Train ID coverage missing')
    allowed = cfg['unavailable_supporting_facts']
    gold={qid:prepare_train_gold(sample,raw_index[qid],unknown_facts={tuple(f) for f in allowed.get(qid,[])}) for qid,sample in index.items()};del raw
    unknown_items=[{'question_id':qid,'title':g['title'],'sentence_index':f['sentence_index']} for qid,gs in gold.items() for g in gs for f in g['facts'] if f.get('unavailable_official_sentence')]
    require(len(unknown_items)==2, 'Approved unavailable support fact count changed')
    write_json(HERE/'reports/approved_unknown_facts.json',{'policy':'user_approved_unknown_keep_original_denominator','items':unknown_items})
    manifest=read_json(source/'runs/run_manifest.json');require(sha256(source/'data/train_5000.jsonl')==manifest['datasets']['full']['sha256'],'Frozen dataset hash mismatch')
    curves={qid:{'question_id':qid,'question_text':s['question_text'],'type':s['type'],'level':s['level'],'gold_answer':raw_index[qid]['answer'],'k_values':cfg['k_values'],'f1_curve':[],'em_curve':[],'token_cost_curve':[],'evidence_recall_curve':[],'evidence_diagnostic':[],'predictions':[],'answer_fallback_curve':[]} for qid,s in index.items()};del raw_index
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from transformers import PreTrainedTokenizerFast
    tokenizer=PreTrainedTokenizerFast.from_pretrained(cfg['context_tokenizer'],local_files_only=True,model_max_length=120000)
    @lru_cache(maxsize=100000)
    def count_tokens(text):return len(tokenizer.encode(text,add_special_tokens=False))
    base=source/'runs/predictions'/manifest['set_names']['full'];sources={};anomalies=collections.Counter();per_k={}
    regex=re.compile(r'(?s).* answer is:? ([^\r\n]*)\.?')
    for k in cfg['k_values']:
        d=next(base.glob('*retrieval_count__'+str(k)+'___*'));marker=read_json(d/'complete.json');require(marker['status']=='complete' and marker['predict_exit_code']==0,'Incomplete RAG output k='+str(k))
        binding={'manifest_sha256':sha256(source/'runs/run_manifest.json'),'input_sha256':manifest['datasets']['full']['sha256'],'config_sha256':manifest['configs'][str(k)]['sha256'],'k':k,'set_name':manifest['set_names']['full']}
        require(marker['binding']==binding,'Completion binding mismatch')
        for name,digest in marker['output_hashes'].items():require(sha256(d/name)==digest,'Output hash mismatch '+str(d/name))
        pred_path=d/'prediction__hotpotqa_to_hotpotqa__train_5000.json';predictions=read_json(pred_path);require(set(predictions)==set(qids),'Prediction IDs differ')
        seen=set();stats=collections.Counter()
        for row in rows(d/'contexts.jsonl'):
            qid=row['qid'];require(qid in curves and qid not in seen,'Unexpected/duplicate record');seen.add(qid)
            require(row['binding']==binding and row['k']==k,'Context binding mismatch')
            require(row['input_question']==index[qid]['question_text'].strip(),'Question text mismatch')
            require(isinstance(predictions[qid],str) and predictions[qid]==row['prediction'],'Prediction mismatch')
            require(text_hash(row['effective_context_text'])==row['context_sha256'],'Context hash mismatch')
            require(text_hash(row['chain'])==row['chain_sha256'],'Chain hash mismatch')
            require(row['effective_context_verified_in_prompt'] is True,'Context missing from prompt')
            require(0<len(row['raw_contexts'])==len(row['reader_contexts'])<=k,'Context count mismatch')
            require([render_context(d) for d in row['raw_contexts']]==row['reader_contexts'],'Reader context mismatch')
            require('\n\n'.join(row['reader_contexts'])==row['effective_context_text'],'Effective context mismatch')
            require(row['api']['finish_reasons'] in (None,['stop']),'Abnormal API finish')
            count=count_tokens(row['effective_context_text']);require(count>0,'Nonpositive context token count')
            ev=match_evidence(gold[qid],[model_doc(d) for d in row['raw_contexts']]);c=curves[qid]
            c['f1_curve'].append(f1_score(row['prediction'],c['gold_answer'])[0]);c['em_curve'].append(int(exact_match_score(row['prediction'],c['gold_answer'])))
            c['token_cost_curve'].append(count);c['evidence_recall_curve'].append(ev['sentence_recall_lower']);c['predictions'].append(row['prediction'])
            c['evidence_diagnostic'].append({key:value for key,value in ev.items() if key not in ('paragraphs','facts')})
            raw_generation=row['raw_generation'].strip()
            try:
                decoded=json.loads(raw_generation)
                if isinstance(decoded,str):raw_generation=decoded
            except ValueError:pass
            fallback=regex.match(raw_generation) is None;c['answer_fallback_curve'].append(fallback)
            stats['empty_answer']+=not row['prediction'].strip();stats['answer_extraction_fallback']+=fallback;stats['api_usage_missing']+=row['api']['usage'] is None;stats['prompt_cropped']+=row['prompt_cropped'];stats['sentence_unknown']+=ev['sentence_unknown']
        require(seen==set(qids),'Context IDs missing');per_k[str(k)]=dict(stats);anomalies.update(stats)
        sources[str(k)]={'complete_sha256':sha256(d/'complete.json'),'prediction_sha256':sha256(pred_path),'contexts_sha256':sha256(d/'contexts.jsonl')}
        print('OFFLINE k={} count={} elapsed={:.1f}s stats={}'.format(k,len(seen),time.time()-start,dict(stats)),flush=True)
    labels=[];params=Parameters.from_dict(cfg['soft_label'])
    for qid in qids:
        c=curves[qid];c['original_hard_oracle_k']=1+max(range(15),key=lambda i:(c['f1_curve'][i],-i))
        label=build_label(make_question(c),params);labels.append(label)
    split_ids=qids[:];random.Random(cfg['seed']).shuffle(split_ids);splits={'seed':cfg['seed'],'policy':'shuffle all sample IDs with random.Random(seed), independent of labels; 70/10/20','train':split_ids[:3500],'validation':split_ids[3500:4000],'test':split_ids[4000:]}
    require(len(set(splits['train'])&set(splits['test']))==0 and len(set(split_ids))==5000,'Invalid split')
    label_map={r['question_id']:r for r in labels};splits['counts']={s:{'all':len(splits[s]),'valid':sum(label_map[q]['label_status']=='valid' for q in splits[s])} for s in ('train','validation','test')}
    write_rows(out/'queries.jsonl',[{'question_id':qid,'question_text':index[qid]['question_text']} for qid in qids]);write_rows(out/'curves.jsonl',curves.values());write_rows(out/'labels.jsonl',labels);write_json(out/'splits.json',splits)
    current=summary(labels);distribution={'train5000':current,'historical_dev1000':reference,'caveat':'Different populations: natural Train5000 vs balanced 500 bridge + 500 comparison Dev1000. Not a controlled causal comparison.','anomalies':dict(anomalies),'anomalies_by_k':per_k,'api_usage_policy':'Unavailable API usage remains unavailable; costs use only independently tokenized actual reader context.','unavailable_supporting_facts':unknown_items}
    write_json(HERE/'reports/label_distribution.json',distribution)
    metadata={'configuration':cfg,'source_input_sha256':sha256(source/'data/train_5000.jsonl'),'official_train_sha256':sha256(raw_path),'source_outputs':sources,'built_questions':len(labels),'built_qid_k_pairs':len(labels)*15,'reference_reproduction_passed':True,'seconds':time.time()-start,'artifacts_sha256':{p.name:sha256(p) for p in out.iterdir() if p.suffix in ('.json','.jsonl')},'code_sha256':{p.name:sha256(p) for p in (HERE/'src').glob('*.py')}}
    write_json(out/'metadata.json',metadata);print('BUILD_COMPLETE',json.dumps(current),flush=True)
if __name__=='__main__':main()
