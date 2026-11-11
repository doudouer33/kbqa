"""Offline, same-test-set comparison. Never tunes parameters on test labels.
Soft CE/KL use valid labels; answer/cost metrics cover ALL test IDs and a
separate common valid-label subset. Mean baseline is fitted only on train.
Expected policy metrics and deployed argmax metrics are reported separately.
"""
import math,time
import numpy as np
from common import *


def normalize_rows(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    require(p.ndim==2 and p.shape[1]==15 and np.isfinite(p).all() and (p>=0).all(),'Invalid probability matrix')
    require(np.allclose(p.sum(1),1,atol=1e-6,rtol=0),'Probabilities do not sum to one')
    return p/p.sum(1,keepdims=True)


def ce_values(target,p):
    # For deterministic fixed-k policies, exact CE can be infinite. A separately
    # named epsilon-floor score is supplied, without pretending it is exact CE.
    bad=((target>0)&(p==0)).any(axis=1)
    ce=-(target*np.log(np.maximum(p,1e-12))).sum(1)
    entropy=-(target*np.log(np.maximum(target,1e-300))).sum(1)
    return ce,ce-entropy,bad


def policy_metrics(p,f1,tokens,evidence,targets,valid):
    p=normalize_rows(p);n=len(p);idx=p.argmax(1);ar=np.arange(n);ks=np.arange(1,16)
    ce,kl,infinite=ce_values(targets[valid],p[valid])
    selected=f1[ar,idx];selected_tokens=tokens[ar,idx]
    return {'count':n,'valid_label_count':int(valid.sum()),'soft_ce':None if infinite.any() else float(ce.mean()),'soft_kl':None if infinite.any() else float(kl.mean()),'infinite_soft_ce_count':int(infinite.sum()),'soft_ce_epsilon_floor_1e12':float(ce.mean()),'soft_kl_epsilon_floor_1e12':float(kl.mean()),'argmax_answer_f1':float(selected.mean()),'argmax_context_tokens':float(selected_tokens.mean()),'argmax_evidence_recall_lower':float(evidence[ar,idx].mean()),'expected_answer_f1':float((p*f1).sum(1).mean()),'expected_context_tokens':float((p*tokens).sum(1).mean()),'expected_evidence_recall_lower':float((p*evidence).sum(1).mean()),'argmax_normalized_utility':float((selected-.1*selected_tokens/tokens.max(1)).mean()),'expected_normalized_utility':float((p*(f1-.1*tokens/tokens.max(1,keepdims=True))).sum(1).mean()),'mean_argmax_k':float((idx+1).mean()),'mean_expected_k':float((p*ks).sum(1).mean()),'argmax_k_histogram':{str(k):int((idx==k-1).sum()) for k in ks},'mean_probability':p.mean(0).tolist(),'mean_prediction_entropy_nats':float(-(p*np.log(np.maximum(p,1e-300))).sum(1).mean())}


def paired_interval(delta,seed=42,repetitions=1000):
    x=np.asarray(delta);rng=np.random.RandomState(seed)
    means=np.asarray([x[rng.randint(0,len(x),len(x))].mean() for _ in range(repetitions)])
    return {'mean_delta':float(x.mean()),'ci95_low':float(np.percentile(means,2.5)),'ci95_high':float(np.percentile(means,97.5)),'paired_bootstrap_repetitions':repetitions}


def main():
    cfg=config();split=read_json(HERE/'data/splits.json');ids=split['test'];validation_ids=split['validation']
    labels={r['question_id']:r for r in rows(HERE/'data/labels.jsonl')};curves={r['question_id']:r for r in rows(HERE/'data/curves.jsonl')}
    saved=list(rows(HERE/'data/bert_predictions.jsonl'));testpred={r['question_id']:r for r in saved if r['split']=='test'};valpred={r['question_id']:r for r in saved if r['split']=='validation'}
    require(set(testpred)==set(ids) and len(testpred)==len(ids),'Test predictions IDs mismatch');require(set(valpred)==set(validation_ids),'Validation predictions IDs mismatch')
    require(not(set(ids)&set(split['train'])),'Train-test overlap')
    train_valid=[q for q in split['train'] if labels[q]['label_status']=='valid'];mean=np.mean([labels[q]['soft_label'] for q in train_valid],axis=0)
    require(np.allclose(mean,read_json(HERE/'data/train_mean_distribution.json')['probabilities'],atol=1e-12,rtol=0),'Train mean baseline source differs')
    p=normalize_rows([testpred[q]['probabilities'] for q in ids]);meanp=np.tile(mean,(len(ids),1))
    f1=np.array([curves[q]['f1_curve'] for q in ids]);tokens=np.array([curves[q]['token_cost_curve'] for q in ids]);evidence=np.array([curves[q]['evidence_recall_curve'] for q in ids]);valid=np.array([labels[q]['label_status']=='valid' for q in ids]);targets=np.array([labels[q]['soft_label'] if labels[q]['soft_label'] is not None else [0.]*15 for q in ids])
    require(len(ids)==1000 and valid.any(),'Unexpected test population')
    methods={'bert':p,'train_mean_distribution':meanp}
    methods.update({'fixed_k_'+str(k):np.tile(np.eye(15)[k-1],(len(ids),1)) for k in range(1,16)})
    oracle_idx=f1.argmax(1);methods['offline_max_f1_oracle']=np.eye(15)[oracle_idx]
    result={'test_ids_sha256':text_hash('\n'.join(ids)),'checkpoint_selection':read_json(HERE/'checkpoints/best/selection.json'),'test_used_for_selection':False,'train_mean_fit_count':len(train_valid),'evaluation_scope':'Same 1000 test IDs for all methods; exact soft CE/KL only on common valid labels. Fixed-k/oracle exact CE is marked infinite when support is incompatible. Expected metrics correspond to sampling k; argmax metrics correspond to deterministic deployment.','all_test':{},'valid_test':{}}
    for name,prob in methods.items():
        result['all_test'][name]=policy_metrics(prob,f1,tokens,evidence,targets,valid)
        result['valid_test'][name]=policy_metrics(prob[valid],f1[valid],tokens[valid],evidence[valid],targets[valid],np.ones(valid.sum(),dtype=bool))
    result['valid_test']['offline_original_soft_oracle']=policy_metrics(targets[valid],f1[valid],tokens[valid],evidence[valid],targets[valid],np.ones(valid.sum(),dtype=bool))
    # Validation chooses the best fixed-k comparison, without viewing test F1.
    vf1=np.array([curves[q]['f1_curve'] for q in validation_ids]);best_fixed=int(vf1.mean(0).argmax())+1
    result['validation_selected_fixed_k']={'k':best_fixed,'criterion':'highest mean validation answer F1; smallest k tie','validation_f1':float(vf1[:,best_fixed-1].mean()),'test_metrics':result['all_test']['fixed_k_'+str(best_fixed)]}
    ce_b,_,_=ce_values(targets[valid],p[valid]);ce_m,_,_=ce_values(targets[valid],meanp[valid]);ar=np.arange(len(ids));ib=p.argmax(1);im=meanp.argmax(1)
    result['bert_minus_train_mean']={'soft_ce':paired_interval(ce_b-ce_m),'argmax_answer_f1':paired_interval(f1[ar,ib]-f1[ar,im]),'argmax_context_tokens':paired_interval(tokens[ar,ib]-tokens[ar,im]),'expected_answer_f1':paired_interval((p*f1).sum(1)-(meanp*f1).sum(1)),'expected_context_tokens':paired_interval((p*tokens).sum(1)-(meanp*tokens).sum(1))}
    rng=np.random.RandomState(cfg['seed']);permuted_ce=[];pv=p[valid];tv=targets[valid]
    for _ in range(200):permuted_ce.append(float(ce_values(tv,pv[rng.permutation(len(pv))])[0].mean()))
    midpoint=(p+meanp)/2;js=.5*(p*np.log(np.maximum(p,1e-300)/np.maximum(midpoint,1e-300))).sum(1)+.5*(meanp*np.log(np.maximum(meanp,1e-300)/np.maximum(midpoint,1e-300))).sum(1)
    result['query_specificity']={'distinct_argmax_k_count':int(len(set(ib.tolist()))),'mean_per_class_probability_std':float(p.std(0).mean()),'mean_js_to_train_mean':float(js.mean()),'aligned_soft_ce':float(ce_b.mean()),'shuffled_predictions_soft_ce_mean':float(np.mean(permuted_ce)),'aligned_minus_shuffled_soft_ce':float(ce_b.mean()-np.mean(permuted_ce)),'permutation_p_lower_ce':float((1+sum(x<=ce_b.mean() for x in permuted_ce))/(len(permuted_ce)+1)),'permutations':len(permuted_ce),'interpretation':'Variation alone is insufficient; aligned CE below shuffled CE checks whether variation is query-specific rather than noise.'}
    # Hold the predicted argmax-k histogram fixed, then break question/decision
    # alignment. This is a post-training diagnostic, not a tunable baseline.
    shuffled_f1=[];shuffled_tokens=[];decision_rng=np.random.RandomState(cfg['seed'])
    for _ in range(200):
        shuffled_idx=ib[decision_rng.permutation(len(ib))]
        shuffled_f1.append(float(f1[ar,shuffled_idx].mean()));shuffled_tokens.append(float(tokens[ar,shuffled_idx].mean()))
    marginal=np.bincount(ib,minlength=15)/len(ib)
    result['query_specificity'].update({'argmax_k_histogram_preserving_shuffle_f1_mean':float(np.mean(shuffled_f1)),
        'argmax_k_histogram_preserving_shuffle_tokens_mean':float(np.mean(shuffled_tokens)),
        'aligned_argmax_f1_minus_shuffled':float(f1[ar,ib].mean()-np.mean(shuffled_f1)),
        'permutation_p_greater_argmax_f1':float((1+sum(x>=f1[ar,ib].mean() for x in shuffled_f1))/201),
        'same_argmax_histogram_independent_policy_expected_f1':float((f1*marginal).sum(1).mean()),
        'same_argmax_histogram_independent_policy_expected_tokens':float((tokens*marginal).sum(1).mean())})
    valvalid=np.array([labels[q]['label_status']=='valid' for q in validation_ids]);vt=np.array([labels[q]['soft_label'] if labels[q]['soft_label'] is not None else [0.]*15 for q in validation_ids]);vp=normalize_rows([valpred[q]['probabilities'] for q in validation_ids]);vmean=np.tile(mean,(len(validation_ids),1))
    result['validation_soft_ce']={'bert':float(ce_values(vt[valvalid],vp[valvalid])[0].mean()),'train_mean':float(ce_values(vt[valvalid],vmean[valvalid])[0].mean())}
    write_json(HERE/'reports/evaluation.json',result)
    decisions=[]
    for i,q in enumerate(ids):decisions.append({'question_id':q,'label_status':labels[q]['label_status'],'bert_argmax_k':int(ib[i]+1),'train_mean_argmax_k':int(im[i]+1),'bert_f1':float(f1[i,ib[i]]),'train_mean_f1':float(f1[i,im[i]]),'bert_context_tokens':int(tokens[i,ib[i]]),'train_mean_context_tokens':int(tokens[i,im[i]])})
    write_rows(HERE/'data/test_decisions.jsonl',decisions)
    make_figures(result)
    write_report(result,cfg)
    print('EVALUATION_COMPLETE',json.dumps({'bert':result['all_test']['bert'],'mean':result['all_test']['train_mean_distribution'],'query_specificity':result['query_specificity']}),flush=True)


def make_figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));fixed=[result['all_test']['fixed_k_'+str(k)] for k in range(1,16)];ax.plot([r['argmax_context_tokens'] for r in fixed],[r['argmax_answer_f1'] for r in fixed],'.-',alpha=.6,label='Fixed k=1..15')
    for name,label in [('bert','BERT'),('train_mean_distribution','Train mean'),('offline_max_f1_oracle','Offline max-F1 oracle')]:
        r=result['all_test'][name];ax.scatter(r['argmax_context_tokens'],r['argmax_answer_f1'],s=80,label=label+' (argmax)')
        if name!='offline_max_f1_oracle':ax.scatter(r['expected_context_tokens'],r['expected_answer_f1'],marker='x',s=80,label=label+' (expected)')
    ax.set_xlabel('Mean actual context tokens');ax.set_ylabel('Official answer F1');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(HERE/'reports/f1_token_tradeoff.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));xs=np.arange(1,16)
    for name,label in [('bert','BERT test mean probability'),('train_mean_distribution','Train mean probability')]:ax.plot(xs,result['all_test'][name]['mean_probability'],'.-',label=label)
    ax.set_xticks(xs);ax.set_xlabel('k');ax.set_ylabel('Probability');ax.legend();fig.tight_layout();fig.savefig(HERE/'reports/predicted_k_distribution.png',dpi=180);plt.close(fig)


def write_report(result,cfg):
    dist=read_json(HERE/'reports/label_distribution.json');training=read_json(HERE/'reports/training_summary.json');gpu=read_json(HERE/'reports/gpu_preflight.json');split=read_json(HERE/'data/splits.json');b=result['all_test']['bert'];m=result['all_test']['train_mean_distribution'];d=result['bert_minus_train_mean'];qs=result['query_specificity']
    better=b['soft_ce']<m['soft_ce'];dominates=b['argmax_answer_f1']>=m['argmax_answer_f1'] and b['argmax_context_tokens']<=m['argmax_context_tokens']
    lines=['# HotpotQA Train5000: BERT 动态 Top-K 实验结果','',f"完整构造 5000×15 曲线；Original Soft Label：Scheme D，ε=0.1，τ=0.5，λ=0.1，T=0.1。",'成本仅为实际检索上下文的 Qwen tokenizer tokens，Evidence 为原有句子级保守 lower bound。两条越界支持句按用户确认记 unknown、保留原始分母。','',f"训练设备：{gpu['gpu_name']}，UUID `{gpu['gpu_uuid']}`。全参数 BERT，输入仅 question text。",f"训练 {cfg['training']['epochs']} epoch；最佳 epoch={training['best']['epoch']}，仅依据 Validation Soft CE 选择。",f"峰值模型分配显存：{training['final_memory']['peak_allocated_mib']:.1f} MiB。",'','## 标签分布','', '| 数据 | valid | all_zero_f1 | empty_candidates | 平均熵（nats） |','|---|---:|---:|---:|---:|']
    for name,st in [('Train5000',dist['train5000']),('历史 Dev1000',dist['historical_dev1000']['summary'])]:lines.append('| {} | {} | {} | {} | {:.4f} |'.format(name,st['statuses']['valid']['count'],st['statuses']['all_zero_f1']['count'],st['statuses']['empty_candidates']['count'],st['mean_entropy_nats']))
    lines+=['','历史 Dev1000 为 bridge/comparison 各500，当前为自然抽样Train，两者分布差异不能视为受控比较。',f"数据划分及有效标签数量：`{json.dumps(split['counts'])}`。无效标签维持原状态，不参加 Soft CE 训练；所有测试题均参加 Answer F1/成本评价。",f"异常计数（75000次输出）：`{json.dumps(dist['anomalies'])}`。",'','## 相同测试集的结果','', '| 方法 | Soft CE（有效标签） | KL（有效标签） | argmax F1 | argmax tokens | expected F1 | expected tokens |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ['bert','train_mean_distribution']+['fixed_k_'+str(k) for k in range(1,16)]+['offline_max_f1_oracle']:
        r=result['all_test'][name];ce='∞' if r['soft_ce'] is None else '{:.5f}'.format(r['soft_ce']);kl='∞' if r['soft_kl'] is None else '{:.5f}'.format(r['soft_kl']);lines.append('| {} | {} | {} | {:.5f} | {:.1f} | {:.5f} | {:.1f} |'.format(name,ce,kl,r['argmax_answer_f1'],r['argmax_context_tokens'],r['expected_answer_f1'],r['expected_context_tokens']))
    lines+=['','Fixed-k 的确定性分布与多数 Soft Label 支持集不兼容，因此精确 CE/KL 为无穷；JSON另列有明确标记的1e-12截断分数。',f"Validation 选择的 fixed-k 为 {result['validation_selected_fixed_k']['k']}；全部 fixed-k 测试结果仅作描述，不使用Test选参。",'','## BERT 与 Train Mean','',f"BERT 的测试 Soft CE {'低于' if better else '未低于'} Train Mean；差值 BERT−Mean={d['soft_ce']['mean_delta']:.5f}，paired bootstrap 95% CI=[{d['soft_ce']['ci95_low']:.5f}, {d['soft_ce']['ci95_high']:.5f}]。",f"确定性策略 F1 差值={d['argmax_answer_f1']['mean_delta']:.5f}；平均 tokens 差值={d['argmax_context_tokens']['mean_delta']:.1f}。",f"F1与成本同时不劣于Mean：{dominates}。",f"BERT argmax 选中了 {qs['distinct_argmax_k_count']} 种 k，逐类预测概率平均标准差={qs['mean_per_class_probability_std']:.6f}。",f"打乱题目与BERT预测对应关系后平均CE={qs['shuffled_predictions_soft_ce_mean']:.5f}，原对应CE={qs['aligned_soft_ce']:.5f}；置换检验 p={qs['permutation_p_lower_ce']:.5f}。",'只有预测变化与题目对应后改善损失，才能支持有效query-specific学习；仅概率变化或argmax变化不足以得出结论。','本实验仅单个固定划分和随机种子，bootstrap区间反映测试样本波动，不涵盖训练随机性。','','![F1 token trade-off](f1_token_tradeoff.png)','','![Predicted k distribution](predicted_k_distribution.png)','']
    lines+=['', '## 固定决策分布后的动态收益', '',
        f"将BERT的argmax-k分布保持不变，只打乱题目与决策对应关系：平均F1={qs['argmax_k_histogram_preserving_shuffle_f1_mean']:.5f}，tokens={qs['argmax_k_histogram_preserving_shuffle_tokens_mean']:.1f}。",
        f"BERT原对应F1相对打乱的增量={qs['aligned_argmax_f1_minus_shuffled']:.5f}，置换检验p={qs['permutation_p_greater_argmax_f1']:.5f}。这是训练后的描述性诊断，不参与模型或超参数选择。",
        '', '## 结论', '',
        '本次BERT没有在主要Soft CE/KL目标上优于训练集Mean Distribution。argmax F1上升同时增加了上下文成本，因此不能据此宣称整体优于Mean。',
        f"BERT的测试argmax-k计数：`{json.dumps(b['argmax_k_histogram'])}`。",
        '训练损失继续下降，但第2轮后Validation Loss上升；按预先规则采用第2轮，不使用测试集重新选模型。',
        '观察到少量query-dependent概率变化；部署决策的实际收益应结合上述固定决策分布的诊断判断，不能仅凭不同问题选出不同k作结论。', '']
    (HERE/'reports/report.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
