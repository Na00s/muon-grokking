"""Descriptive feature/classifier mean trajectories for the NFI compatibility test."""
import csv
import json
import torch
from diagnostics import WORK,configure_runtime,timecourse_paths,make_arm,original


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float((a*b).sum())/denominator if denominator else None


def feature_summary(h,y,head_mean):
    mean=h.mean(0);centered=h-mean
    classes=torch.unique(y)
    class_means=torch.stack([h[y==label].mean(0) for label in classes])
    class_balanced=class_means.mean(0)
    matched_means=class_means[y]
    within=float((h-matched_means).square().sum())
    between=float((matched_means-mean).square().sum())
    return dict(mean_norm=float(mean.norm()),rms_feature_norm=float(h.square().sum(1).mean().sqrt()),
                centered_rms_feature_norm=float(centered.square().sum(1).mean().sqrt()),
                global_mean_power_fraction=float(mean.square().sum()/h.square().sum(1).mean()),
                classifier_feature_mean_cosine=cosine(head_mean,mean),
                class_balanced_mean_norm=float(class_balanced.norm()),
                classifier_balanced_feature_mean_cosine=cosine(head_mean,class_balanced),
                within_class_centered_energy=within,between_class_centered_energy=between,
                within_to_between_class_energy=within/between if between else None)


@torch.no_grad()
def run(seed,out):
    x,y,tx,ty=original.generate_modular_addition_data(seed=seed)
    all_x=torch.cat([x,tx]);all_y=torch.cat([y,ty]);rows=[]
    for path in timecourse_paths(seed):
        state=torch.load(path,map_location='cpu',weights_only=False)
        model,_,_=make_arm(state,'original32')
        h=original.residuals(model,all_x).double();w=model.unembedding.weight.double()
        mean=w.mean(0)
        rows.append(dict(seed=seed,step=int(state['step']),checkpoint=str(path),
                         head_norm=float(w.norm()),head_class_mean_norm=float(mean.norm()),
                         head_centered_norm=float((w-mean).norm()),
                         head_class_common_power_fraction=float((mean.square().sum()*len(w))/w.square().sum()),
                         training=feature_summary(h[:len(y)],y,mean),
                         full_grid=feature_summary(h,all_y,mean)))
    out.write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__=='__main__':
    configure_runtime(1);all_rows=[]
    for seed in range(5):
        rows=run(seed,WORK/f'causal_study/diagnostics_results/seed{seed}/mean_drift.json')
        all_rows.extend(rows);print('completed',seed,flush=True)
    flat=[]
    for r in all_rows:
        f={k:v for k,v in r.items() if k not in ['training','full_grid']}
        for label in ['training','full_grid']:
            for k,v in r[label].items(): f[label+'_'+k]=v
        flat.append(f)
    with (WORK/'causal_study/diagnostics_results/mean_drift.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
