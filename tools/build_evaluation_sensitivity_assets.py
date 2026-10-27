"""Display completed AP/threshold/pair sensitivity and a PELT+VAR adapter.

Every operating point remains in the numerical outputs. This presentation
selects fixed conditions to explain definitions; it neither fits nor selects
a best threshold, residual weight or method configuration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_reviewer_controls_assets import one, fmt, interval, tabular, point, local_conditions, short_condition

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/information_sciences")
DEST = BASE / "evaluation_assets"
SOURCES = (
    "boundary_sensitivity/ap_summary.csv", "boundary_sensitivity/ap_paired_vs_raw.csv",
    "boundary_sensitivity/threshold_summary.csv", "boundary_sensitivity/threshold_paired_vs_raw.csv",
    "boundary_sensitivity/residual_summary.csv", "boundary_sensitivity/residual_paired_vs_hard.csv",
    "boundary_sensitivity/diagnostics.json", "segmentation_validation/summary.csv",
    "segmentation_validation/contrasts.csv", "segmentation_validation/diagnostics.json",
)
FIGURES = ("fig_boundary_sensitivity", "fig_threshold_curves", "fig_residual_pairs")
TABLES = ("tab_external_framework", "tab_ap_sensitivity", "tab_residual_pairs", "tab_external_contrasts")
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#007F60", "#8B4D9E"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root=ROOT):
    return {name: json.loads((root / BASE / name).read_text(encoding="utf-8")) if name.endswith(".json")
            else pd.read_csv(root / BASE / name, float_precision="round_trip") for name in SOURCES}


def condition(study, duration, schedule="fixed_phase", h=300, stride=150):
    cell = f"D{duration}_h{h}_s{stride}" if study == "C" else f"D{duration}_{schedule}"
    return dict(study=study, scorer="lasso", duration=duration, schedule=schedule, cell=cell)


def central_conditions():
    return [(a,b,c,h,s) for a,b,c,_,h,s in local_conditions() if a != "C" or (h,s)==(300,150)]


def external_table(data):
    source=data["segmentation_validation/summary.csv"]; rows=[]
    for d in (300,600):
        for method,label in (("raw_window","Raw"),("forced_hard","Hard"),("pelt_var_adapter","PELT + VAR")):
            values=[one(source,duration=d,method=method,metric=m)["mean"] for m in
                    ("pooled_temporal_auroc","temporal_delta_auroc","boundary_standard_ap",
                     "boundary_slot_f1_raw_q95","boundary_slot_f1_method_q95")]
            rows.append(f"{d} & {label} & "+" & ".join(fmt(v) for v in values))
    return tabular(r"$D$ & Method & Pooled & State & AP$_{\rm std}$ & F1$_{\rm raw}$ & F1$_{\rm self}$",rows,"rlrrrrr")


def ap_table(data):
    ap=data["boundary_sensitivity/ap_summary.csv"]; paired=data["boundary_sensitivity/ap_paired_vs_raw.csv"]
    threshold=data["boundary_sensitivity/threshold_paired_vs_raw.csv"]; rows=[]
    for a,d,s,_,h,stride in local_conditions():
        where=condition(a,d,s,h,stride)
        raw=one(ap,**where,method="raw_window",metric="boundary_standard_ap")
        values=[interval(one(paired,**where,method="forced_hard",metric=m)) for m in
                ("boundary_positive_change_ap","boundary_standard_ap")]
        values += [interval(one(threshold,**where,method="forced_hard",metric="boundary_slot_f1",q=.95,threshold_mode=mode))
                   for mode in ("raw_shared","method_self")]
        rows.append(short_condition(a,d,s,h,stride)+f" & {int(raw.n_finite)} & {fmt(raw['mean'])} & "+" & ".join(values))
    return tabular(r"Condition & $n_{\rm AP}$ & Raw AP$_{\rm std}$ & $\Delta$AP$_+$ & $\Delta$AP$_{\rm std}$ & $\Delta$F1$_{\rm raw}$ & $\Delta$F1$_{\rm self}$",rows,"lrrrrrr")


def residual_table(data):
    source=data["boundary_sensitivity/residual_summary.csv"]; rows=[]
    for a in ("A","C"):
        for d in (300,600):
            where=condition(a,d)
            for lam in (1e-8,.25,.5,1.):
                method=f"residual_{lam:g}"
                get=lambda metric:one(source,**where,method=method,**{"lambda":lam},metric=metric)
                weight=r"$10^{-8}$" if lam==1e-8 else f"{lam:g}"
                values=[fmt(get(m)["mean"]) for m in ("hard_state_auroc","residual_state_auroc")]
                values += [interval(get(m)) for m in ("within_auc_delta","across_auc_delta","total_auc_delta")]
                rows.append(f"{a}, {d} & {weight} & "+" & ".join(values))
    return tabular(r"Study, $D$ & $\lambda$ & Hard & Residual & Within $\Delta$ & Cross $\Delta$ & Total $\Delta$",rows,"llrrrrr")


def external_contrasts(data):
    source=data["segmentation_validation/contrasts.csv"];rows=[]
    metrics=(("pooled_temporal_auroc","Pooled"),("temporal_delta_auroc","State"),
             ("boundary_positive_change_ap",r"AP$_+$"),("boundary_standard_ap",r"AP$_{\rm std}$"),
             ("boundary_slot_f1_raw_q95",r"F1$_{\rm raw}$"),("boundary_slot_f1_method_q95",r"F1$_{\rm self}$"),
             ("stable_false_change_rate_method_q95","Stable exceedance"))
    for d in (300,600):
        for metric,label in metrics:
            cells=[one(source,duration=d,method=m,metric=metric,reference="raw_window") for m in ("forced_hard","pelt_var_adapter")]
            rows.append(f"{d} & {label} & "+" & ".join(interval(row) for row in cells)+f" & {int(cells[-1].n_paired)}")
    return tabular(r"$D$ & Target & Hard minus raw & PELT + VAR minus raw & $n$",rows,"rlrrr")


def tables(data):
    return {"tab_external_framework":external_table(data),"tab_ap_sensitivity":ap_table(data),
            "tab_residual_pairs":residual_table(data),"tab_external_contrasts":external_contrasts(data)}


def boundary_figure(data):
    ap=data["boundary_sensitivity/ap_paired_vs_raw.csv"]; threshold=data["boundary_sensitivity/threshold_paired_vs_raw.csv"]
    conditions=central_conditions(); fig,axes=plt.subplots(1,2,figsize=(8.5,5.8),sharey=True,layout="constrained")
    for index,(a,d,s,h,stride) in enumerate(conditions):
        where=condition(a,d,s,h,stride)
        for metric,shift,color,marker,label in (("boundary_positive_change_ap",-.13,BLUE,"o","Positive-change AP"),
                ("boundary_standard_ap",.13,ORANGE,"s","Standard AP")):
            row=one(ap,**where,method="forced_hard",metric=metric)
            point(axes[0],row['mean'],index+shift,row.ci95_low,row.ci95_high,color,marker,label if index==0 else None)
        for mode,shift,color,marker,label in (("raw_shared",-.13,BLUE,"o","Raw q95"),("method_self",.13,ORANGE,"s","Method q95")):
            row=one(threshold,**where,method="forced_hard",metric="boundary_slot_f1",q=.95,threshold_mode=mode)
            point(axes[1],row['mean'],index+shift,row.ci95_low,row.ci95_high,color,marker,label if index==0 else None)
    axes[0].set(yticks=range(len(conditions)),yticklabels=[short_condition(*row) for row in conditions],title="AP zero-group convention")
    axes[1].set(title="F1 amplitude reference")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0,color="#777777",lw=.8);ax.set_xlabel("Hard minus raw on the same slots")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),fontsize=8,frameon=False)
    return fig


def threshold_figure(data):
    source=data["boundary_sensitivity/threshold_summary.csv"]
    fig,axes=plt.subplots(2,2,figsize=(8,5.8),sharex=True,sharey=True,layout="constrained")
    for column,study in enumerate(("B","C")):
        for row,mode in enumerate(("raw_shared","method_self")):
            ax=axes[row,column];where=condition(study,600)
            for method,label,color,marker in (("raw_window","Raw",BLUE,"o"),("forced_hard","Hard",ORANGE,"s"),
                    ("residual_hard_l050","Residual 0.5",GREEN,"^")):
                frame=source
                for key,value in {**where,"method":method,"threshold_mode":mode,"metric":"boundary_slot_f1"}.items():frame=frame[frame[key]==value]
                frame=frame.sort_values("q")
                ax.plot(frame.q,frame['mean'],color=color,marker=marker,ms=4,label=label,lw=1)
                ax.fill_between(frame.q,frame.ci95_low,frame.ci95_high,color=color,alpha=.12)
            ax.set_title(f"{study}, D=600; "+("raw reference" if mode=="raw_shared" else "method reference"))
            ax.spines[["top","right"]].set_visible(False);ax.set_xticks([.8,.9,.95,.99])
    for ax in axes[1]:ax.set_xlabel("Fixed quantile q")
    for ax in axes[:,0]:ax.set_ylabel("Boundary-slot F1")
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False,fontsize=8)
    return fig


def residual_figure(data):
    source=data["boundary_sensitivity/residual_summary.csv"]
    conditions=[(a,d) for a in ("A","C") for d in (300,600)]
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),sharey=True,layout="constrained")
    for ax,lam in zip(axes,(1e-8,.5)):
        for index,(a,d) in enumerate(conditions):
            for metric,shift,color,marker,label in (("within_auc_delta",-.18,BLUE,"o","Within group"),
                    ("across_auc_delta",0,ORANGE,"s","Cross group"),("total_auc_delta",.18,GREEN,"^","Total")):
                row=one(source,**condition(a,d),method=f"residual_{lam:g}",**{"lambda":lam},metric=metric)
                point(ax,row['mean'],index+shift,row.ci95_low,row.ci95_high,color,marker,label if index==0 else None)
        ax.axvline(0,color="#777777",lw=.8);ax.set_xlabel("Residual minus hard AUROC credit")
        ax.set_title("Diagnostic weight 1e-8" if lam==1e-8 else "Weight 0.5")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set(yticks=range(len(conditions)),yticklabels=[f"{a}, D={d}" for a,d in conditions]);axes[0].invert_yaxis()
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False,fontsize=8)
    return fig


def build(root=ROOT):
    root=Path(root);data=load(root);output=root/DEST;output.mkdir(parents=True,exist_ok=True)
    generated=[]
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}):
        for name,function in zip(FIGURES,(boundary_figure,threshold_figure,residual_figure)):
            fig=function(data)
            fig.savefig(output/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None,'Author':None,'Creator':None,'Producer':None})
            fig.savefig(output/f'{name}.png',dpi=600);plt.close(fig);generated.extend([name+'.pdf',name+'.png'])
    for name,value in tables(data).items():
        (output/f'{name}.tex').write_text(value,encoding='utf-8',newline='\n');generated.append(name+'.tex')
    frames=[]
    for name,value in data.items():
        if isinstance(value,pd.DataFrame):frames.append(value.assign(source_table=name))
    pd.concat(frames,ignore_index=True).to_csv(output/'display_source.csv',index=False);generated.append('display_source.csv')
    sources={str(BASE/name):sha(root/BASE/name) for name in SOURCES}
    sources.update({name:sha(root/name) for name in ('tools/build_evaluation_sensitivity_assets.py','tools/build_reviewer_controls_assets.py')})
    prose=tuple(p for p in output.glob('results_*.tex'))
    sources.update({p.relative_to(root).as_posix():sha(p) for p in prose})
    manifest={'source_sha256':sources,'output_sha256':{name:sha(output/name) for name in generated},
              'figures':list(FIGURES),'tables':list(TABLES),'new_fits':0,'new_simulations':0,
              'uncertainty':'95% pointwise paired-repeat percentile bootstrap; fixed grid; no best-q selection',
              'scope':'Retained metric sensitivity, diagnostic residual pairs and a separately fitted PELT+VAR adapter'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return {'outputs':len(generated),'figures':len(FIGURES),'tables':len(TABLES),'new_fits':0}


def verify_assets(root=ROOT):
    root=Path(root);output=root/DEST;manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for prefix,mapping in ((root,manifest['source_sha256']),(output,manifest['output_sha256'])):
        for name,digest in mapping.items():
            path=(prefix/name).resolve()
            if not path.is_relative_to(prefix.resolve()) or sha(path)!=digest:raise ValueError('Sensitivity display checksum mismatch: '+name)
    if manifest['figures']!=list(FIGURES) or manifest['tables']!=list(TABLES) or manifest['new_fits']!=0:raise ValueError('Sensitivity display inventory drift')
    data=load(root)
    for name,value in tables(data).items():
        if (output/f'{name}.tex').read_text(encoding='utf-8')!=value:raise ValueError('Sensitivity table reconstruction differs: '+name)
    expected=[]
    for name,value in data.items():
        if isinstance(value,pd.DataFrame):expected.append(value.assign(source_table=name))
    pd.testing.assert_frame_equal(pd.read_csv(output/'display_source.csv',float_precision='round_trip',low_memory=False),pd.concat(expected,ignore_index=True),check_dtype=False,check_exact=False,atol=1e-14,rtol=1e-12)
    return {'status':'passed','figures':len(FIGURES),'tables':len(TABLES),'new_fits':0,'source_files':len(manifest['source_sha256'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    print(json.dumps(verify_assets() if args.verify else build()))
