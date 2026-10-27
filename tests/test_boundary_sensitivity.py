"""AP zero groups, scale references and exact residual-pair regression fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_boundary_sensitivity as b


@pytest.mark.parametrize("truth,strength", [([1,0,1,0],[1,0,0,0]), ([1,0,1,0],[2,2,1,0]),
                                           ([1,0,0,0],[0,0,0,0]), ([1,0,1,0],[.5,.4,.3,.2])])
def test_standard_AP_matches_noninterpolated_AP_and_exact_zero_group_identity(truth, strength):
    result = b.average_precision_components(truth, strength)
    assert result["standard_ap"] == pytest.approx(average_precision_score(truth, strength))
    assert result["standard_ap"] == pytest.approx(result["positive_ap"] + result["prevalence"] * (1-result["positive_recall"]))
    assert result["identity_gap"] < 1e-15


def test_allzero_AP_has_prevalence_but_positive_AP_abstains():
    result = b.average_precision_components([1,0,0,0], [0,0,0,0])
    assert result["standard_ap"] == result["zero_increment"] == .25
    assert result["positive_ap"] == result["positive_recall"] == 0


@pytest.mark.parametrize("truth", [[0,0,0], [1,1,1]])
def test_no_positive_or_no_negative_target_is_missing_under_both_AP_conventions(truth):
    assert all(np.isnan(value) for value in b.average_precision_components(truth,[0,1,2]).values())


def boundary_case():
    support = np.zeros((2,1,2,2),int);support[1,0,0,1]=1
    regimes = np.r_[np.zeros(2,int),np.ones(8,int)]
    bounds = np.array([[n,n+2] for n in range(0,10,2)])
    raw = np.zeros((5,1,2,2));raw[:,0,0,1]=[0,1,1.2,1.3,1.3]
    return raw,bounds,regimes,support


def test_user_fixed_threshold_counterexample_and_method_quantile_scale_invariance():
    truth=np.array([1,0,0,0],bool);strength=np.array([1,.2,.1,0])
    f1=lambda detected: 2*np.sum(detected&truth)/(2*np.sum(detected&truth)+np.sum(detected&~truth)+np.sum(~detected&truth))
    assert f1(strength>.5)==1 and f1(.1*strength>.5)==0
    assert b.average_precision_components(truth,strength)["standard_ap"]==b.average_precision_components(truth,.1*strength)["standard_ap"]
    raw,bounds,regimes,support=boundary_case()
    rows=b.evaluate_methods({"raw":raw,"scaled":.1*raw},raw,bounds,regimes,support)
    assert rows[0]["boundary_slot_f1_raw_q95"]==1
    assert rows[1]["boundary_slot_f1_raw_q95"]==0
    assert rows[0]["boundary_slot_f1_method_q95"]==rows[1]["boundary_slot_f1_method_q95"]==1
    assert rows[1]["method_change_threshold_q95"]==pytest.approx(.1*rows[0]["method_change_threshold_q95"])


def test_every_quantile_and_reference_is_retained_and_constant_self_threshold_emits_nothing():
    raw,bounds,regimes,support=boundary_case()
    ap,threshold,arrays=b.evaluate_scores({"raw":raw,"constant":np.zeros_like(raw)},raw,bounds,regimes,support)
    assert len(ap)==2 and len(threshold)==2*len(b.QUANTILES)*2
    constant=[r for r in threshold if r["method"]=="constant" and r["threshold_mode"]=="method_self"]
    assert all(row["threshold"]==0 and row["boundary_tp"]==0 and row["boundary_slot_f1"]==0 for row in constant)
    assert arrays["standard_ap"][1,0]==pytest.approx(.25)


def residual_case():
    raw=np.zeros((4,1,2,2));raw[:,0,0,1]=[.1,.8,.2,.9]
    hard=np.zeros_like(raw);hard[:,0,0,1]=[.45,.45,.55,.55]
    truth=np.zeros_like(raw,dtype=bool);truth[:,0,0,1]=[0,1,0,1]
    changing=np.zeros((1,2,2),bool);changing[0,0,1]=True
    return raw,hard,np.array([0,0,1,1]),truth,changing


def test_pair_components_sum_to_residual_minus_hard_with_ties_half_credit():
    raw,hard,labels,truth,changing=residual_case()
    rows,arrays=b.residual_pair_decomposition(raw,hard,labels,truth,changing)
    for row in rows:
        assert row["within_auc_delta"]+row["across_auc_delta"]==pytest.approx(row["residual_state_auroc"]-row["hard_state_auroc"])
        assert row["hard_state_auroc"]==.5
    near=rows[1]
    assert near["lambda"]==1e-8 and near["within_auc_delta"]==.25 and near["across_auc_delta"]==0
    assert near["residual_state_auroc"]==.75 and near["within_lost_ties"]==2
    assert rows[-1]["within_auc_delta"]==rows[-1]["across_auc_delta"]==.25


def test_pair_credit_is_normalized_per_edge_before_macro_not_pooled_over_pairs():
    raw,hard,labels,truth,changing=residual_case()
    raw[:,0,1,0]=[.5,.4,.3,.1];hard[:,0,1,0]=[.45,.45,.2,.2]
    truth[:,0,1,0]=[0,0,0,1];changing[0,1,0]=True
    rows,_=b.residual_pair_decomposition(raw,hard,labels,truth,changing,lambdas=(1.,))
    assert rows[0]["total_auc_delta"]==pytest.approx((.5-1/6)/2)
    assert rows[0]["total_auc_delta"]!=pytest.approx((2-.5)/7)


def test_nearzero_finite_precision_diagnostic_retains_failed_tie_breaks():
    raw,hard,labels,truth,changing=residual_case();hard[:,0,0,1]=1e16
    rows,_=b.residual_pair_decomposition(raw,hard,labels,truth,changing,lambdas=(0.,1e-8))
    assert rows[1]["total_auc_delta"]==0
    assert rows[1]["within_raw_order_mismatches"]>0 and rows[1]["within_lost_ties"]==0
    hard[1,0,0,1]=2e16
    with pytest.raises(ValueError,match="constant within"):
        b.residual_pair_decomposition(raw,hard,labels,truth,changing)


def test_shared_bootstrap_counts_reuse_repeat_units_and_preserve_missing_denominators():
    cache={};rows=[]
    for seed in range(3):
        for method,value in [("raw_window",.2),("alternative",.4)]:
            rows.append({"study":"fixture","scorer":"lasso","duration":300,"schedule":"fixed","cell":"one",
                         "repeat_seed":seed,"method":method,"score":value})
    absolute,paired=b.summarise_block(pd.DataFrame(rows),["score"],cache=cache)
    result=paired[paired.method=="alternative"].iloc[0]
    assert result.n_paired==3 and result["mean"]==pytest.approx(.2)
    assert result.ci95_low==result.ci95_high==result["mean"]
    frame=pd.DataFrame(rows);frame.loc[(frame.repeat_seed==1)&(frame.method=="alternative"),"score"]=np.nan
    _,paired=b.summarise_block(frame,["score"],cache=cache)
    assert paired[paired.method=="alternative"].iloc[0].n_missing_pairs==1


def test_frozen_roundtrip_rebuild_and_hash_valid_numeric_corruption_rejection(tmp_path,monkeypatch):
    raw,bounds,regimes,support=boundary_case()
    ap,threshold,_=b.evaluate_scores({"raw_window":raw},raw,bounds,regimes,support,quantiles=(.95,))
    identity={"study":"fixture","scorer":"lasso","duration":300,"schedule":"fixed","cell":"one","repeat_seed":1}
    frames={"ap_detail.csv":pd.DataFrame([{**identity,**row}for row in ap]),
            "threshold_detail.csv":pd.DataFrame([{**identity,**row}for row in threshold]),
            "residual_detail.csv":pd.DataFrame([{**identity,"method":"residual_0","lambda":0.,**{k:0. for k in b.PAIR_METRICS}}])}
    frames.update(b.summarize(frames));report={"cases":1,"new_fits":0,"new_simulations":0}
    monkeypatch.setattr(b,"input_snapshot",lambda root:{"fixture":"unchanged"})
    monkeypatch.setattr(b,"protocol",lambda digest,root:{"input_manifest_sha256":digest,"source_sha256":{}})
    monkeypatch.setattr(b,"compute",lambda root,**kwargs:(frames,report))
    output,log=tmp_path/'output',tmp_path/'log'
    b.freeze(tmp_path,output,log);b.run(tmp_path,output,log)
    before=b.directory_hashes(output)
    assert b.verify(tmp_path,output)["read_only"] and before==b.directory_hashes(output)
    b.rebuild(tmp_path,output)
    assert before==b.directory_hashes(output)
    changed=pd.read_csv(output/'ap_detail.csv');changed.loc[0,'boundary_standard_ap']=.1
    changed.to_csv(output/'ap_detail.csv',index=False)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf8'))
    manifest['sha256']['ap_detail.csv']=b.loc.sha(output/'ap_detail.csv')
    (output/'manifest.json').write_bytes(b.loc.json_bytes(manifest))
    with pytest.raises(AssertionError):b.verify(tmp_path,output)
