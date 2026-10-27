#!/usr/bin/env python3
"""Independent mechanism checks for a window-score graph clustering manuscript.

These experiments do NOT reproduce the author's implementation or data.
Run: python mechanism_probes.py --out results --reps 200
Dependencies: numpy, scipy, scikit-learn. CPU only; fixed seeds.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from threadpoolctl import threadpool_limits


def standardized(x: NDArray[np.float64]) -> NDArray[np.float64]:
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=0)
    return xc / np.where(sd > 1e-12, sd, 1.0)


def all_bipartitions(m: int) -> NDArray[np.float64]:
    """All 2^(m-1)-1 nonempty unlabeled bipartitions, with row 0 in group 0."""
    if not 2 <= m <= 12:
        raise ValueError('Exhaustive enumeration is intentionally limited to 2 <= m <= 12.')
    codes = np.arange(1, 2 ** (m - 1), dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(m - 1, dtype=np.uint64)) & 1).astype(float)
    return np.column_stack((np.zeros(len(codes)), bits))


def exact_two_cluster_gain(x: NDArray[np.float64], partitions: NDArray[np.float64]) -> float:
    """Maximize 1 - SSE2/SSE1 by exhaustive bipartition enumeration."""
    x = x - x.mean(axis=0)
    total = float(np.sum(x * x))
    if total <= 0:
        raise ValueError('All observations are identical.')
    n = partitions.sum(axis=1)
    sums = partitions @ x
    between = np.sum(sums * sums, axis=1) * (1.0 / n + 1.0 / (len(x) - n))
    return float(np.max(between) / total)


def gate_probe(reps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, raw = [], []
    conditions = [(5, 168, 'iid_normal'), (7, 168, 'iid_normal'),
                  (9, 168, 'iid_normal'), (7, 168, 'overlap_half'),
                  (7, 168, 'absolute_normal'), (7, 16, 'iid_normal'),
                  (7, 512, 'iid_normal'), (19, 168, 'iid_normal')]
    for m, p, kind in conditions:
        gains, fitted, random_gains = [], [], []
        partitions = all_bipartitions(m) if m <= 12 else None
        for seed in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([7314, m, p, seed]))
            if kind == 'overlap_half':
                a = rng.normal(size=(m + 1, p))
                x = (a[:-1] + a[1:]) / np.sqrt(2)
            elif kind == 'absolute_normal':
                x = np.abs(rng.normal(size=(m, p)))
            else:
                x = rng.normal(size=(m, p))
            x = standardized(x)
            total = float(np.sum(x * x))
            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x)
            gain = 1.0 - float(km.inertia_) / total
            opt = exact_two_cluster_gain(x, partitions) if partitions is not None else None
            order = rng.permutation(m)
            group = x[order[:m//2]]
            ss = np.sum(group, axis=0)
            bg = float((ss @ ss) * (1.0/len(group) + 1.0/(m-len(group))) / total)
            if opt is not None:
                assert opt + 1e-12 >= 1.0 / (m - 1)
                gains.append(opt)
            fitted.append(gain)
            random_gains.append(bg)
            raw.append(dict(m=m, p=p, null=kind, seed=seed, optimal_gain=opt,
                            kmeans_gain=gain, random_balanced_gain=bg))
        a = np.array(gains) if gains else np.array(fitted)
        f = np.array(fitted)
        summary.append(dict(m=m, p=p, null=kind, reps=reps,
                            exact_enumeration=partitions is not None,
                            deterministic_optimum_lower_bound=1.0/(m-1),
                            mean_best_gain=float(a.mean()), min_best_gain=float(a.min()),
                            kmeans_mean_gain=float(f.mean()),
                            kmeans_acceptance_012=float((f > .12).mean()),
                            mean_random_balanced_gain=float(np.mean(random_gains))))
    return summary, raw


def static_metric_probe() -> dict[str, Any]:
    """Exact finite fixture, not an estimate of performance on the manuscript."""
    w, e = 100, 100
    y = np.zeros((w, e), dtype=int)
    y[:, :10] = 1
    y[:50, 10:15] = 1
    y[50:, 15:20] = 1
    p = y.mean(axis=0)
    scores = np.broadcast_to(p, y.shape)
    auc = float(roc_auc_score(y.ravel(), scores.ravel()))
    num = 0.0
    for i in range(e):
        for j in range(e):
            h = float(p[i] > p[j]) + .5 * float(p[i] == p[j])
            num += p[i] * (1.0-p[j]) * h
    formula = num / (p.sum() * (1.0-p).sum())
    changing = np.flatnonzero((p > 0) & (p < 1))
    temporal = float(np.mean([roc_auc_score(y[:,j], scores[:,j]) for j in changing]))
    rng = np.random.default_rng(8427)
    permuted = np.column_stack([rng.permutation(y[:,j]) for j in range(e)])
    perm_auc = float(roc_auc_score(permuted.ravel(), scores.ravel()))
    assert abs(auc-formula) < 1e-12 and abs(auc-perm_auc) < 1e-12
    assert temporal == .5
    return dict(windows=w, candidate_edges=e, always_active=10,
                changing_edges=10, never_active=80, prevalence=float(p.mean()),
                static_pooled_auc=auc, analytic_pooled_auc=formula,
                independently_time_shuffled_pooled_auc=perm_auc,
                static_changing_edge_temporal_auc=temporal,
                true_time_varying_graph_pooled_auc=float(roc_auc_score(y.ravel(),y.ravel())))


def window_probe(reps: int) -> list[dict[str, Any]]:
    """Exogenous-driver switched linear regression with common design covariance.

    Equivalent to lagged drivers X1[t-1], X2[t-1] predicting a third node.
    The coefficients alternate between (.9,0) and (0,.9) every 150 targets.
    IID drivers avoid the covariance-shift complications of a general VAR.
    """
    t, period, stride, sigma = 2400, 150, 75, .5
    states = (np.arange(t)//period) % 2
    beta = np.where(states[:,None] == 0, np.array([.9, 0.]), np.array([0., .9]))
    outputs = []
    for h in [75, 150, 300, 450, 600]:
        starts = np.arange(0, t-h+1, stride)
        centers = starts + h//2
        mix = np.array([(states[s:s+h] == 0).mean() for s in starts])
        pop = np.column_stack((.9*mix, .9*(1-mix)))
        aris, mse = [], []
        for seed in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([9530, seed]))
            x = rng.normal(size=(t, 2))
            y = np.sum(x*beta,axis=1) + rng.normal(scale=sigma,size=t)
            estimates = np.array([np.linalg.lstsq(x[s:s+h], y[s:s+h], rcond=None)[0]
                                  for s in starts])
            labels = KMeans(n_clusters=2,n_init=10,random_state=seed).fit_predict(
                standardized(np.abs(estimates)))
            aris.append(adjusted_rand_score(states[centers],labels))
            mse.append(float(np.mean((estimates-pop)**2)))
        a = np.array(aris)
        outputs.append(dict(window=h, period=period, stride=stride, targets=t,
                            num_windows=len(starts), reps=reps,
                            population_state0_fractions=np.unique(mix).tolist(),
                            mean_purity=float(np.maximum(mix,1-mix).mean()),
                            population_max_pairwise_score_distance=float(np.linalg.norm(pop.max(axis=0)-pop.min(axis=0))),
                            mean_state_ari=float(a.mean()),
                            state_ari_mc_se=float(a.std(ddof=1)/np.sqrt(reps)),
                            mean_ols_mse_vs_window_population=float(np.mean(mse))))
    return outputs


def selection_and_value_probe() -> dict[str, Any]:
    rng = np.random.default_rng(41501)
    n = 200000
    a, b = rng.normal(size=(2, n))
    groups = (a >= 0)
    self_centroids = [float(a[~groups].mean()),float(a[groups].mean())]
    independent_centroids = [float(b[~groups].mean()),float(b[groups].mean())]
    eta = .30
    true = rng.integers(0,2,size=n)
    estimated = np.logical_xor(true, rng.random(n)<eta).astype(int)
    soft = np.where(estimated==1,1-eta,eta)
    static = np.full(n,.5)
    return dict(single_gaussian_true_mean=0., samples=n,
                self_partition_centroids=self_centroids,
                theoretical_self_centroids=[-np.sqrt(2/np.pi),np.sqrt(2/np.pi)],
                independent_view_centroids=independent_centroids,
                symmetric_state_error=eta,
                brier_empirical_static=float(np.mean((static-true)**2)),
                brier_empirical_hard=float(np.mean((estimated-true)**2)),
                brier_empirical_soft=float(np.mean((soft-true)**2)),
                brier_exact_static=.25,brier_exact_hard=eta,brier_exact_soft=eta*(1-eta),
                brier_exact_information_value=(.5-eta)**2)


def pelt_scale_probe(reps: int) -> list[dict[str, Any]]:
    """Exact contiguous SSE segmentation: cost scaling, not PELT implementation.

    Input consists of stationary independent Gaussian features standardized across
    seven rows. Five non-whitened principal components retain the original scale.
    A column-standardized-PC condition isolates scale. It is not a recommended
    change for the manuscript without independent calibration.
    """
    m, p, rank = 7, 168, 5
    partitions = []
    for bits in range(1 << (m-1)):
        ends = [i+1 for i in range(m-1) if bits & (1 << i)] + [m]
        starts = [0] + ends[:-1]
        partitions.append(list(zip(starts,ends)))
    counts = {(variant,factor): [] for variant in ['raw_PC','unit_variance_PC']
              for factor in [.25,1.,4.,16.,32.]}
    variance_scales = []
    for seed in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([6632,seed]))
        x=standardized(rng.normal(size=(m,p)))
        u,sv,_=np.linalg.svd(x,full_matrices=False)
        pcs=u[:,:rank]*sv[:rank]
        variance_scales.append(float(np.mean(pcs**2)))
        for variant,features in [('raw_PC',pcs),('unit_variance_PC',standardized(pcs))]:
            costs=[]
            for segments in partitions:
                costs.append(sum(float(np.sum((features[a:b]-features[a:b].mean(axis=0))**2))
                                 for a,b in segments))
            sizes=np.array([len(part) for part in partitions])
            for factor in [.25,1.,4.,16.,32.]:
                beta=factor*rank*np.log(m)
                best=int(np.argmin(np.array(costs)+beta*sizes))
                counts[(variant,factor)].append(int(sizes[best]))
    return [dict(variant=variant,penalty_factor=factor,base_penalty=float(rank*np.log(m)),
                 reps=reps,mean_segments=float(np.mean(values)),
                 singleton_partition_rate=float(np.mean(np.array(values)==m)),
                 one_segment_rate=float(np.mean(np.array(values)==1)),
                 raw_pc_mean_squared_coordinate=float(np.mean(variance_scales)))
            for (variant,factor),values in counts.items()]


def pooled_ols_probe() -> dict[str, Any]:
    rng = np.random.default_rng(8451)
    true = np.array([1.,-.4])
    xs, ys, qs, estimates = [], [], [], []
    for n, scales in [(40,[1.,.2]),(140,[.3,2.])]:
        x = rng.normal(size=(n,2))*scales
        y = x@true + rng.normal(scale=.5,size=n)
        xs.append(x); ys.append(y); qs.append(x.T@x)
        estimates.append(np.linalg.lstsq(x,y,rcond=None)[0])
    direct = np.linalg.lstsq(np.vstack(xs),np.concatenate(ys),rcond=None)[0]
    info_weighted = np.linalg.solve(sum(qs),sum(q@b for q,b in zip(qs,estimates)))
    arithmetic = np.mean(estimates,axis=0)
    assert np.max(np.abs(direct-info_weighted)) < 1e-10
    return dict(true_coefficient=true.tolist(),
                pooled_fit=direct.tolist(),information_weighted_fit=info_weighted.tolist(),
                arithmetic_average=arithmetic.tolist(),
                exact_identity_error=float(np.max(np.abs(direct-info_weighted))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--reps',type=int,default=200)
    args=parser.parse_args()
    if args.reps < 2: raise ValueError('--reps must be at least two')
    args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        gate,raw=gate_probe(args.reps)
        result=dict(disclaimer='Independent toy probes, not a replication of STCG or original manuscript data.',
                    gate=gate,static_metric=static_metric_probe(),
                    window=window_probe(max(50,args.reps//2)),
                    selection_and_value=selection_and_value_probe(),
                    pooled_ols=pooled_ols_probe(),
                    pelt_scale=pelt_scale_probe(args.reps))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.out/'gate_replicates.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
