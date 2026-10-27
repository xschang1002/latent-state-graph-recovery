"""Study C thin-wrapper checks on synthetic retained-case dictionaries."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_symmetric_controls as wrapper


def example():
    support = np.zeros((2, 1, 2, 2), dtype=int); support[1, 0, 0, 1] = 1
    regimes = np.array([0,0,1,1])
    scores = np.zeros((4,1,2,2)); scores[:,0,0,1] = [0,1,2,3]
    arrays = {"state_support":support, "regimes":regimes, "bounds":np.array([[i,i+1] for i in range(4)]),
              "window_scores":scores, "emission_forced_hard":scores.copy(), "emission_native_hard":scores.copy(),
              "true_window_labels":regimes}
    identity = {"seed":70000,"duration":300,"h":1,"stride":1,"cell":"D300_h1_s1"}
    metric = wrapper.core.score_metrics(scores, support[regimes], support[0]!=support[1])
    methods = [{"method":name, **metric} for name in ["raw_window","forced_hard","native_hard"]]
    return {"identity":identity,"arrays":arrays,"record":{"methods":methods,"diagnostic":{"error":"failure"}},
            "fit_status":"ok","gate_accepted":True}


def test_adapter_uses_window_centres_and_retains_window_identity():
    case=wrapper.adapt_case(example())
    assert (case['study'],case['M'],case['h'],case['stride'])==('C',4,1,1)
    np.testing.assert_array_equal(case['active_truth'][:,0,0,1],[False,False,True,True])


def test_window_attributes_are_not_averaged_or_merged_across_cells():
    detail=pd.DataFrame([{'cell':'a','h':200,'stride':100,'M':35,'h_over_D':2/3},
                         {'cell':'b','h':300,'stride':150,'M':23,'h_over_D':1}])
    displayed=wrapper.attach_window_attributes(pd.DataFrame({'cell':['b','a']}),detail)
    assert displayed.h.tolist()==[300,200]
    assert displayed.M.tolist()==[23,35]
    with pytest.raises(ValueError): wrapper.attach_window_attributes(pd.DataFrame({'cell':['a']}),pd.concat([detail,detail.assign(h=500)]))


def test_complete_case_uses_frozen_core_and_all_prespecified_methods(monkeypatch):
    monkeypatch.setattr(wrapper.study,'iter_cases',lambda root:iter([example()]))
    tables,report=wrapper.compute(Path('.'))
    assert len(tables['detail.csv'])==13 and report['reference_method_rows_checked']==3
    assert tables['affine_events.csv'].empty
    assert tables['summary.csv'].h.eq(1).all() and tables['contrasts.csv'].M.eq(4).all()
    assert report['new_fits']==report['new_simulations']==0


def test_failed_case_retains_methods_and_empty_qa(monkeypatch):
    failed=example();failed['fit_status']='failed';failed['gate_accepted']=None
    monkeypatch.setattr(wrapper.study,'iter_cases',lambda root:iter([failed]))
    tables,report=wrapper.compute(Path('.'))
    assert len(tables['detail.csv'])==13 and tables['detail.csv'].status.eq('failed').all()
    assert report['failed_cases']==1 and tables['affine_events.csv'].empty
