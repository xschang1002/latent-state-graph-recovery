import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "experiments" / "logs" / "test_real_data_entry"


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def workspace_tmp(name):
    path = TEST_TMP / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def inventory_args(workdir, **overrides):
    args = {
        "raw_dir": workdir,
        "input_csv": None,
        "columns": "",
        "time_column": "timestamp",
        "dataset_name": "",
        "min_rows": 5,
        "preferred_rows": 10,
        "min_numeric_columns": 2,
        "preferred_numeric_columns": 3,
        "output_dir": workdir,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def write_ready_csv(path):
    rows = ["timestamp,a,b,c"]
    for idx in range(10):
        rows.append(f"2024-01-{idx + 1:02d},{idx},{idx + 1},{idx + 2}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_real_data_inventory_marks_ready_csv():
    inventory = load_script("check_real_data_inventory")
    workdir = workspace_tmp("ready")
    csv_path = workdir / "ready.csv"
    write_ready_csv(csv_path)

    row = inventory.inspect_csv(csv_path, inventory_args(workdir))

    assert row["status"] == "ready"
    assert row["rows"] == 10
    assert row["numeric_columns"] == 3
    assert row["selected_columns"] == "a,b,c"
    assert "run_downstream_forecasting.py" in row["suggested_command"]


def test_real_data_inventory_blocks_too_few_numeric_columns():
    inventory = load_script("check_real_data_inventory")
    workdir = workspace_tmp("blocked")
    csv_path = workdir / "blocked.csv"
    csv_path.write_text("timestamp,a\n2024-01-01,1\n2024-01-02,2\n", encoding="utf-8")

    row = inventory.inspect_csv(csv_path, inventory_args(workdir))

    assert row["status"] == "blocked"
    assert "row count below minimum" in row["issues"]
    assert "numeric column count below minimum" in row["issues"]


def test_real_data_inventory_autodetects_time_column():
    inventory = load_script("check_real_data_inventory")
    workdir = workspace_tmp("autodetect_time")
    csv_path = workdir / "auto.csv"
    csv_path.write_text(
        "datetime,a,b,c\n"
        "2024-01-01 00:00:00,1,2,3\n"
        "2024-01-01 01:00:00,2,3,4\n"
        "2024-01-01 02:00:00,3,4,5\n"
        "2024-01-01 03:00:00,4,5,6\n"
        "2024-01-01 04:00:00,5,6,7\n",
        encoding="utf-8",
    )

    row = inventory.inspect_csv(csv_path, inventory_args(workdir, time_column="", preferred_rows=5))

    assert row["status"] == "ready"
    assert row["time_column"] == "datetime"
    assert "--time-column datetime" in row["suggested_command"]


def test_real_data_inventory_autodetects_time_index_column():
    inventory = load_script("check_real_data_inventory")
    workdir = workspace_tmp("autodetect_time_index")
    csv_path = workdir / "auto_index.csv"
    csv_path.write_text(
        "time_index,a,b,c\n"
        "0,1,2,3\n"
        "1,2,3,4\n"
        "2,3,4,5\n"
        "3,4,5,6\n"
        "4,5,6,7\n",
        encoding="utf-8",
    )

    row = inventory.inspect_csv(csv_path, inventory_args(workdir, time_column="", preferred_rows=5))

    assert row["status"] == "ready"
    assert row["time_column"] == "time_index"
    assert row["selected_columns"] == "a,b,c"
    assert "--time-column time_index" in row["suggested_command"]


def runner_args(workdir):
    return SimpleNamespace(
        output_root=workdir / "real_benchmarks",
        missing="interpolate",
        seeds=[0, 1],
        max_lag=2,
        train_fraction=0.7,
        ridge_alpha=1.0,
        lasso_alpha=0.005,
        graph_fraction=0.25,
        graph_min_weight=0.10,
        graph_self_weight=1.0,
        graph_weight_power=1.0,
        window_size=80,
        window_stride=40,
        window_aggregate="max",
        state_aggregate="mean",
        stcg_v1_states=2,
        stcg_v1_min_improvement=0.12,
        stcg_v1_force_states=False,
        diagnostic_top_k=5,
    )


def test_real_benchmark_runner_builds_dataset_command():
    runner = load_script("run_real_benchmark_from_inventory")
    workdir = workspace_tmp("runner_command")
    row = {
        "file": str(ROOT / "data" / "raw" / "example.csv"),
        "dataset": "Example Dataset",
        "status": "warning",
        "rows": "800",
        "numeric_columns": "4",
        "selected_columns": "a,b,c,d",
        "time_column": "timestamp",
    }

    command, output_dir, figure_dir = runner.build_command(row, runner_args(workdir))

    assert command[1] == "scripts\\run_downstream_forecasting.py"
    assert "--input-csv" in command
    assert "--columns" in command
    assert "a,b,c,d" in command
    assert "--stcg-v1-states" in command
    assert "--stcg-v1-min-improvement" in command
    assert output_dir.name == "tables"
    assert figure_dir.name == "figures"
    assert "Example_Dataset" in str(output_dir)


def test_real_benchmark_runner_writes_dry_run_plan():
    runner = load_script("run_real_benchmark_from_inventory")
    workdir = workspace_tmp("runner_plan")
    row = {
        "file": str(ROOT / "data" / "raw" / "example.csv"),
        "dataset": "example",
        "status": "ready",
        "rows": "1200",
        "numeric_columns": "4",
        "selected_columns": "a,b,c,d",
        "time_column": "",
    }
    command, output_dir, figure_dir = runner.build_command(row, runner_args(workdir))
    plan_path = workdir / "plan.md"

    runner.write_plan(plan_path, [row], [(row, command, output_dir, figure_dir)], execute=False)

    text = plan_path.read_text(encoding="utf-8")
    assert "Mode: `dry-run`" in text
    assert "python scripts\\run_downstream_forecasting.py" in text
    assert "example" in text


def test_beijing_air_quality_prepare_pm25_zip():
    fetcher = load_script("fetch_beijing_air_quality")
    workdir = workspace_tmp("beijing_fetch")
    zip_path = workdir / "beijing.zip"
    output_path = workdir / "BeijingPM25.csv"

    station_csv = (
        '"No","year","month","day","hour","PM2.5","station"\n'
        '1,2013,3,1,0,4,"A"\n'
        '2,2013,3,1,1,NA,"A"\n'
        '3,2013,3,1,2,8,"A"\n'
    )
    station_csv_b = station_csv.replace('"A"', '"B"').replace(",4,", ",10,").replace(",8,", ",14,")
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as nested:
        nested.writestr("PRSA_Data_20130301-20170228/PRSA_Data_A.csv", station_csv)
        nested.writestr("PRSA_Data_20130301-20170228/PRSA_Data_B.csv", station_csv_b)
    with zipfile.ZipFile(zip_path, "w") as outer:
        outer.writestr(fetcher.NESTED_ZIP, nested_bytes.getvalue())

    result = fetcher.prepare_pm25_zip(zip_path, output_path)

    assert result["rows"] == 3
    assert result["n_stations"] == 2
    text = output_path.read_text(encoding="utf-8")
    assert "datetime,A,B" in text
    assert "2013-03-01 01:00:00,6.0,12.0" in text


def test_occupancy_prepare_sensor_csv_and_labels():
    fetcher = load_script("fetch_uci_occupancy")
    workdir = workspace_tmp("occupancy_fetch")
    zip_path = workdir / "occupancy.zip"
    output_path = workdir / "OccupancySensors.csv"
    label_path = workdir / "OccupancySensors_labels.csv"

    files = {
        "datatest.txt": (
            "date,Temperature,Humidity,Light,CO2,HumidityRatio,Occupancy\n"
            "2015-02-02 14:19:00,21.0,25.0,10.0,400.0,0.0030,0\n"
            "2015-02-02 14:20:00,21.1,25.1,,410.0,0.0031,0\n"
        ),
        "datatraining.txt": (
            "date,Temperature,Humidity,Light,CO2,HumidityRatio,Occupancy\n"
            "2015-02-04 17:51:00,21.2,25.2,30.0,420.0,0.0032,1\n"
            "2015-02-04 17:52:00,21.3,25.3,40.0,430.0,0.0033,1\n"
        ),
        "datatest2.txt": (
            "date,Temperature,Humidity,Light,CO2,HumidityRatio,Occupancy\n"
            "2015-02-11 14:48:00,21.4,25.4,50.0,440.0,0.0034,0\n"
            "2015-02-11 14:49:00,21.5,25.5,60.0,450.0,0.0035,1\n"
        ),
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    result = fetcher.prepare_occupancy_zip(zip_path, output_path, label_path)

    assert result["rows"] == 6
    assert result["n_sensors"] == 5
    assert result["missing_cells"] == 0
    assert result["occupancy_fraction"] == 0.5
    assert result["occupancy_transitions"] == 3
    assert "datetime,Temperature,Humidity,Light,CO2,HumidityRatio" in output_path.read_text(encoding="utf-8")
    assert "2015-02-02 14:20:00,21.1,25.1,20.0,410.0,0.0031" in output_path.read_text(
        encoding="utf-8"
    )
    label_text = label_path.read_text(encoding="utf-8")
    assert "datetime,Occupancy,source_split" in label_text
    assert "2015-02-04 17:51:00,1,datatraining" in label_text


def test_real_state_label_alignment_scores_windows():
    comparer = load_script("compare_real_state_labels")
    workdir = workspace_tmp("state_label_alignment")
    states_path = workdir / "states.csv"
    labels_path = workdir / "labels.csv"

    states_path.write_text(
        "dataset,seed,window_id,start,end,center,inferred_state\n"
        "Toy,0,0,0,4,2.0,5\n"
        "Toy,0,1,4,8,6.0,7\n"
        "Toy,1,0,0,4,2.0,5\n"
        "Toy,1,1,4,8,6.0,5\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "datetime,mode\n"
        "t0,A\n"
        "t1,A\n"
        "t2,A\n"
        "t3,B\n"
        "t4,B\n"
        "t5,B\n"
        "t6,B\n"
        "t7,A\n",
        encoding="utf-8",
    )

    summary, detail = comparer.compare_state_labels(states_path, labels_path, "Toy", "mode")

    assert len(summary) == 2
    assert len(detail) == 4
    assert summary[0]["true_label_counts"] == "A:1;B:1"
    assert summary[0]["inferred_state_count"] == 2
    assert summary[0]["aligned_accuracy"] == 1.0
    assert summary[1]["inferred_state_count"] == 1
    assert summary[1]["aligned_accuracy"] == 0.5
    assert detail[0]["true_label"] == "A"
    assert detail[0]["label_purity"] == 0.75


def test_hydraulic_prepare_downsamples_cycles_and_repeats_labels():
    fetcher = load_script("fetch_hydraulic_systems")
    workdir = workspace_tmp("hydraulic_fetch")
    zip_path = workdir / "hydraulic.zip"
    output_path = workdir / "HydraulicSystems.csv"
    label_path = workdir / "HydraulicSystems_labels.csv"

    def sensor_rows(rate, offset):
        width = rate * fetcher.SECONDS_PER_CYCLE
        row_a = " ".join(str(offset + idx) for idx in range(width))
        row_b = " ".join(str(offset + 1000 + idx) for idx in range(width))
        return f"{row_a}\n{row_b}\n"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("profile.txt", "3 100 0 130 1\n20 90 1 115 0\n")
        archive.writestr("PS1.txt", sensor_rows(100, 0))
        archive.writestr("FS1.txt", sensor_rows(10, 10_000))
        archive.writestr("TS1.txt", sensor_rows(1, 20_000))

    result = fetcher.prepare_hydraulic_zip(zip_path, output_path, label_path, ["PS1", "FS1", "TS1"])

    assert result["rows"] == 120
    assert result["cycles"] == 2
    assert result["n_sensors"] == 3
    assert result["missing_cells"] == 0
    assert result["label_state_counts"]["stable_flag"] == {"0": 1, "1": 1}
    output_text = output_path.read_text(encoding="utf-8")
    assert "time_index,PS1,FS1,TS1" in output_text
    assert "0,49.5,10004.5,20000.0" in output_text
    assert "60,1049.5,11004.5,21000.0" in output_text
    label_text = label_path.read_text(encoding="utf-8")
    assert "time_index,cycle,second,cooler_condition,valve_condition" in label_text
    assert "0,0,0,3,100,0,130,1" in label_text
    assert "60,1,0,20,90,1,115,0" in label_text


def test_arem_prepare_concatenates_trials_and_labels():
    fetcher = load_script("fetch_uci_arem")
    workdir = workspace_tmp("arem_fetch")
    zip_path = workdir / "arem.zip"
    output_path = workdir / "AReMActivities.csv"
    label_path = workdir / "AReMActivities_labels.csv"

    trial_a = (
        "# Task: walking\n"
        "# Frequency (Hz): 20\n"
        "# Clock (millisecond): 250\n"
        "# Duration (seconds): 120\n"
        "# Columns: time,avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23\n"
        "0,1,2,3,4,5,6\n"
        "250,2,3,4,5,6,7\n"
    )
    trial_b = trial_a.replace("# Task: walking", "# Task: lying").replace("0,1,2,3,4,5,6", "0,10,20,30,40,50,60")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("walking/dataset1.csv", trial_a)
        archive.writestr("lying/dataset1.csv", trial_b)

    result = fetcher.prepare_arem_zip(zip_path, output_path, label_path)

    assert result["rows"] == 4
    assert result["n_sequences"] == 2
    assert result["n_sensors"] == 6
    assert result["activity_counts"] == {"lying": 2, "walking": 2}
    assert result["activity_transitions"] == 1
    output_text = output_path.read_text(encoding="utf-8")
    assert "time_index,avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23" in output_text
    assert "0,10,20,30,40,50,60" in output_text
    label_text = label_path.read_text(encoding="utf-8")
    assert "time_index,activity,sequence_id,sample_in_sequence,source_file" in label_text
    assert "0,lying,0,0,lying/dataset1.csv" in label_text
    assert "2,walking,1,0,walking/dataset1.csv" in label_text


def test_arem_threshold_forecasting_summary_tracks_stcg_delta():
    sweep = load_script("run_arem_threshold_forecasting")
    rows = []
    for seed, dense_rmse, dynvar_rmse, stcg_rmse in [(0, 1.00, 0.95, 0.91), (1, 1.10, 0.97, 0.93)]:
        common = {
            "dataset": "Toy",
            "seed": seed,
            "setting": "auto_min_0.00",
            "ridge_alpha": 100.0,
            "state_count": 3,
            "auto_states": True,
            "min_state_improvement": 0.0,
            "threshold_passed": True,
            "kmeans_improvement": 0.05,
            "temporal_coherence": 0.8,
            "n_windows": 4,
            "true_state_count": 2,
            "inferred_state_count": 2,
            "adjusted_rand": 0.4,
            "aligned_accuracy": 0.75,
            "mean_label_purity": 1.0,
            "true_label_counts": "A:2;B:2",
            "inferred_state_counts": "0:2;1:2",
            "state_mapping": '{"0": "A", "1": "B"}',
            "mae": 0.5,
            "node_rmse_mean": 0.8,
            "node_rmse_max": 1.1,
        }
        rows.append({**common, "method": "Dense Ridge", "graph_min_weight": None, "graph_weight_power": None, "graph_self_weight": None, "rmse": dense_rmse})
        rows.append({**common, "method": "DynVAR-Lasso Weighted Ridge", "graph_min_weight": 0.01, "graph_weight_power": 4.0, "graph_self_weight": 1.0, "rmse": dynvar_rmse})
        rows.append({**common, "method": "STCG-v1 Weighted Ridge", "graph_min_weight": 0.01, "graph_weight_power": 4.0, "graph_self_weight": 1.0, "rmse": stcg_rmse})

    summary = sweep.aggregate_summary(rows, meaningful_rmse_delta=0.001)
    stcg = [row for row in summary if row["method"] == "STCG-v1 Weighted Ridge"][0]

    assert round(stcg["rmse_mean"], 3) == 0.920
    assert round(stcg["delta_vs_dynvar_weighted_rmse"], 3) == -0.040
    assert stcg["meaningfully_beats_dynvar_weighted"] is True
    assert stcg["inferred_state_count_distribution"] == '{"2": 2}'


def test_state_label_prediction_stratified_split_keeps_each_label():
    predictor = load_script("run_real_state_label_prediction")
    labels = ["A", "A", "A", "A", "B", "B", "B", "B"]

    train, test = predictor.stratified_window_split(labels, test_fraction=0.5, seed=0)

    assert len(train) == 4
    assert len(test) == 4
    assert {labels[idx] for idx in train} == {"A", "B"}
    assert {labels[idx] for idx in test} == {"A", "B"}


def test_state_label_prediction_maps_states_from_train_windows():
    predictor = load_script("run_real_state_label_prediction")
    states = predictor.np.array([0, 0, 1, 1])
    labels = ["cool", "cool", "hot", "hot"]
    train = predictor.np.array([0, 2])
    test = predictor.np.array([1, 3])

    mapping, fallback = predictor.fit_state_label_mapping(states, labels, train)
    predicted = predictor.predict_from_state_mapping(states, mapping, fallback, test)
    metrics = predictor.metric_values([labels[idx] for idx in test], predicted)

    assert mapping == {0: "cool", 1: "hot"}
    assert predicted == ["cool", "hot"]
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_state_count_selection_score_penalizes_tiny_clusters():
    selector = load_script("run_real_state_count_selection")

    good_score, good_balance = selector.state_count_selection_score(0.08, 0.90, 0.30, 3)
    tiny_score, tiny_balance = selector.state_count_selection_score(0.10, 0.90, 0.02, 5)

    assert good_score > 0.0
    assert round(good_balance, 3) == 0.9
    assert tiny_score == 0.0
    assert tiny_balance == 0.0


def test_state_count_selection_marks_best_label_free_score():
    selector = load_script("run_real_state_count_selection")
    rows = [
        {"seed": 0, "candidate_state_count": 2, "selection_score": 0.05, "temporal_coherence": 0.8},
        {"seed": 0, "candidate_state_count": 3, "selection_score": 0.08, "temporal_coherence": 0.7},
        {"seed": 1, "candidate_state_count": 2, "selection_score": 0.06, "temporal_coherence": 0.8},
        {"seed": 1, "candidate_state_count": 3, "selection_score": 0.06, "temporal_coherence": 0.9},
    ]

    selector.mark_selected(rows)

    selected = [(row["seed"], row["candidate_state_count"]) for row in rows if row["selected"]]
    assert selected == [(0, 3), (1, 3)]


def test_transition_warning_targets_future_label_changes():
    warning = load_script("run_hydraulic_transition_warning")
    labels = ["A", "A", "B", "B", "B", "A"]

    horizon_one = warning.transition_targets(labels, horizon=1)
    horizon_two = warning.transition_targets(labels, horizon=2)

    assert horizon_one.tolist() == [0, 1, 0, 0, 1]
    assert horizon_two.tolist() == [1, 1, 0, 1]


def test_transition_warning_chronological_split_preserves_order():
    warning = load_script("run_hydraulic_transition_warning")

    train, test = warning.chronological_split(10, train_fraction=0.6)

    assert train.tolist() == [0, 1, 2, 3, 4, 5]
    assert test.tolist() == [6, 7, 8, 9]


def test_transition_warning_validation_split_preserves_order():
    warning = load_script("run_hydraulic_transition_warning")

    train, validation, test = warning.chronological_train_validation_test_split(
        10,
        train_fraction=0.7,
        validation_fraction=0.2,
    )

    assert train.tolist() == [0, 1, 2, 3, 4]
    assert validation.tolist() == [5, 6]
    assert test.tolist() == [7, 8, 9]


def test_transition_warning_calibrates_threshold_on_validation():
    warning = load_script("run_hydraulic_transition_warning")
    y_validation = warning.np.array([0, 0, 1, 1])
    scores = warning.np.array([0.1, 0.2, 0.6, 0.8])

    threshold, value = warning.calibrate_threshold(y_validation, scores, "f1")

    assert threshold == 0.6
    assert value == 1.0


def test_transition_warning_stcg_score_uses_parsimony_and_balance():
    warning = load_script("run_hydraulic_transition_warning")

    good_score, good_balance = warning.stcg_selection_score(0.08, 0.90, 0.30, 3)
    split_score, split_balance = warning.stcg_selection_score(0.08, 0.90, 0.05, 6)
    tiny_score, tiny_balance = warning.stcg_selection_score(0.10, 0.90, 0.02, 5)

    assert good_score > split_score
    assert round(good_balance, 3) == 0.9
    assert round(split_balance, 3) == 0.3
    assert tiny_score == 0.0
    assert tiny_balance == 0.0


def test_transition_warning_stcg_transition_score_requires_multistate_signal():
    warning = load_script("run_hydraulic_transition_warning")

    good_score, good_balance = warning.stcg_transition_selection_score(0.08, 0.0, 0.30, 3)
    one_state_score, one_state_balance = warning.stcg_transition_selection_score(0.08, 0.0, 1.00, 1)
    tiny_score, tiny_balance = warning.stcg_transition_selection_score(0.08, 0.0, 0.02, 4)

    assert good_score > 0.0
    assert round(good_balance, 3) == 0.9
    assert one_state_score == 0.0
    assert one_state_balance == 0.0
    assert tiny_score == 0.0
    assert tiny_balance == 0.0


def test_transition_warning_state_dynamics_features_are_fixed_length():
    warning = load_script("run_hydraulic_transition_warning")
    labels = warning.np.asarray([2, 2, 1, 1, 1, 3])

    features = warning.stcg_state_dynamics_features(labels, max_states=4)

    assert features.shape == (18,)
    assert features[:4].round(3).tolist() == [0.5, 0.333, 0.167, 0.0]
    assert features[4:8].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert features[8:12].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert round(float(features[12]), 3) == 0.4
    assert round(float(features[13]), 3) == 0.167
    assert round(float(features[15]), 3) == 0.5
    assert round(float(features[17]), 3) == 0.75


def test_hydraulic_warning_ablation_selects_by_validation_metric():
    ablation = load_script("run_hydraulic_warning_ablation")
    rows = [
        {
            "label_column": "stable_flag",
            "horizon_cycles": 5,
            "method": "Sensor+STCG-v1 Logistic",
            "validation_threshold_metric": 0.20,
            "average_precision": 0.99,
            "config_rank": 0,
        },
        {
            "label_column": "stable_flag",
            "horizon_cycles": 5,
            "method": "Sensor+STCG-v1 Logistic",
            "validation_threshold_metric": 0.30,
            "average_precision": 0.10,
            "config_rank": 1,
        },
    ]

    selected = ablation.select_validation_rows(rows, ["Sensor+STCG-v1 Logistic"], "validation_threshold_metric")

    assert selected == [rows[1]]


def test_hydraulic_warning_ablation_formats_count_distribution():
    ablation = load_script("run_hydraulic_warning_ablation")
    values = ablation.np.asarray([3.0, 1.0, 3.0, 2.0, 2.0, 3.0])

    assert ablation.format_count_distribution(values) == "1:1;2:2;3:3"


def test_har_prepare_window_signal_means_and_labels():
    fetcher = load_script("fetch_uci_har")
    workdir = workspace_tmp("har_fetch")
    zip_path = workdir / "har.zip"
    output_path = workdir / "UCIHARWindowSignals.csv"
    label_path = workdir / "UCIHARWindowSignals_labels.csv"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("UCI HAR Dataset/activity_labels.txt", "1 WALKING\n2 SITTING\n")
        for split, labels, subjects in [("train", [1, 2], [7, 7]), ("test", [2], [8])]:
            archive.writestr(f"UCI HAR Dataset/{split}/y_{split}.txt", "\n".join(str(label) for label in labels) + "\n")
            archive.writestr(
                f"UCI HAR Dataset/{split}/subject_{split}.txt",
                "\n".join(str(subject) for subject in subjects) + "\n",
            )
            for signal_index, signal in enumerate(fetcher.SIGNALS):
                rows = []
                for row_index in range(len(labels)):
                    base = 10 * signal_index + row_index
                    rows.append(f"{base}.0 {base + 2}.0")
                archive.writestr(
                    f"UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt",
                    "\n".join(rows) + "\n",
                )

    result = fetcher.prepare_har_zip(zip_path, output_path, label_path)

    assert result["rows"] == 3
    assert result["n_features"] == 9
    assert result["n_subjects"] == 2
    assert result["activity_counts"] == {"SITTING": 2, "WALKING": 1}
    output_text = output_path.read_text(encoding="utf-8")
    assert "time_index,body_acc_x_mean,body_acc_y_mean" in output_text
    assert "0,1.0,11.0" in output_text
    label_text = label_path.read_text(encoding="utf-8")
    assert "time_index,activity_id,activity,subject,source_split,window_id" in label_text
    assert "0,1,WALKING,7,train,0" in label_text
    assert "2,2,SITTING,8,test,0" in label_text
