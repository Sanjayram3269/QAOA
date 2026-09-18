import networkx as nx

from src.evaluation import SCHEMA_VERSION, evaluate_graph


REQUIRED_COLUMNS = {
    "experiment_id",
    "graph_id",
    "graph_family",
    "num_nodes",
    "num_edges",
    "graph_seed",
    "noise_condition",
    "config_id",
    "depth",
    "optimizer",
    "shots_per_circuit",
    "expected_cut",
    "best_sampled_cut",
    "exact_optimum",
    "expected_approximation_ratio",
    "best_sampled_approximation_ratio",
    "optimizer_evaluations",
    "circuit_executions",
    "total_executed_shots",
    "two_qubit_gates",
    "circuit_depth",
    "simulator_runtime_seconds",
    "run_seed",
    "run_status",
    "failure_reason",
    "schema_version",
}


def test_evaluate_graph_returns_all_configs_for_each_paired_seed():
    graph = nx.cycle_graph(4)

    results = evaluate_graph(
        graph=graph,
        graph_id="GTEST",
        graph_family="test",
        graph_seed=123,
        noise_condition="N0",
        run_seeds=(1000, 1001),
        exact_max_nodes=20,
        max_circuit_executions=2,
    )

    assert len(results) == 24
    assert REQUIRED_COLUMNS.issubset(results.columns)
    assert results["experiment_id"].is_unique
    assert set(results["run_seed"]) == {1000, 1001}

    expected_configs = {f"C{i:02d}" for i in range(1, 13)}
    for _, seed_rows in results.groupby("run_seed"):
        assert set(seed_rows["config_id"]) == expected_configs

    assert (results["graph_id"] == "GTEST").all()
    assert (results["graph_family"] == "test").all()
    assert (results["graph_seed"] == 123).all()
    assert (results["noise_condition"] == "N0").all()
    assert (results["exact_optimum"] == 4).all()
    assert (results["run_status"] == "success").all()
    assert (results["schema_version"] == SCHEMA_VERSION).all()

    assert (
        results["total_executed_shots"]
        == results["shots_per_circuit"] * results["circuit_executions"]
    ).all()
    assert (results["circuit_executions"] <= 2).all()
    assert (results["two_qubit_gates"] > 0).all()

    assert results["expected_approximation_ratio"].between(0.0, 1.0).all()
    assert results["best_sampled_approximation_ratio"].between(0.0, 1.0).all()
