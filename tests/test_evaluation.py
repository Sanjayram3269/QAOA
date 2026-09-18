import networkx as nx

from src.evaluation import evaluate_graph


def test_evaluate_graph_returns_all_configurations():
    graph = nx.cycle_graph(4)

    results = evaluate_graph(
        graph=graph,
        graph_id="GTEST",
        graph_family="test",
        graph_seed=123,
        noise_condition="N0",
        run_seed_start=1000,
        exact_max_nodes=20,
        max_optimizer_evals=1,
    )

    assert len(results) == 12
    assert set(results["config_id"]) == {
        f"C{i:02d}" for i in range(1, 13)
    }

    assert (results["graph_id"] == "GTEST").all()
    assert (results["graph_family"] == "test").all()
    assert (results["graph_seed"] == 123).all()
    assert (results["noise_condition"] == "N0").all()

    assert (results["exact_optimum"] == 4).all()

    assert (
        (results["expected_approximation_ratio"] >= 0.0)
        & (results["expected_approximation_ratio"] <= 1.0)
    ).all()

    assert (
        (results["best_sampled_approximation_ratio"] >= 0.0)
        & (results["best_sampled_approximation_ratio"] <= 1.0)
    ).all()