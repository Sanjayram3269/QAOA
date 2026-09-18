import networkx as nx
import numpy as np

from src.qaoa import build_qaoa_circuit, run_qaoa
from src.noise import build_noise_model


def test_qaoa_circuit_has_expected_parameters():
    graph = nx.cycle_graph(4)

    circuit = build_qaoa_circuit(
        graph=graph,
        depth=2,
        gamma=np.array([0.2, 0.3]),
        beta=np.array([0.1, 0.2]),
    )

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 4
    assert circuit.depth() > 0
    assert circuit.count_ops().get("measure", 0) == 4


def test_noise_conditions_are_available():
    assert build_noise_model("N0") is None
    assert build_noise_model("N1") is not None
    assert build_noise_model("N2") is not None


def test_qaoa_result_has_valid_metrics():
    graph = nx.cycle_graph(4)

    result = run_qaoa(
        graph=graph,
        depth=1,
        optimizer="COBYLA",
        shots=32,
        noise_condition="N0",
        seed=123,
        max_optimizer_evals=5,
    )

    assert 0.0 <= result.expected_cut <= graph.number_of_edges()
    assert 0 <= result.best_sampled_cut <= graph.number_of_edges()
    assert result.optimizer == "COBYLA"
    assert result.depth == 1
    assert result.shots == 32
    assert result.noise_condition == "N0"
    assert result.optimizer_evaluations <= 5
    assert len(result.optimal_parameters) == 2


def test_qaoa_supports_spsa():
    graph = nx.cycle_graph(4)

    result = run_qaoa(
        graph=graph,
        depth=1,
        optimizer="SPSA",
        shots=32,
        noise_condition="N0",
        seed=123,
        max_optimizer_evals=7,
    )

    assert result.optimizer == "SPSA"
    assert result.optimizer_evaluations <= 7
    assert len(result.optimal_parameters) == 2