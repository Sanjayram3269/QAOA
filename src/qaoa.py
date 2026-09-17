"""Minimal, reproducible QAOA engine for Max-Cut.

The implementation intentionally avoids a high-level estimator API so that the
experiment has explicit control over circuit construction, shots, noise, seeds,
and optimizer evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .maxcut import cut_value
from .noise import build_noise_model


@dataclass(frozen=True)
class QAOAResult:
    expected_cut: float
    best_sampled_cut: int
    optimizer: str
    depth: int
    shots: int
    noise_condition: str
    optimizer_evaluations: int
    two_qubit_gates: int
    circuit_depth: int
    simulator_runtime_seconds: float
    seed: int
    optimal_parameters: tuple[float, ...]


def build_qaoa_circuit(
    graph: nx.Graph,
    depth: int,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """Build a QAOA Max-Cut circuit for concrete gamma/beta parameters."""
    n = graph.number_of_nodes()
    if len(gamma) != depth or len(beta) != depth:
        raise ValueError("gamma and beta must each have length equal to depth")

    circuit = QuantumCircuit(n, n)
    circuit.h(range(n))

    for layer in range(depth):
        # For C_ij=(1-Z_i Z_j)/2, RZZ(-gamma) implements the ZZ phase
        # up to a global phase, which does not affect measurement statistics.
        for u, v in graph.edges():
            circuit.rzz(-float(gamma[layer]), u, v)

        for qubit in range(n):
            circuit.rx(2.0 * float(beta[layer]), qubit)

    circuit.measure(range(n), range(n))
    return circuit


def _counts_to_cut_statistics(
    graph: nx.Graph,
    counts: dict[str, int],
) -> tuple[float, int]:
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("Simulator returned zero measurement shots")

    expected = 0.0
    best = 0
    n = graph.number_of_nodes()

    for bitstring, frequency in counts.items():
        if len(bitstring) != n:
            bitstring = bitstring.replace(" ", "")
        # Qiskit displays classical bits as c[n-1] ... c[0].
        partition = [int(bitstring[n - 1 - node]) for node in range(n)]
        value = cut_value(graph, partition)
        expected += value * frequency
        best = max(best, value)

    return expected / total, best


def _run_once(
    graph: nx.Graph,
    depth: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    shots: int,
    noise_condition: str,
    seed: int,
) -> tuple[float, int, int, int]:
    circuit = build_qaoa_circuit(graph, depth, gamma, beta)
    noise_model = build_noise_model(noise_condition)
    simulator = AerSimulator(noise_model=noise_model)
    compiled = transpile(circuit, simulator, seed_transpiler=seed)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts(compiled)
    expected, best = _counts_to_cut_statistics(graph, counts)
    return expected, best, compiled.depth(), compiled.count_ops().get("cx", 0)


def _spsa(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    max_evals: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Small SPSA implementation with exactly bounded objective evaluations."""
    rng = np.random.default_rng(seed)
    theta = initial.astype(float).copy()
    best_theta = theta.copy()
    best_value = objective(theta)
    evaluations = 1

    iterations = max(1, (max_evals - 1) // 2)
    for k in range(1, iterations + 1):
        if evaluations + 2 > max_evals:
            break
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        ck = 0.10 / (k ** 0.101)
        ak = 0.15 / ((k + 10) ** 0.602)

        plus = theta + ck * delta
        minus = theta - ck * delta
        y_plus = objective(plus)
        y_minus = objective(minus)
        evaluations += 2

        gradient = (y_plus - y_minus) / (2.0 * ck) * delta
        theta = theta - ak * gradient

        current = objective(theta)
        evaluations += 1
        if current < best_value:
            best_value = current
            best_theta = theta.copy()
        if evaluations >= max_evals:
            break

    return best_theta, best_value, evaluations


def run_qaoa(
    graph: nx.Graph,
    depth: int,
    optimizer: str,
    shots: int,
    noise_condition: str = "N0",
    seed: int = 0,
    max_optimizer_evals: int = 80,
) -> QAOAResult:
    """Optimize and evaluate a QAOA configuration on one graph."""
    if depth < 1:
        raise ValueError("QAOA depth must be at least 1")
    if optimizer not in {"COBYLA", "SPSA"}:
        raise ValueError("Optimizer must be COBYLA or SPSA")
    if shots <= 0:
        raise ValueError("shots must be positive")

    rng = np.random.default_rng(seed)
    initial = np.concatenate([
        rng.uniform(0.0, np.pi, depth),
        rng.uniform(0.0, np.pi / 2.0, depth),
    ])

    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        gamma = parameters[:depth]
        beta = parameters[depth:]
        expected, _, _, _ = _run_once(
            graph, depth, gamma, beta, shots, noise_condition, seed + evaluations
        )
        evaluations += 1
        return -expected

    start = perf_counter()
    if optimizer == "COBYLA":
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            options={"maxiter": max_optimizer_evals, "rhobeg": 0.5},
        )
        parameters = np.asarray(result.x, dtype=float)
    else:
        parameters, _, _ = _spsa(objective, initial, max_optimizer_evals, seed)

    gamma = parameters[:depth]
    beta = parameters[depth:]
    expected, best_sampled, circuit_depth, two_qubit_gates = _run_once(
        graph, depth, gamma, beta, shots, noise_condition, seed + 100000
    )
    runtime = perf_counter() - start

    return QAOAResult(
        expected_cut=expected,
        best_sampled_cut=best_sampled,
        optimizer=optimizer,
        depth=depth,
        shots=shots,
        noise_condition=noise_condition,
        optimizer_evaluations=evaluations,
        two_qubit_gates=two_qubit_gates,
        circuit_depth=circuit_depth,
        simulator_runtime_seconds=runtime,
        seed=seed,
        optimal_parameters=tuple(float(x) for x in parameters),
    )
