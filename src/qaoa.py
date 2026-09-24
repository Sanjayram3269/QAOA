"""Minimal, reproducible QAOA engine for Max-Cut.

The implementation explicitly controls circuit construction, transpilation,
shots, noise, seeds, and optimizer evaluations.

Performance notes
-----------------
* Optimized runs build and transpile one parameterized circuit and reuse it
  for every optimizer evaluation.
* N1/N2 evaluation reuses the parameters learned from N0 and performs exactly
  one simulator execution per configuration.
* The simulator device is selected with QAOA_DEVICE=CPU or QAOA_DEVICE=GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .maxcut import cut_value
from .noise import build_noise_model

import os


@dataclass(frozen=True)
class QAOAResult:
    expected_cut: float
    best_sampled_cut: int
    optimizer: str
    depth: int
    shots: int
    noise_condition: str
    optimizer_evaluations: int
    circuit_executions: int
    total_executed_shots: int
    two_qubit_gates: int
    circuit_depth: int
    simulator_runtime_seconds: float
    seed: int
    optimal_parameters: tuple[float, ...]


def _get_device() -> str:
    """Return the configured Aer device."""
    device = os.getenv("QAOA_DEVICE", "CPU").upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError("QAOA_DEVICE must be CPU or GPU")
    return device


def _make_simulator(noise_condition: str) -> AerSimulator:
    """Create the Aer simulator for a requested noise condition."""
    noise_model = build_noise_model(noise_condition)
    device = _get_device()

    if noise_model is None:
        return AerSimulator(
            method="statevector",
            device=device,
        )

    return AerSimulator(
        method="automatic",
        noise_model=noise_model,
        device=device,
    )


def build_qaoa_circuit(
    graph: nx.Graph,
    depth: int,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """Build a concrete QAOA circuit for supplied gamma/beta values."""
    n = graph.number_of_nodes()

    if len(gamma) != depth or len(beta) != depth:
        raise ValueError("gamma and beta must each have length equal to depth")

    circuit = QuantumCircuit(n, n)
    circuit.h(range(n))

    for layer in range(depth):
        for u, v in graph.edges():
            circuit.rzz(-float(gamma[layer]), u, v)

        for qubit in range(n):
            circuit.rx(2.0 * float(beta[layer]), qubit)

    circuit.measure(range(n), range(n))
    return circuit


def _build_parameterized_circuit(
    graph: nx.Graph,
    depth: int,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build one reusable parameterized QAOA circuit."""
    n = graph.number_of_nodes()

    gammas = [Parameter(f"gamma_{layer}") for layer in range(depth)]
    betas = [Parameter(f"beta_{layer}") for layer in range(depth)]

    circuit = QuantumCircuit(n, n)
    circuit.h(range(n))

    for layer in range(depth):
        for u, v in graph.edges():
            circuit.rzz(-gammas[layer], u, v)

        for qubit in range(n):
            circuit.rx(2.0 * betas[layer], qubit)

    circuit.measure(range(n), range(n))
    return circuit, gammas, betas


def _counts_to_cut_statistics(
    graph: nx.Graph,
    counts: dict[str, int],
) -> tuple[float, int]:
    """Convert measurement counts into expected and best sampled cut values."""
    total = sum(counts.values())

    if total == 0:
        raise RuntimeError("Simulator returned zero measurement shots")

    expected = 0.0
    best = 0
    n = graph.number_of_nodes()

    for bitstring, frequency in counts.items():
        bitstring = bitstring.replace(" ", "")

        if len(bitstring) != n:
            raise RuntimeError(
                f"Expected {n} measured bits, received {len(bitstring)}"
            )

        partition = [int(bitstring[n - 1 - node]) for node in range(n)]
        value = cut_value(graph, partition)
        expected += value * frequency
        best = max(best, value)

    return expected / total, best


def _spsa(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    max_evals: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Small SPSA implementation with a strict evaluation bound."""
    if max_evals < 1:
        raise ValueError("SPSA requires at least one objective evaluation")

    rng = np.random.default_rng(seed)
    theta = initial.astype(float).copy()
    best_theta = theta.copy()

    best_value = objective(theta)
    evaluations = 1
    iteration = 1

    while evaluations + 2 <= max_evals:
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        ck = 0.10 / (iteration**0.101)
        ak = 0.15 / ((iteration + 10) ** 0.602)

        plus = theta + ck * delta
        minus = theta - ck * delta

        y_plus = objective(plus)
        y_minus = objective(minus)
        evaluations += 2

        if y_plus < best_value:
            best_value = y_plus
            best_theta = plus.copy()

        if y_minus < best_value:
            best_value = y_minus
            best_theta = minus.copy()

        gradient = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = theta - ak * gradient

        if evaluations < max_evals:
            current = objective(theta)
            evaluations += 1

            if current < best_value:
                best_value = current
                best_theta = theta.copy()

        iteration += 1

    return best_theta, best_value, evaluations


def _run_once(
    graph: nx.Graph,
    depth: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    shots: int,
    noise_condition: str,
    seed: int,
) -> tuple[float, int, int, int]:
    """Run one fixed-parameter QAOA circuit execution.

    This helper is used by evaluate_qaoa_parameters() for N1/N2. It builds
    the concrete circuit, transpiles it once, executes it once, and returns
    quality plus resource metrics.
    """
    circuit = build_qaoa_circuit(
        graph=graph,
        depth=depth,
        gamma=gamma,
        beta=beta,
    )

    simulator = _make_simulator(noise_condition)

    compiled = transpile(
        circuit,
        simulator,
        seed_transpiler=seed,
    )

    two_qubit_gates = sum(
        1 for instruction in compiled.data if len(instruction.qubits) == 2
    )
    circuit_depth = int(compiled.depth())

    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts(compiled)
    expected, best_sampled = _counts_to_cut_statistics(graph, counts)

    return expected, best_sampled, circuit_depth, two_qubit_gates


def evaluate_qaoa_parameters(
    graph: nx.Graph,
    depth: int,
    optimizer: str,
    shots: int,
    noise_condition: str,
    seed: int,
    parameters: tuple[float, ...],
) -> QAOAResult:
    """Evaluate fixed QAOA parameters without running an optimizer."""
    if depth < 1:
        raise ValueError("QAOA depth must be at least 1")

    if shots <= 0:
        raise ValueError("shots must be positive")

    parameters_array = np.asarray(parameters, dtype=float)

    if len(parameters_array) != 2 * depth:
        raise ValueError(
            f"Expected {2 * depth} parameters, got {len(parameters_array)}"
        )

    gamma = parameters_array[:depth]
    beta = parameters_array[depth:]

    start = perf_counter()

    expected, best_sampled, circuit_depth, two_qubit_gates = _run_once(
        graph=graph,
        depth=depth,
        gamma=gamma,
        beta=beta,
        shots=shots,
        noise_condition=noise_condition,
        seed=seed,
    )

    runtime = perf_counter() - start

    return QAOAResult(
        expected_cut=expected,
        best_sampled_cut=best_sampled,
        optimizer=optimizer,
        depth=depth,
        shots=shots,
        noise_condition=noise_condition,
        optimizer_evaluations=0,
        circuit_executions=1,
        total_executed_shots=shots,
        two_qubit_gates=two_qubit_gates,
        circuit_depth=circuit_depth,
        simulator_runtime_seconds=runtime,
        seed=seed,
        optimal_parameters=tuple(float(x) for x in parameters_array),
    )


def run_qaoa(
    graph: nx.Graph,
    depth: int,
    optimizer: str,
    shots: int,
    noise_condition: str = "N0",
    seed: int = 0,
    max_circuit_executions: int = 200,
) -> QAOAResult:
    """Optimize and evaluate one QAOA configuration."""
    if depth < 1:
        raise ValueError("QAOA depth must be at least 1")

    if optimizer not in {"COBYLA", "SPSA"}:
        raise ValueError("Optimizer must be COBYLA or SPSA")

    if shots <= 0:
        raise ValueError("shots must be positive")

    if max_circuit_executions < 2:
        raise ValueError("At least two circuit executions are required")

    rng = np.random.default_rng(seed)
    initial = np.concatenate(
        [
            rng.uniform(0.0, np.pi, depth),
            rng.uniform(0.0, np.pi / 2.0, depth),
        ]
    )

    # Build and transpile once; reuse the compiled parameterized circuit.
    parameterized_circuit, gamma_params, beta_params = (
        _build_parameterized_circuit(graph, depth)
    )
    simulator = _make_simulator(noise_condition)

    compiled = transpile(
        parameterized_circuit,
        simulator,
        seed_transpiler=seed,
    )

    two_qubit_gates = sum(
        1 for instruction in compiled.data if len(instruction.qubits) == 2
    )
    circuit_depth = int(compiled.depth())

    optimizer_evaluations = 0
    optimizer_budget = max_circuit_executions - 1

    def objective(parameters: np.ndarray) -> float:
        nonlocal optimizer_evaluations

        gamma = parameters[:depth]
        beta = parameters[depth:]

        parameter_values = {
            parameter: float(gamma[index])
            for index, parameter in enumerate(gamma_params)
        }
        parameter_values.update(
            {
                parameter: float(beta[index])
                for index, parameter in enumerate(beta_params)
            }
        )

        bound_circuit = compiled.assign_parameters(
            parameter_values,
            inplace=False,
        )

        evaluation_seed = seed + optimizer_evaluations

        result = simulator.run(
            bound_circuit,
            shots=shots,
            seed_simulator=evaluation_seed,
        ).result()

        counts = result.get_counts(bound_circuit)
        expected, _ = _counts_to_cut_statistics(graph, counts)

        optimizer_evaluations += 1
        return -expected

    start = perf_counter()

    if optimizer == "COBYLA":
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            options={
                "maxiter": optimizer_budget,
                "rhobeg": 0.5,
            },
        )
        parameters = np.asarray(result.x, dtype=float)
    else:
        parameters, _, _ = _spsa(
            objective,
            initial,
            optimizer_budget,
            seed,
        )

    # One final sampling execution is deliberately kept separate from the
    # optimizer evaluations so circuit_executions = optimizer_evaluations + 1.
    gamma = parameters[:depth]
    beta = parameters[depth:]

    final_values = {
        parameter: float(gamma[index])
        for index, parameter in enumerate(gamma_params)
    }
    final_values.update(
        {
            parameter: float(beta[index])
            for index, parameter in enumerate(beta_params)
        }
    )

    final_circuit = compiled.assign_parameters(
        final_values,
        inplace=False,
    )

    final_result = simulator.run(
        final_circuit,
        shots=shots,
        seed_simulator=seed + 100000,
    ).result()

    final_counts = final_result.get_counts(final_circuit)
    expected, best_sampled = _counts_to_cut_statistics(graph, final_counts)

    runtime = perf_counter() - start
    circuit_executions = optimizer_evaluations + 1

    if circuit_executions > max_circuit_executions:
        raise RuntimeError("Circuit-execution cap was exceeded")

    return QAOAResult(
        expected_cut=expected,
        best_sampled_cut=best_sampled,
        optimizer=optimizer,
        depth=depth,
        shots=shots,
        noise_condition=noise_condition,
        optimizer_evaluations=optimizer_evaluations,
        circuit_executions=circuit_executions,
        total_executed_shots=shots * circuit_executions,
        two_qubit_gates=two_qubit_gates,
        circuit_depth=circuit_depth,
        simulator_runtime_seconds=runtime,
        seed=seed,
        optimal_parameters=tuple(float(x) for x in parameters),
    )
