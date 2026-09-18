"""Run the smallest end-to-end quantum pipeline before scaling experiments."""

import networkx as nx

from src.maxcut import exact_maxcut
from src.qaoa import run_qaoa


def main() -> None:
    graph = nx.cycle_graph(6)
    optimum, partition = exact_maxcut(graph)
    print(f"Exact Max-Cut optimum: {optimum}")
    print(f"Exact partition: {partition}")

    result = run_qaoa(
        graph=graph,
        depth=1,
        optimizer="COBYLA",
        shots=256,
        noise_condition="N0",
        seed=20260917,
        max_circuit_executions=10,
    )

    print("QAOA smoke-test result")
    print(f"  expected cut: {result.expected_cut:.4f}")
    print(f"  best sampled cut: {result.best_sampled_cut}")
    print(f"  optimizer evaluations: {result.optimizer_evaluations}")\n    print(f"  circuit executions: {result.circuit_executions}")\n    print(f"  total executed shots: {result.total_executed_shots}")
    print(f"  circuit depth: {result.circuit_depth}")
    print(f"  two-qubit gates: {result.two_qubit_gates}")
    print(f"  runtime (s): {result.simulator_runtime_seconds:.3f}")


if __name__ == "__main__":
    main()
