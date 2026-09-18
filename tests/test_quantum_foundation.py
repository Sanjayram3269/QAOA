import networkx as nx

from src.configurations import CONFIGURATIONS, get_config
from src.graph_generation import GraphSpec, generate_graph
from src.maxcut import approximation_ratio, cut_value, exact_maxcut


def test_configuration_registry_has_12_configs():
    assert len(CONFIGURATIONS) == 12
    assert get_config("C01").depth == 1
    assert get_config("C12").shots == 512


def test_graph_generation_is_deterministic():
    spec = GraphSpec(
        graph_id="GTEST",
        family="erdos_renyi",
        num_nodes=10,
        seed=12345,
        probability=0.35,
    )
    g1 = generate_graph(spec)
    g2 = generate_graph(spec)
    assert sorted(g1.edges()) == sorted(g2.edges())


def test_exact_maxcut_cycle():
    graph = nx.cycle_graph(6)
    optimum, partition = exact_maxcut(graph)
    assert optimum == 6
    assert cut_value(graph, partition) == optimum


def test_approximation_ratio():
    assert approximation_ratio(5, 10) == 0.5
    assert approximation_ratio(0, 0) == 1.0
