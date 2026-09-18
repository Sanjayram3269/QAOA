"""Canonical QAOA configuration registry.

The configuration IDs are part of the shared data contract with the ML pipeline.
Do not change the registry without recording the decision in DECISIONS.md.
"""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class QAOAConfig:
    config_id: str
    depth: int
    optimizer: str
    shots: int


DEPTHS = (1, 2, 3)
OPTIMIZERS = ("COBYLA", "SPSA")
SHOTS = (256, 512)


CONFIGURATIONS = tuple(
    QAOAConfig(
        config_id=f"C{index:02d}",
        depth=depth,
        optimizer=optimizer,
        shots=shots,
    )
    for index, (depth, optimizer, shots) in enumerate(
        product(DEPTHS, OPTIMIZERS, SHOTS), start=1
    )
)

CONFIG_BY_ID = {config.config_id: config for config in CONFIGURATIONS}


def get_config(config_id: str) -> QAOAConfig:
    """Return a configuration by its stable ID."""
    try:
        return CONFIG_BY_ID[config_id]
    except KeyError as exc:
        raise ValueError(f"Unknown configuration ID: {config_id}") from exc


if __name__ == "__main__":
    for config in CONFIGURATIONS:
        print(config)
