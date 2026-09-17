"""Controlled Aer noise conditions used by the NQComp experiments."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


@dataclass(frozen=True)
class NoiseCondition:
    condition_id: str
    single_qubit_error: float
    two_qubit_error: float
    readout_error: float
    description: str


NOISE_CONDITIONS = {
    "N0": NoiseCondition(
        condition_id="N0",
        single_qubit_error=0.0,
        two_qubit_error=0.0,
        readout_error=0.0,
        description="ideal simulation",
    ),
    "N1": NoiseCondition(
        condition_id="N1",
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        readout_error=0.02,
        description="controlled moderate noise",
    ),
    "N2": NoiseCondition(
        condition_id="N2",
        single_qubit_error=0.005,
        two_qubit_error=0.03,
        readout_error=0.05,
        description="controlled higher noise",
    ),
}


def build_noise_model(condition_id: str) -> NoiseModel | None:
    """Build a deterministic, explicitly parameterized Aer noise model.

    N0 returns None so the simulator remains ideal. N1/N2 add depolarizing
    noise to one- and two-qubit operations plus symmetric readout noise.
    """
    if condition_id not in NOISE_CONDITIONS:
        raise ValueError(f"Unknown noise condition: {condition_id}")

    condition = NOISE_CONDITIONS[condition_id]
    if condition_id == "N0":
        return None

    model = NoiseModel()

    if condition.single_qubit_error > 0:
        error_1q = depolarizing_error(condition.single_qubit_error, 1)
        model.add_all_qubit_quantum_error(error_1q, ["x", "sx"])

    if condition.two_qubit_error > 0:
        error_2q = depolarizing_error(condition.two_qubit_error, 2)
        model.add_all_qubit_quantum_error(error_2q, ["cx"])

    p = condition.readout_error
    if p > 0:
        readout = ReadoutError([[1 - p, p], [p, 1 - p]])
        model.add_all_qubit_readout_error(readout)

    return model
