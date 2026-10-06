# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
Qudit observables for Hybridlane.

Provides QuditLevelProjector — a single-level projector |ℓ><ℓ| for
measuring populations of individual qudit levels.

On trapped-ion hardware this maps directly to shelving readout:
drive |ℓ> -> bright/dark state, detect fluorescence. No ancilla,
no SQR trick, no approximation.

Usage:
    # Measure P_A = <|A><A|> on qudit wire 'q'
    hqml.expval(QuditLevelProjector(level=1, wires='q'))

This is the correct measurement primitive for qudit-oscillator
vibronic simulations using QCR, QCD, QPS, QT gates.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike


class QuditLevelProjector(Operator):
    r"""Single-level projector observable :math:`\Pi_\ell = |\ell\rangle\langle\ell|`

    Measures the population of a single level :math:`\ell` of a qudit.
    The expectation value gives the probability of finding the qudit in
    level :math:`\ell`:

    .. math::

        \langle \Pi_\ell \rangle = \langle \psi | \ell \rangle\langle \ell | \psi \rangle
        = P(\text{qudit in level } \ell)

    **Trapped-ion hardware mapping:**
    This observable maps directly to shelving readout — drive the
    transition :math:`|\ell\rangle \to |\text{bright}\rangle`, detect
    fluorescence. This is a native measurement on trapped-ion qudit
    processors (e.g. QSCOUT) with no ancilla or decomposition required.

    **Eigenvalues:** 0 (qudit not in level :math:`\ell`) and 1 (qudit in level :math:`\ell`).

    For a 4-level qudit encoding chromophore states
    :math:`\{|G\rangle, |A\rangle, |B\rangle, |C\rangle\}`:

    .. code-block:: python

        # Measure chromophore populations
        P_A = hqml.expval(QuditLevelProjector(level=1, wires='q'))
        P_B = hqml.expval(QuditLevelProjector(level=2, wires='q'))
        P_C = hqml.expval(QuditLevelProjector(level=3, wires='q'))

    Args:
        level (int): qudit level to project onto (:math:`\ell \geq 0`)
        wires (WiresLike): single qudit wire
        dim (int): qudit dimension (default: 4 for chromophore encoding)

    .. note::

        This is a pure qudit observable — no qumode wire involved.
        For qumode Fock-state projectors see
        :class:`~hybridlane.FockStateProjector`.

    .. seealso::

        :class:`~hybridlane.FockStateProjector`
    """

    num_params = 0
    num_wires = 1
    is_hermitian = True

    def __init__(
        self,
        level: int,
        wires: WiresLike,
        dim: int = 4,
        id: str | None = None,
    ):
        if level < 0:
            raise ValueError(f"Qudit level must be >= 0; got {level}")
        if level >= dim:
            raise ValueError(
                f"Level {level} out of range for qudit dimension {dim}"
            )
        self.hyperparameters["level"] = level
        self.hyperparameters["dim"] = dim
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @staticmethod
    def compute_matrix(level: int, dim: int = 4) -> np.ndarray:
        """Return the matrix representation |ℓ><ℓ| in the qudit basis."""
        P = np.zeros((dim, dim), dtype=complex)
        P[level, level] = 1.0
        return P

    def matrix(self, wire_order=None) -> np.ndarray:
        level = self.hyperparameters["level"]
        dim   = self.hyperparameters["dim"]
        return self.compute_matrix(level, dim)

    def eigvals(self) -> np.ndarray:
        """Eigenvalues: 0 for all levels except ℓ which gives 1."""
        dim   = self.hyperparameters["dim"]
        level = self.hyperparameters["level"]
        evals = np.zeros(dim)
        evals[level] = 1.0
        return evals

    @staticmethod
    def compute_diagonalizing_gates(level: int, dim: int, wires: WiresLike):
        """
        Diagonalizing gates: permute |ℓ> to |0> so standard Z measurement works.
        On trapped-ion hardware this is done via shelving — no explicit gate needed.
        Returns empty list (measurement is native).
        """
        return []

    def diagonalizing_gates(self):
        level = self.hyperparameters["level"]
        dim   = self.hyperparameters["dim"]
        return self.compute_diagonalizing_gates(level, dim, self.wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires      = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(hyperparams["level"], wires, dim=hyperparams["dim"])

    def label(self, decimals=None, base_label=None, cache=None):
        level = self.hyperparameters["level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"Π_{{{level}}}",
            cache=cache,
        )

    def __repr__(self):
        level = self.hyperparameters["level"]
        wire  = self.wires[0]
        return f"QuditLevelProjector(level={level}, wire={wire})"


QLP = QuditLevelProjector
r"""Qudit level projector :math:`\Pi_\ell = |\ell\rangle\langle\ell|`

This is an alias for :class:`~hybridlane.QuditLevelProjector`
"""