# SPDX-FileCopyrightText: 2026 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Stinespring-style dissipative channels on a 4-level qudit.

This module provides:

* :class:`QuditFlagFlip` -- a unitary that flips a qubit ``flag`` iff a qudit
  is in a specific level :math:`\\ell`.  Concretely

  .. math::
      U_{\\text{flag}}(\\ell) = (I - |\\ell\\rangle\\langle\\ell|) \\otimes I
        + |\\ell\\rangle\\langle\\ell| \\otimes X .

* :class:`ControlledQuditSwapRG` -- swap :math:`|0\\rangle \\leftrightarrow
  |R\\rangle` on a qudit conditioned on a control qubit being :math:`|1\\rangle`
  (the population-transfer step of qudit amplitude damping).

* :class:`QubitReset` -- mid-circuit |0> reset of a qubit (becomes
  ``qc.reset(qb)`` on the bosonic-qiskit backend).  Together with the unitary
  Stinespring dilation it implements the ancilla-traced channel.

* Helper-function API: :func:`set_flag_R`, :func:`amp_channel_qudit`,
  :func:`dep_channel_qudit` -- compose the above to apply per-level
  amplitude damping / pure dephasing on a qudit chromophore.

The flag-ancilla pattern (use a flag qubit toggled by the qudit level so the
remaining steps use only standard qubit-control primitives) sidesteps the
absence of a generic ``QuditConditioned`` primitive in Hybridlane.

Wire conventions:

* ``qudit_wire``   -- a Qudit-typed wire (dim 4 in this project).
* ``flag_wire``    -- a fresh Qubit wire; reused once :func:`QubitReset` brings
  it back to :math:`|0\\rangle`.
* ``ancilla_wire`` -- a fresh Qubit wire; also reset after the channel.

Both ``flag_wire`` and ``ancilla_wire`` are explicit arguments (the
class-level prescription did not pass them in, but they must come from the
device's wire registry, and we want callers to be able to reuse them across
channel invocations within a Trotter step).
"""
from __future__ import annotations

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import WiresLike

from ..ops.mixins import Hybrid


# ----------------------------------------------------------------------
# Primitive unitary / channel ops
# ----------------------------------------------------------------------


class QuditFlagFlip(Operation, Hybrid):
    r"""Flip a qubit ``flag`` iff a qudit is in level :math:`\ell`.

    .. math::
        U = (I_q - |\ell\rangle\langle\ell|) \otimes I_f
            + |\ell\rangle\langle\ell| \otimes X_f .

    Self-inverse.  Wires are ``(qudit_wire, flag_wire)``.
    """

    num_params = 0
    num_wires = 2
    resource_keys = set()

    def __init__(self, level: int, wires: WiresLike, id: str | None = None):
        if level < 0:
            raise ValueError(f"level must be >= 0; got {level}")
        self.hyperparameters["level"] = level
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit, Qubit  # noqa: PLC0415
        return {self.wires[0]: Qudit(dim=4), self.wires[1]: Qubit()}

    def adjoint(self):
        return QuditFlagFlip(self.hyperparameters["level"], self.wires)

    def pow(self, z):
        # Self-inverse: U^{2k} = I, U^{2k+1} = U
        n = int(round(float(z)))
        if n % 2 == 0:
            return [qml.Identity(self.wires)]
        return [QuditFlagFlip(self.hyperparameters["level"], self.wires)]

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hp = dict(metadata[1])
        return cls(hp["level"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        lv = self.hyperparameters["level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"FlagFlip_{{{lv}}}",
            cache=cache,
        )


class ControlledQuditSwapRG(Operation, Hybrid):
    r"""Swap qudit :math:`|0\rangle \leftrightarrow |R\rangle` iff a control qubit is :math:`|1\rangle`.

    .. math::
        U = |0\rangle\langle 0|_c \otimes I_q
            + |1\rangle\langle 1|_c \otimes \mathrm{SWAP}_{0,R}^{(q)} .

    Self-inverse.  Wires are ``(control_qubit, qudit_wire)``.
    """

    num_params = 0
    num_wires = 2
    resource_keys = set()

    def __init__(self, target_level: int, wires: WiresLike, id: str | None = None):
        if target_level <= 0:
            raise ValueError(f"target_level must be > 0 (cannot swap |0> with |0>); got {target_level}")
        self.hyperparameters["target_level"] = target_level
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit, Qubit  # noqa: PLC0415
        return {self.wires[0]: Qubit(), self.wires[1]: Qudit(dim=4)}

    def adjoint(self):
        return ControlledQuditSwapRG(self.hyperparameters["target_level"], self.wires)

    def pow(self, z):
        n = int(round(float(z)))
        if n % 2 == 0:
            return [qml.Identity(self.wires)]
        return [ControlledQuditSwapRG(self.hyperparameters["target_level"], self.wires)]

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hp = dict(metadata[1])
        return cls(hp["target_level"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        lv = self.hyperparameters["target_level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"CSwap_{{0,{lv}}}",
            cache=cache,
        )


class QubitReset(Operation):
    r"""Mid-circuit reset of a qubit to :math:`|0\rangle`.

    On the bosonic-qiskit backend this is emitted as a literal ``qc.reset(qb)``
    instruction, which qiskit-aer realises by sampling the qubit's measurement
    outcome and flipping when necessary.  Equivalent to ``measure-and-reset``;
    we don't expose the measurement outcome.
    """

    num_params = 0
    num_wires = 1
    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        return cls(wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "|0>",
            cache=cache,
        )


# ----------------------------------------------------------------------
# Helper function API
# ----------------------------------------------------------------------


def set_flag_R(qudit_wire, target_level: int, flag_wire) -> None:
    """Queue a :class:`QuditFlagFlip` so the flag qubit is :math:`|1\\rangle` iff
    the qudit is in level ``target_level``.

    Calling :func:`set_flag_R` a second time with the same arguments is the
    inverse (the gate is self-inverse): use that to "unset" the flag.
    """
    QuditFlagFlip(target_level, wires=[qudit_wire, flag_wire])


def amp_channel_qudit(
    qudit_wire,
    target_level: int,
    flag_wire,
    ancilla_wire,
    theta: float,
) -> None:
    r"""Amplitude damping :math:`|R\rangle \to |0\rangle` on a qudit chromophore.

    Per-call jump probability :math:`p = \sin^2(\theta/2)`; choose
    :math:`\theta = 2 \arcsin(\sqrt{\gamma \, \mathrm{d}t})` for rate
    :math:`\gamma` and Trotter step :math:`\mathrm{d}t`.

    Stinespring sequence::

        set_flag_R(qudit, target_level, flag)      # flag <- 1 iff qudit==R
        CRY(theta, flag -> anc)                    # rotate anc if qudit==R
        set_flag_R(qudit, target_level, flag)      # unflag (self-inverse)
        ControlledQuditSwapRG(target_level,
                               wires=[anc, qudit]) # if anc=1, swap |0><->|R>
        QubitReset(anc); QubitReset(flag)          # return ancillas to |0>

    The CRY+CSwap pair is the standard 2-qubit amplitude-damping dilation,
    here gated by the flag so it only acts on the :math:`|R\rangle` subspace
    of the qudit.  After tracing out the ancilla and flag (achieved by
    :class:`QubitReset`), the qudit evolves as

    .. math::
        \rho \mapsto K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger , \quad
        K_0 = I - (1-\cos(\theta/2)) |R\rangle\langle R| , \quad
        K_1 = \sin(\theta/2) \, |0\rangle\langle R| .
    """
    set_flag_R(qudit_wire, target_level, flag_wire)
    qml.CRY(theta, wires=[flag_wire, ancilla_wire])
    set_flag_R(qudit_wire, target_level, flag_wire)
    ControlledQuditSwapRG(target_level, wires=[ancilla_wire, qudit_wire])
    QubitReset(wires=ancilla_wire)
    QubitReset(wires=flag_wire)


def dep_channel_qudit(
    qudit_wire,
    target_level: int,
    flag_wire,
    ancilla_wire,
    theta: float,
) -> None:
    r"""Pure dephasing on level :math:`R` of a qudit.

    Damps the off-diagonal :math:`|R\rangle\!\langle\neg R|` coherences by a
    factor :math:`\cos\theta` per call (diagonals are unaffected).  Choose
    :math:`\theta = \arccos(1 - \gamma \, \mathrm{d}t)` for rate
    :math:`\gamma` and Trotter step :math:`\mathrm{d}t`.

    Stinespring sequence::

        set_flag_R(qudit, target_level, flag)      # flag <- 1 iff qudit==R
        RY(theta, anc)                             # rotate anc
        CZ(flag, anc)                              # phase-kick if qudit==R
        set_flag_R(qudit, target_level, flag)      # unflag
        QubitReset(anc); QubitReset(flag)          # return ancillas to |0>
    """
    set_flag_R(qudit_wire, target_level, flag_wire)
    qml.RY(theta, wires=ancilla_wire)
    qml.CZ(wires=[flag_wire, ancilla_wire])
    set_flag_R(qudit_wire, target_level, flag_wire)
    QubitReset(wires=ancilla_wire)
    QubitReset(wires=flag_wire)
