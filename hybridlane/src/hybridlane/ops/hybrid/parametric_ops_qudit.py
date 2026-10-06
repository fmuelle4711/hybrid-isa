# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Hybrid qudit-qumode parametric operations.

GATE NAMES.  The manuscript these operations were built for renames the five
ISA primitives.  The class names, keyword arguments and drawn labels in this
module keep the original short names, so the mapping is recorded here once:

    paper name       class                              short name
    Z_d              QuditPhaseShift                    QPS
    X_E              QuditTransition(sideband=False)    QT
    C_{\Pi_z}D       QuditConditionalDisplacement       QCD
    C_{\Pi}R         QuditConditionalRotation           QCR
    C_{\Pi_x}D       QuditTransition(sideband=True)     QTsb

Note that one class, :class:`QuditTransition`, carries two paper primitives:
with ``sideband=False`` it is the carrier transition :math:`X_E`, and with
``sideband=True`` it is the transition-conditioned displacement
:math:`C_{\Pi_x}D`.  Multitone forms write the transition set as plain
:math:`E`.

No identifier in this module is renamed; the mapping is documentation only.
"""
import math

import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import adjoint_rotation, pow_rotation
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from ..mixins import Hybrid
# Qubit/Qumode imported lazily in __init__ to avoid circular imports


class QuditConditionalRotation(Operation, Hybrid):
    r"""Qudit-level-selective conditional rotation gate :math:`QCR(\theta, \ell)`

    Paper name: :math:`C_{\Pi}R`, the level-conditioned mode rotation.

    This operation implements a phase-space rotation on a qumode, conditioned on
    a specific level :math:`\ell` of a qudit (d-level system). It generalizes
    :class:`~hybridlane.ConditionalRotation` from 2-level qubit control to
    arbitrary qudit level control:

    .. math::

        QCR(\theta, \ell) = \exp\!\left[-i\theta|\ell\rangle\langle\ell| \hat{n}\right]

    .. note::

        There is **no** factor of one half here, unlike
        :class:`~hybridlane.ConditionalRotation`, whose executed form is
        :math:`\exp[-i\frac{\theta}{2}\sigma_z\hat n]`. The bosonic-qiskit
        backend applies ``expm(-1j * theta * kron(projector(level), n_hat))``
        (``devices/bosonic_qiskit/simulate.py``), and this docstring
        previously carried a :math:`1/2` that the code does not. Passing
        :math:`\theta=\chi\tau/2` therefore realizes
        :math:`\exp[-i\chi(\tau/2)\Pi_\ell\hat n]` directly, which is the
        half-step of a lab-frame dispersive term :math:`\chi\Pi_\ell\hat n`.

    where :math:`|\ell\rangle\langle\ell|` is the projector onto level :math:`\ell`
    of the qudit and :math:`\hat{n} = \hat{a}^\dagger \hat{a}` is the number operator
    of the qumode.

    This gate is native to trapped-ion qudit-oscillator hardware with all-to-all
    connectivity. Unlike the qubit-controlled :class:`~hybridlane.ConditionalRotation`,
    no SWAP gates or ancilla qubits are required.

    The ``wires`` attribute is assumed to be ``(qudit, qumode)``.

    .. note::

        The qudit level :math:`\ell` is a non-trainable hyperparameter.
        For a 4-level qudit encoding chromophore states
        :math:`\{|G\rangle, |A\rangle, |B\rangle, |C\rangle\}`,
        use ``level=0,1,2,3`` respectively.

    Args:
        theta (float): rotation angle :math:`\theta`
        level (int): qudit level to condition on (:math:`\ell \geq 0`)
        wires (WiresLike): ``(qudit_wire, qumode_wire)``

    .. seealso::

        :class:`~hybridlane.ConditionalRotation`
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0,)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        level: int,
        wires: WiresLike,
        id: str | None = None,
    ):
        if level < 0:
            raise ValueError(f"Qudit level must be >= 0; got {level}")
        self.hyperparameters["level"] = level
        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit, Qumode
        return {self.wires[0]: Qudit(dim=4), self.wires[1]: Qumode()}

    def adjoint(self):
        theta = self.parameters[0]
        level = self.hyperparameters["level"]
        return QuditConditionalRotation(-theta, level, self.wires)

    def pow(self, z: int | float):
        level = self.hyperparameters["level"]
        return [QuditConditionalRotation(self.data[0] * z, level, self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)
        level = self.hyperparameters["level"]

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return QuditConditionalRotation(theta, level, self.wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(data[0], hyperparams["level"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        level = self.hyperparameters["level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"QCR_{{{level}}}",
            cache=cache,
        )


qml.add_decomps("Adjoint(QuditConditionalRotation)", adjoint_rotation)
qml.add_decomps("Pow(QuditConditionalRotation)", pow_rotation)


# ============================================================
# Decompositions into ConditionalRotation / ConditionalDisplacement
# for bosonicqiskit.hybrid device compatibility.
#
# 2-qubit encoding: |G>=|00>, |A>=|10>, |B>=|01>, |C>=|11>
# qudit wire q is encoded as two qubits q0 (MSB), q1 (LSB)
# where q = wires[0] and mode = wires[1]
#
# Level->qubit mapping:
#   level 1 |A>=|10>: q0=1, q1=0  -> CR on q0
#   level 2 |B>=|01>: q0=0, q1=1  -> CR on q1
#   level 3 |C>=|11>: q0=1, q1=1  -> CR on q0 AND q1 (split half each)
#   all_excited (-1):              -> CR on q0 + CR on q1 (half each)
#
# NOTE: these decompositions assume the qudit wire is the MSB qubit
# in a 2-qubit pair. The LSB qubit wire must be provided via
# the circuit context. For single-wire qudit usage, these decomps
# are approximations — exact qudit gates need native hardware support.
# ============================================================

QCR = QuditConditionalRotation
r"""Qudit-level-selective conditional rotation (QCR) gate

.. math::

    QCR(\theta, \ell) = \exp\!\left[-i\frac{\theta}{2}|\ell\rangle\langle\ell|\hat{n}\right]

This is an alias for :class:`~hybridlane.QuditConditionalRotation`
"""


CPiR = QuditConditionalRotation
r"""Paper name for :class:`~hybridlane.QuditConditionalRotation`: :math:`C_{\Pi}R`, the level-conditioned mode rotation (short name ``QCR``)."""


class QuditConditionalDisplacement(Operation, Hybrid):
    r"""Qudit-level-selective conditional displacement gate :math:`QCD(\alpha, \ell)`

    Paper name: :math:`C_{\Pi_z}D`, the level-conditioned displacement.

    This operation implements a displacement on a qumode conditioned on a specific
    level :math:`\ell` of a qudit. It generalizes
    :class:`~hybridlane.ConditionalDisplacement` from 2-level qubit control to
    arbitrary qudit level control:

    .. math::

        QCD(\alpha, \ell) = \exp\!\left[|\ell\rangle\langle\ell|
            \otimes (\alpha \hat{a}^\dagger - \alpha^* \hat{a})\right]

    where :math:`\alpha = ae^{i\phi} \in \mathbb{C}`.

    On trapped-ion hardware with all-to-all connectivity, this gate is native and
    requires no SWAP gates or ancilla, unlike cQED implementations.

    The ``wires`` attribute is assumed to be ``(qudit, qumode)``.

    Args:
        a (float): displacement magnitude :math:`|\alpha|`
        phi (float): displacement phase :math:`\arg(\alpha)`
        level (int): qudit level to condition on (:math:`\ell \geq 0`)
        wires (WiresLike): ``(qudit_wire, qumode_wire)``

    .. seealso::

        :class:`~hybridlane.ConditionalDisplacement`
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        level: int,
        wires: WiresLike,
        id: str | None = None,
    ):
        if level < 0:
            raise ValueError(f"Qudit level must be >= 0; got {level}")
        self.hyperparameters["level"] = level
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit, Qumode
        return {self.wires[0]: Qudit(dim=4), self.wires[1]: Qumode()}

    def adjoint(self):
        level = self.hyperparameters["level"]
        return QuditConditionalDisplacement(-self.data[0], self.data[1], level, self.wires)

    def pow(self, z: int | float):
        a, phi = self.data
        level = self.hyperparameters["level"]
        return [QuditConditionalDisplacement(a * z, phi, level, self.wires)]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)
        level = self.hyperparameters["level"]

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return QuditConditionalDisplacement(a, phi, level, self.wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(data[0], data[1], hyperparams["level"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        level = self.hyperparameters["level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"QCD_{{{level}}}",
            cache=cache,
        )


@qml.register_resources({QuditConditionalDisplacement: 1})
def _adjoint_qcd(a, phi, wires, level, **_):
    QuditConditionalDisplacement(a, phi + math.pi, level, wires=wires)


@qml.register_resources({QuditConditionalDisplacement: 1})
def _pow_qcd(a, phi, wires, z, level, **_):
    QuditConditionalDisplacement(z * a, phi, level, wires=wires)


qml.add_decomps("Adjoint(QuditConditionalDisplacement)", _adjoint_qcd)
qml.add_decomps("Pow(QuditConditionalDisplacement)", _pow_qcd)

QCD = QuditConditionalDisplacement
r"""Qudit-level-selective conditional displacement (QCD) gate

.. math::

    QCD(\alpha, \ell) = \exp\!\left[|\ell\rangle\langle\ell|
        \otimes (\alpha\hat{a}^\dagger - \alpha^*\hat{a})\right]

This is an alias for :class:`~hybridlane.QuditConditionalDisplacement`
"""


CPizD = QuditConditionalDisplacement
r"""Paper name for :class:`~hybridlane.QuditConditionalDisplacement`: :math:`C_{\Pi_z}D`, the level-conditioned displacement (short name ``QCD``)."""


class QuditPhaseShift(Operation):
    r"""Qudit single-level phase shift gate :math:`QPS(\theta, \ell)`

    Paper name: :math:`Z_d`, the qudit phase shift.

    This gate imparts a phase :math:`e^{-i\theta}` to a single level :math:`\ell`
    of a qudit, leaving all other levels unchanged:

    .. math::

        QPS(\theta, \ell) = \exp\!\left[-i\theta|\ell\rangle\langle\ell|\right]
        = \mathbb{I} + (e^{-i\theta} - 1)|\ell\rangle\langle\ell|

    This is a pure qudit gate with no qumode involved. It is used to implement
    energy detuning terms (e.g. :math:`\delta_{AB}|A\rangle\langle A|`) in
    vibronic Hamiltonian simulations.

    Args:
        theta (float): phase angle :math:`\theta`
        level (int): qudit level to dephase (:math:`\ell \geq 0`)
        wires (WiresLike): single qudit wire

    .. note::

        This is a pure qudit gate (no qumode). It acts on a single wire.
        The qudit wire is treated as a Qubit in Hybridlane's wire type system.
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        level: int,
        wires: WiresLike,
        id: str | None = None,
    ):
        if level < 0:
            raise ValueError(f"Qudit level must be >= 0; got {level}")
        self.hyperparameters["level"] = level
        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit
        return {self.wires[0]: Qudit(dim=4)}

    def adjoint(self):
        theta = self.parameters[0]
        level = self.hyperparameters["level"]
        return QuditPhaseShift(-theta, level, self.wires)

    def pow(self, z: int | float):
        level = self.hyperparameters["level"]
        return [QuditPhaseShift(self.data[0] * z, level, self.wires)]

    def simplify(self):
        theta = self.data[0] % (2 * math.pi)
        level = self.hyperparameters["level"]

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return QuditPhaseShift(theta, level, self.wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(data[0], hyperparams["level"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        level = self.hyperparameters["level"]
        return super().label(
            decimals=decimals,
            base_label=base_label or f"QPS_{{{level}}}",
            cache=cache,
        )


qml.add_decomps("Adjoint(QuditPhaseShift)", adjoint_rotation)
qml.add_decomps("Pow(QuditPhaseShift)", pow_rotation)

QPS = QuditPhaseShift
r"""Qudit single-level phase shift (QPS) gate

.. math::

    QPS(\theta, \ell) = \exp[-i\theta|\ell\rangle\langle\ell|]

This is an alias for :class:`~hybridlane.QuditPhaseShift`
"""


Zd = QuditPhaseShift
r"""Paper name for :class:`~hybridlane.QuditPhaseShift`: :math:`Z_d`, the qudit phase shift (short name ``QPS``)."""


class QuditTransition(Operation, Hybrid):
    r"""Qudit level-transition gate :math:`QT(\theta, \ell_i, \ell_j)` with optional sideband

    Paper names: with ``sideband=False`` this is :math:`X_E`, the qudit
    transition driven on the carrier; with ``sideband=True`` it is
    :math:`C_{\Pi_x}D`, the transition-conditioned displacement.  Multitone
    forms write the transition set as plain :math:`E`.

    This gate drives coherent transitions between two levels :math:`\ell_i` and
    :math:`\ell_j` of a qudit, with an optional coupling to a qumode sideband.

    **Pure qudit transition** (no qumode, ``sideband=False``):

    .. math::

        QT(\theta, \ell_i, \ell_j)
        = \exp\!\left[-i\theta(|\ell_i\rangle\langle\ell_j|
          + |\ell_j\rangle\langle\ell_i|)\right]

    **Sideband transition** (with qumode, ``sideband=True``):

    .. math::

        QT(\theta, \ell_i, \ell_j)
        = \exp\!\left[-i\theta(|\ell_i\rangle\langle\ell_j|
          + |\ell_j\rangle\langle\ell_i|)
          \otimes(\hat{a}+\hat{a}^\dagger)\right]

    For a 4-level qudit encoding chromophore states
    :math:`\{|G\rangle, |A\rangle, |B\rangle, |C\rangle\}`,
    this gate implements:

    - ``(level_i=1, level_j=2)``: :math:`|A\rangle\leftrightarrow|B\rangle`
      (replaces IsingXY / RXX+RYY)
    - ``(level_i=1, level_j=3)``: :math:`|A\rangle\leftrightarrow|C\rangle`
      (replaces CRX)
    - With ``sideband=True``: sideband coupling terms :math:`\eta(l+l^\dagger)\Sigma_{AB}`

    Args:
        theta (float): coupling angle :math:`\theta`
        level_i (int): first qudit level (:math:`\ell_i \geq 0`)
        level_j (int): second qudit level (:math:`\ell_j \geq 0`, :math:`\ell_j \neq \ell_i`)
        wires (WiresLike): ``[qudit_wire]`` if ``sideband=False``,
            ``[qudit_wire, qumode_wire]`` if ``sideband=True``
        sideband (bool): if ``True``, couples to qumode :math:`(\hat{a}+\hat{a}^\dagger)`
    """

    num_params = 1
    ndim_params = (0,)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        level_i: int,
        level_j: int,
        wires: WiresLike,
        sideband: bool = False,
        id: str | None = None,
    ):
        if level_i < 0 or level_j < 0:
            raise ValueError(f"Qudit levels must be >= 0; got {level_i}, {level_j}")
        if level_i == level_j:
            raise ValueError(f"level_i and level_j must differ; got {level_i} == {level_j}")

        self.hyperparameters["level_i"] = level_i
        self.hyperparameters["level_j"] = level_j
        self.hyperparameters["sideband"] = sideband

        self.num_wires = 2 if sideband else 1
        # type_signature is used by Hybridlane wire inference (instead of num_qumodes)
        # qudit wire = Qubit, qumode wire = Qumode
        # Lazy import to avoid circular import through sa/__init__.py
        if sideband:
            from hybridlane.sa.base import Qubit, Qumode  # noqa: PLC0415
            self.type_signature = [Qubit(), Qumode()]
        else:
            self.type_signature = None
            self.num_qumodes = None  # pure qudit gate — no qumode

        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def wire_types(self):
        from hybridlane.sa.base import Qudit, Qumode
        sb = self.hyperparameters["sideband"]
        if sb:
            return {self.wires[0]: Qudit(dim=4), self.wires[1]: Qumode()}
        else:
            return {self.wires[0]: Qudit(dim=4)}

    def adjoint(self):
        li = self.hyperparameters["level_i"]
        lj = self.hyperparameters["level_j"]
        sb = self.hyperparameters["sideband"]
        return QuditTransition(-self.data[0], li, lj, self.wires, sideband=sb)

    def pow(self, z: int | float):
        li = self.hyperparameters["level_i"]
        lj = self.hyperparameters["level_j"]
        sb = self.hyperparameters["sideband"]
        return [QuditTransition(self.data[0] * z, li, lj, self.wires, sideband=sb)]

    def simplify(self):
        theta = self.data[0] % (2 * math.pi)
        li = self.hyperparameters["level_i"]
        lj = self.hyperparameters["level_j"]
        sb = self.hyperparameters["sideband"]

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return QuditTransition(theta, li, lj, self.wires, sideband=sb)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(
            data[0],
            hyperparams["level_i"],
            hyperparams["level_j"],
            wires,
            sideband=hyperparams["sideband"],
        )

    def label(self, decimals=None, base_label=None, cache=None):
        li = self.hyperparameters["level_i"]
        lj = self.hyperparameters["level_j"]
        sb = self.hyperparameters["sideband"]
        tag = "QTsb" if sb else "QT"
        return super().label(
            decimals=decimals,
            base_label=base_label or f"{tag}_{{{li},{lj}}}",
            cache=cache,
        )


qml.add_decomps("Adjoint(QuditTransition)", adjoint_rotation)
qml.add_decomps("Pow(QuditTransition)", pow_rotation)

QT = QuditTransition
r"""Qudit level-transition (QT) gate

.. math::

    QT(\theta, \ell_i, \ell_j)
    = \exp[-i\theta(|\ell_i\rangle\langle\ell_j| + |\ell_j\rangle\langle\ell_i|)]

This is an alias for :class:`~hybridlane.QuditTransition`
"""


XE = QuditTransition
r"""Paper name for :class:`~hybridlane.QuditTransition` driven on the carrier: :math:`X_E`, the qudit transition (short name ``QT``); pass ``sideband=False``."""

CPixD = QuditTransition
r"""Paper name for :class:`~hybridlane.QuditTransition` driven on the sidebands: :math:`C_{\Pi_x}D`, the transition-conditioned displacement (short name ``QTsb``); pass ``sideband=True``.

.. warning::

    ``XE`` and ``CPixD`` are the SAME class.  The paper distinguishes the two
    primitives, the class distinguishes them by the ``sideband`` keyword, so
    ``XE is CPixD`` is ``True`` and neither alias sets the flag for you.
"""


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )