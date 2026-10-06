# SPDX-FileCopyrightText: 2026 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""Hybridlane channels: dissipative primitives for qudit-qumode circuits.

Currently exposes Stinespring-style amplitude-damping and pure-dephasing
channels on a 4-level qudit via a flag-qubit + ancilla-qubit dilation
(see :mod:`hybridlane.channels.stinespring_qudit`).
"""

from .stinespring_qudit import (
    ControlledQuditSwapRG,
    QubitReset,
    QuditFlagFlip,
    amp_channel_qudit,
    dep_channel_qudit,
    set_flag_R,
)

__all__ = [
    "ControlledQuditSwapRG",
    "QubitReset",
    "QuditFlagFlip",
    "amp_channel_qudit",
    "dep_channel_qudit",
    "set_flag_R",
]
