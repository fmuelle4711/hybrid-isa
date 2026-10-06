# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest
import pennylane as qml

import hybridlane as hqml


class TestQuditConditionalRotation:
    def test_init(self):
        op = hqml.QuditConditionalRotation(0.5, level=1, wires=[0, 1])
        assert op.name == "QuditConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.hyperparameters["level"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="Qudit level must be >= 0"):
            hqml.QuditConditionalRotation(0.5, level=-1, wires=[0, 1])

    def test_adjoint(self):
        op = hqml.QuditConditionalRotation(0.5, level=2, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.QuditConditionalRotation)
        assert adj_op.parameters[0] == -0.5
        assert adj_op.hyperparameters["level"] == 2

    def test_pow(self):
        op = hqml.QuditConditionalRotation(0.5, level=1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.QuditConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0
        assert pow_op[0].hyperparameters["level"] == 1

    def test_simplify_zero(self):
        op = hqml.QuditConditionalRotation(0, level=1, wires=[0, 1])
        simplified = op.simplify()
        assert isinstance(simplified, qml.Identity)

    def test_simplify_nonzero(self):
        op = hqml.QuditConditionalRotation(0.5, level=1, wires=[0, 1])
        simplified = op.simplify()
        assert isinstance(simplified, hqml.QuditConditionalRotation)
        assert simplified.hyperparameters["level"] == 1

    def test_label(self):
        op = hqml.QuditConditionalRotation(0.5, level=1, wires=[0, 1])
        assert "QCR" in op.label()
        assert "1" in op.label()

    def test_alias(self):
        op = hqml.QCR(0.5, level=1, wires=[0, 1])
        assert isinstance(op, hqml.QuditConditionalRotation)

    def test_unflatten(self):
        op = hqml.QuditConditionalRotation(0.5, level=2, wires=[0, 1])
        data, metadata = op._flatten()
        op2 = hqml.QuditConditionalRotation._unflatten(data, metadata)
        assert op2.parameters == op.parameters
        assert op2.hyperparameters["level"] == op.hyperparameters["level"]


class TestQuditConditionalDisplacement:
    def test_init(self):
        op = hqml.QuditConditionalDisplacement(0.3, 0.1, level=1, wires=[0, 1])
        assert op.name == "QuditConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.3, 0.1]
        assert op.hyperparameters["level"] == 1

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="Qudit level must be >= 0"):
            hqml.QuditConditionalDisplacement(0.3, 0.1, level=-1, wires=[0, 1])

    def test_adjoint(self):
        op = hqml.QuditConditionalDisplacement(0.3, 0.1, level=2, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.QuditConditionalDisplacement)
        assert adj_op.parameters[0] == -0.3
        assert adj_op.hyperparameters["level"] == 2

    def test_pow(self):
        op = hqml.QuditConditionalDisplacement(0.3, 0.1, level=1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.QuditConditionalDisplacement)
        assert pow_op[0].parameters[0] == pytest.approx(0.6)

    def test_simplify_zero(self):
        op = hqml.QuditConditionalDisplacement(0.0, 0.1, level=1, wires=[0, 1])
        simplified = op.simplify()
        assert isinstance(simplified, qml.Identity)

    def test_label(self):
        op = hqml.QuditConditionalDisplacement(0.3, 0.1, level=2, wires=[0, 1])
        assert "QCD" in op.label()
        assert "2" in op.label()

    def test_alias(self):
        op = hqml.QCD(0.3, 0.1, level=1, wires=[0, 1])
        assert isinstance(op, hqml.QuditConditionalDisplacement)

    def test_unflatten(self):
        op = hqml.QuditConditionalDisplacement(0.3, 0.1, level=3, wires=[0, 1])
        data, metadata = op._flatten()
        op2 = hqml.QuditConditionalDisplacement._unflatten(data, metadata)
        assert op2.parameters == op.parameters
        assert op2.hyperparameters["level"] == op.hyperparameters["level"]


class TestQuditPhaseShift:
    def test_init(self):
        op = hqml.QuditPhaseShift(0.5, level=1, wires=[0])
        assert op.name == "QuditPhaseShift"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.hyperparameters["level"] == 1

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="Qudit level must be >= 0"):
            hqml.QuditPhaseShift(0.5, level=-1, wires=[0])

    def test_adjoint(self):
        op = hqml.QuditPhaseShift(0.5, level=1, wires=[0])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.QuditPhaseShift)
        assert adj_op.parameters[0] == -0.5
        assert adj_op.hyperparameters["level"] == 1

    def test_pow(self):
        op = hqml.QuditPhaseShift(0.5, level=1, wires=[0])
        pow_op = op.pow(3)
        assert isinstance(pow_op[0], hqml.QuditPhaseShift)
        assert pow_op[0].parameters[0] == pytest.approx(1.5)

    def test_simplify_zero(self):
        op = hqml.QuditPhaseShift(0.0, level=1, wires=[0])
        simplified = op.simplify()
        assert isinstance(simplified, qml.Identity)

    def test_label(self):
        op = hqml.QuditPhaseShift(0.5, level=2, wires=[0])
        assert "QPS" in op.label()
        assert "2" in op.label()

    def test_alias(self):
        op = hqml.QPS(0.5, level=1, wires=[0])
        assert isinstance(op, hqml.QuditPhaseShift)

    def test_unflatten(self):
        op = hqml.QuditPhaseShift(0.5, level=2, wires=[0])
        data, metadata = op._flatten()
        op2 = hqml.QuditPhaseShift._unflatten(data, metadata)
        assert op2.parameters == op.parameters
        assert op2.hyperparameters["level"] == op.hyperparameters["level"]


class TestQuditTransition:
    def test_init_no_sideband(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=2, wires=[0])
        assert op.name == "QuditTransition"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.hyperparameters["level_i"] == 1
        assert op.hyperparameters["level_j"] == 2
        assert op.hyperparameters["sideband"] is False

    def test_init_sideband(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=2, wires=[0, 1], sideband=True)
        assert op.num_wires == 2
        assert op.hyperparameters["sideband"] is True

    def test_invalid_same_levels(self):
        with pytest.raises(ValueError, match="level_i and level_j must differ"):
            hqml.QuditTransition(0.5, level_i=1, level_j=1, wires=[0])

    def test_invalid_negative_level(self):
        with pytest.raises(ValueError, match="Qudit levels must be >= 0"):
            hqml.QuditTransition(0.5, level_i=-1, level_j=2, wires=[0])

    def test_adjoint(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=2, wires=[0])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.QuditTransition)
        assert adj_op.parameters[0] == -0.5
        assert adj_op.hyperparameters["level_i"] == 1
        assert adj_op.hyperparameters["level_j"] == 2

    def test_pow(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=3, wires=[0])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.QuditTransition)
        assert pow_op[0].parameters[0] == pytest.approx(1.0)

    def test_simplify_zero(self):
        op = hqml.QuditTransition(0.0, level_i=1, level_j=2, wires=[0])
        simplified = op.simplify()
        assert isinstance(simplified, qml.Identity)

    def test_label_no_sideband(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=2, wires=[0])
        label = op.label()
        assert "QT" in label
        assert "1" in label
        assert "2" in label

    def test_label_sideband(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=2, wires=[0, 1], sideband=True)
        label = op.label()
        assert "QTsb" in label

    def test_alias(self):
        op = hqml.QT(0.5, level_i=1, level_j=2, wires=[0])
        assert isinstance(op, hqml.QuditTransition)

    def test_unflatten(self):
        op = hqml.QuditTransition(0.5, level_i=1, level_j=3, wires=[0, 1], sideband=True)
        data, metadata = op._flatten()
        op2 = hqml.QuditTransition._unflatten(data, metadata)
        assert op2.parameters == op.parameters
        assert op2.hyperparameters["level_i"] == op.hyperparameters["level_i"]
        assert op2.hyperparameters["level_j"] == op.hyperparameters["level_j"]
        assert op2.hyperparameters["sideband"] == op.hyperparameters["sideband"]