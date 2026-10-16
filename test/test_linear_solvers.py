# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the quantum linear system solver algorithm."""

import unittest
import warnings

from scipy.linalg import expm
import numpy as np
from ddt import ddt, idata, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ExactReciprocalGate
from qiskit.quantum_info import Operator, partial_trace, Statevector
from qiskit.transpiler.passes import RemoveResetInZeroState


from linear_solvers.hhl import HHL
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
from linear_solvers.observables.absolute_average import AbsoluteAverage
from linear_solvers.observables.matrix_functional import MatrixFunctional

@ddt
class TestMatrices(unittest.TestCase):
    """
    Tests based on the matrices classes.
    """

    @idata(
        [
            [TridiagonalToeplitz(2, 1, -1 / 3)],        # matrix test 1
            [TridiagonalToeplitz(3, 2, 1), 1.1, 3],     #        test 2
            [                                           #        test 3
                NumPyMatrix(
                    np.array(
                        [
                            [1 / 2, 1 / 6, 0, 0],
                            [1 / 6, 1 / 2, 1 / 6, 0],
                            [0, 1 / 6, 1 / 2, 1 / 6],
                            [0, 0, 1 / 6, 1 / 2],
                        ]
                    )
                )
            ],
        ]
    )
    @unpack
    def test_matrices(self, matrix, time=1.0, power=1):
        """Test the different matrix classes."""
        matrix.evolution_time = time

        num_qubits = matrix.num_state_qubits
        pow_circ = matrix.power(power).control(annotated=False)
        circ_qubits = pow_circ.num_qubits

        qc = QuantumCircuit(circ_qubits)
        qc.append(
            matrix.power(power).control(annotated=False), list(range(circ_qubits))
        )

        # extract the parts of the circuit matrix corresponding to TridiagonalToeplitz
        zero_op = (Operator.from_label("I") + Operator.from_label("Z")) / 2
        one_op = (Operator.from_label("I") - Operator.from_label("Z")) / 2

        # Construct projection operator directly with the right dimensions
        identity_op = Operator.from_label("I")
        # Create a projection operator of the exact same dimension as the circuit
        proj_ops = []
        for i in range(circ_qubits - 1):
            proj_ops.append(zero_op if i < pow_circ.num_ancillas else identity_op)
        proj_ops.append(one_op)  # The last position gets the |1><1| projector

        # Combine all operators with tensor products
        proj_op = proj_ops[0]
        for op in proj_ops[1:]:
            proj_op = proj_op.tensor(op)
        proj = proj_op.data

        circ_matrix = Operator(qc).data
        approx_exp = partial_trace(
            np.dot(proj, circ_matrix), [0] + list(range(num_qubits + 1, circ_qubits))
        ).data

        exact_exp = expm(1j * matrix.evolution_time * power * matrix.matrix)
        np.testing.assert_array_almost_equal(approx_exp, exact_exp, decimal=2)

    @idata(
        [
            [TridiagonalToeplitz(2, 1.5, 2.5)],
            [TridiagonalToeplitz(4, -1, 1.6)],
        ]
    )
    @unpack
    def test_eigs_bounds(self, matrix):
        """Test the capability of TridiagonalToeplitz matrix class
        to find accurate absolute eigenvalues bounds."""

        matrix_lambda_min, matrix_lambda_max = matrix.eigs_bounds()

        numpy_matrix = matrix.matrix
        eigenvalues, _ = np.linalg.eig(numpy_matrix)
        abs_eigenvalues = np.abs(eigenvalues)
        exact_lambda_min = np.min(abs_eigenvalues)
        exact_lambda_max = np.max(abs_eigenvalues)

        np.testing.assert_almost_equal(matrix_lambda_min, exact_lambda_min, decimal=6)
        np.testing.assert_almost_equal(matrix_lambda_max, exact_lambda_max, decimal=6)


@ddt
class TestObservables(unittest.TestCase):
    """
    Tests based on the observables classes.
    """

    @idata(
        [
            [AbsoluteAverage(), [1.0, -2.1, 3.2, -4.3]],
            [AbsoluteAverage(), [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12]],
        ]
    )
    @unpack
    def test_absolute_average(self, observable, vector):
        """Test the absolute average observable."""
        init_state = vector / np.linalg.norm(vector)
        num_qubits = int(np.log2(len(vector)))

        qc = QuantumCircuit(num_qubits)
        qc.initialize(init_state, list(range(num_qubits)))
        qc.append(observable.observable_circuit(num_qubits), list(range(num_qubits)))

        # Observable operator
        observable_op = observable.observable(num_qubits)
        # Create the statevector from the quantum circuit
        circuit_statevector = Statevector.from_instruction(qc)
        # Calculate the expectation value
        state_vec = circuit_statevector.expectation_value(observable_op)

        # Obtain result
        result = observable.post_processing(state_vec, num_qubits)

        # Obtain analytical evaluation
        exact = observable.evaluate_classically(init_state)

        np.testing.assert_almost_equal(result, exact, decimal=2)

    @idata(
        [
            [MatrixFunctional(1, -1 / 3), [1.0, -2.1, 3.2, -4.3]],
            [
                MatrixFunctional(2 / 3, 11 / 7),
                [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12],
            ],
        ]
    )
    @unpack
    def test_matrix_functional(self, observable, vector):
        """Test the matrix functional class."""

        tpass = RemoveResetInZeroState()
        init_state = vector / np.linalg.norm(vector)
        num_qubits = int(np.log2(len(vector)))

        # Get observable circuits
        obs_circuits = observable.observable_circuit(num_qubits)

        qcs = []
        for obs_circ in obs_circuits:
            qc = QuantumCircuit(num_qubits)
            qc.initialize(init_state, list(range(num_qubits)))
            qc.append(obs_circ, list(range(num_qubits)))
            qcs.append(tpass(qc.decompose()))

        # Get observables
        observable_ops = observable.observable(num_qubits)
        state_vecs = []

        # First is the norm
        state_vecs.append(Statevector.from_instruction(qcs[0]).
            expectation_value(observable_ops[0]))
        for i in range(1, len(observable_ops), 2):
            state_vecs.append(Statevector.from_instruction(qcs[i]).
                expectation_value(observable_ops[i]))
            state_vecs.append(Statevector.from_instruction(qcs[i + 1]).
                expectation_value(observable_ops[i + 1]))

        # Obtain result
        result = observable.post_processing(state_vecs, num_qubits)

        # Obtain analytical evaluation
        exact = observable.evaluate_classically(init_state)

        np.testing.assert_almost_equal(result, exact, decimal=2)


@ddt
class TestReciprocal(unittest.TestCase):
    """
    Tests based on the reciprocal classes.
    """

    @idata([[2, 0.1, False], [3, 1 / 9, True]])
    @unpack
    def test_exact_reciprocal(self, num_qubits, scaling, neg_vals):
        """Test the ExactReciprocalGate class."""
        reciprocal = ExactReciprocalGate(num_qubits + neg_vals, scaling, neg_vals)

        qc = QuantumCircuit(num_qubits + 1 + neg_vals)
        qc.h(list(range(num_qubits)))
        # If negative eigenvalues, set the sign qubit to 1
        if neg_vals:
            qc.x(num_qubits)
        qc.append(reciprocal, list(range(num_qubits + 1 + neg_vals)))

        # Create the operator 0
        state_vec = Statevector.from_instruction(qc).data[
            -(2**num_qubits) :
        ]

        # Remove the factor from the hadamards
        state_vec *= np.sqrt(2) ** num_qubits

        # Analytic value
        exact = []
        for i in range(0, 2**num_qubits):
            if i == 0:
                exact.append(0)
            else:
                if neg_vals:
                    exact.append(-scaling / (1 - i / (2**num_qubits)))
                else:
                    exact.append(scaling * (2**num_qubits) / i)

        np.testing.assert_array_almost_equal(state_vec, exact, decimal=2)

    @idata([[4, 0.25, 2.6, 2], [5, 0.125, 6.0, 3]])
    @unpack
    def test_chebyshev_reciprocal_matches_legacy(self, nl, constant, kappa, nb):
        """The PiecewiseChebyshevGate reciprocal reproduces the former
        PiecewiseChebyshev circuit (saturation outside [a, 2^nl - 1] included):
        same rotation amplitudes for every clock-register value."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from qiskit.circuit.library.arithmetic.piecewise_chebyshev import (
                    PiecewiseChebyshev,
                )
        except ImportError:  # Qiskit >= 3.0: the legacy class is gone
            self.skipTest("legacy PiecewiseChebyshev not available")

        gate = HHL()._chebyshev_reciprocal(nl, constant, kappa, nb)
        user_breakpoints = [b for b in gate.breakpoints if 0 < b < 2**nl]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), gate.degree, user_breakpoints, nl
            )

        def target_amplitudes(operation):
            # uniform superposition on the clock qubits, target and ancillas |0>
            qc = QuantumCircuit(operation.num_qubits)
            qc.h(range(nl))
            qc.append(operation, qc.qubits)
            probs = Statevector(qc).probabilities(list(range(nl + 1)))
            return np.sqrt(probs.reshape(2, 2**nl)[1] * 2**nl)

        np.testing.assert_allclose(
            target_amplitudes(gate), target_amplitudes(legacy), atol=1e-10
        )


@ddt
class TestLinearSolver(unittest.TestCase):
    """
    Tests based on the linear solvers classes.
    These are rollup-tests which use some of the classes separatelty tested above.
    """

    @idata(
        [
            [
                TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2),
                [1.0, -2.1, 3.2, -4.3],
                MatrixFunctional(1, 1 / 2),
            ],
            [
                np.array(
                    [
                        [0, 0, 1.585, 0],
                        [0, 0, -0.585, 1],
                        [1.585, -0.585, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
                [1.0, 0, 0, 0],
                MatrixFunctional(1, 1 / 2),
            ],
            [
                [
                    [1 / 2, 1 / 6, 0, 0],
                    [1 / 6, 1 / 2, 1 / 6, 0],
                    [0, 1 / 6, 1 / 2, 1 / 6],
                    [0, 0, 1 / 6, 1 / 2],
                ],
                [1.0, -2.1, 3.2, -4.3],
                MatrixFunctional(1, 1 / 2),
            ],
            [
                np.array([[82, 34], [34, 58]]),
                np.array([[1], [0]]),
                AbsoluteAverage(),
                3,
            ],
            [
                TridiagonalToeplitz(3, 1, -1 / 2, trotter_steps=2),
                [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12],
                AbsoluteAverage(),
            ],
            [
                TridiagonalToeplitz(2, 2, -0.5, trotter_steps=3),
                [0.5, 0.5, 0.5, 0.5],
                AbsoluteAverage(),
                2,
            ],
            [
                NumPyMatrix(np.array([[3, 1], [1, 3]])),
                [1.0, 1.0],
                MatrixFunctional(1, 1),
                2,
            ],
            [
                np.array([
                    [4, 1, 0, 0],
                    [1, 4, 1, 0],
                    [0, 1, 4, 1],
                    [0, 0, 1, 4]
                ]),
                [1.0, 0.0, 0.0, 0.0],
                AbsoluteAverage(),
                2,
            ],
            [
                TridiagonalToeplitz(2, 1, 0, trotter_steps=2),  # Diagonal matrix
                [1.0, 2.0, 3.0, 4.0],
                MatrixFunctional(0.5, 0.5),
                2,
            ],
            [
                np.array([
                    [2, -1, 0, 0],
                    [-1, 2, -1, 0],
                    [0, -1, 2, -1],
                    [0, 0, -1, 2]
                ]),  # Standard discretization of 1D Poisson equation
                [0.0, 1.0, 0.0, 0.0],
                AbsoluteAverage(),
                2,
            ],
        ]
    )
    @unpack
    def test_hhl(self, matrix, right_hand_side, observable, decimal=1):
        """Test the HHL class."""
        num_qubits = 0
        if isinstance(matrix, QuantumCircuit):
            num_qubits = matrix.num_state_qubits
        elif isinstance(matrix, (np.ndarray)):
            num_qubits = int(np.log2(matrix.shape[0]))
        elif isinstance(matrix, list):
            num_qubits = int(np.log2(len(matrix)))

        rhs = right_hand_side / np.linalg.norm(right_hand_side)

        # Ensure rhs is a 1D array for initialize
        if isinstance(rhs, np.ndarray) and len(rhs.shape) > 1:
            rhs = rhs.flatten()

        # Initial state circuit
        qc = QuantumCircuit(num_qubits)
        qc.initialize(rhs, list(range(num_qubits)))

        hhl = HHL()
        solution = hhl.solve(matrix, qc, observable)
        approx_result = solution.observable

        # Calculate analytical value
        exact_x = None
        if isinstance(matrix, QuantumCircuit):
            exact_x = np.dot(np.linalg.inv(matrix.matrix), rhs)
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)
            exact_x = np.dot(np.linalg.inv(matrix), rhs)
        exact_result = observable.evaluate_classically(exact_x)

        np.testing.assert_almost_equal(approx_result, exact_result, decimal=decimal)

    @idata(
        [
            # Clock values 2.0 to 6.7 with a = 6: one eigenvalue is fitted, the
            # three below a are saturated. Measured 0.9897 (0.9948 unsquared).
            [
                TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2),
                [1.0, -2.1, 3.2, -4.3],
                0.98,
            ],
            # Clock values 2.0 to 4.7, all below a = 6: every eigenvalue is
            # saturated, so the approximation returns ~b (measured 0.5756). This
            # is the legacy PiecewiseChebyshev behaviour, reproduced exactly (see
            # TestReciprocal.test_chebyshev_reciprocal_matches_legacy); the row
            # pins it so a change of the saturation shows up here.
            [
                np.array([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]]),
                [1.0, 0.0, 0.0, 0.0],
                0.5,
            ],
        ]
    )
    @unpack
    def test_hhl_chebyshev_reciprocal(self, matrix, right_hand_side, min_fidelity):
        """HHL with the piecewise Chebyshev reciprocal (``exact_reciprocal=False``):
        the solution read off the flag qubit reaches the documented fidelity with
        the classical solution. The approximation fits arcsin(delta / x) only for
        clock values x >= a = round((2^nl)^(2/3)) and saturates below, so its
        quality depends on where the eigenvalues land; the exact reciprocal gives
        >= 0.999 on both rows."""
        if isinstance(matrix, QuantumCircuit):
            exact_matrix = matrix.matrix
        else:
            exact_matrix = np.array(matrix)
        rhs = np.array(right_hand_side) / np.linalg.norm(right_hand_side)
        nb = int(np.log2(len(rhs)))

        hhl = HHL(exact_reciprocal=False)
        self.assertFalse(hhl._exact_reciprocal)
        solution = hhl.solve(matrix, rhs)

        # Solution amplitudes: flag qubit (last) |1>, all other work qubits |0>
        state = Statevector(solution.state).data
        start = len(state) // 2
        hhl_x = np.real(state[start : start + 2**nb])
        exact_x = np.linalg.solve(exact_matrix, rhs)
        overlap = abs(np.vdot(hhl_x, exact_x))
        fidelity = (overlap / np.linalg.norm(hhl_x) / np.linalg.norm(exact_x)) ** 2
        self.assertGreaterEqual(fidelity, min_fidelity, f"fidelity {fidelity}")


if __name__ == "__main__":

    # unittest.main(defaultTest="TestMatrices")
    # unittest.main(defaultTest="TestObservables")
    # unittest.main(defaultTest="TestReciprocal")
    # unittest.main(defaultTest="TestLinearSolver")
    unittest.main()  # do them all
