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

"""Hamiltonian simulation of matrices given as numpy arrays."""

from typing import List, Tuple

import numpy as np
import scipy as sp

from qiskit.circuit import QuantumCircuit, QuantumRegister, Register

from .linear_system_matrix import LinearSystemMatrix


class NumPyMatrix(LinearSystemMatrix):
    """Class of matrices given as a numpy array.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from quantum_linear_solvers.linear_solvers.matrices.numpy_matrix import NumPyMatrix

            matrix = NumPyMatrix(np.array([[1 / 2, 1 / 6, 0, 0], [1 / 6, 1 / 2, 1 / 6, 0],
                               [0, 1 / 6, 1 / 2, 1 / 6], [0, 0, 1 / 6, 1 / 2]]))
            power = 2

            num_qubits = matrix.num_state_qubits
            # Controlled power (as used within QPE)
            pow_circ = matrix.power(power).control(annotated=False)
            circ_qubits = pow_circ.num_qubits
            qc = QuantumCircuit(circ_qubits)
            qc.append(
                matrix.power(power).control(annotated=False), list(range(circ_qubits))
            )
    """

    def __init__(
        self,
        matrix: np.ndarray,
        tolerance: float = 1e-2,
        evolution_time: float = 1.0,
        name: str = "np_matrix",
    ) -> None:
        """
        Args:
            matrix: The matrix defining the linear system problem.
            tolerance: The accuracy desired for the approximation.
            evolution_time: The time of the Hamiltonian simulation.
            name: The name of the object.
        """

        # define internal parameters
        self._num_state_qubits = None
        self._tolerance = None
        self._evolution_time = None  # makes sure the eigenvalues are contained in [0,1)
        self._matrix = None

        super().__init__(
            num_state_qubits=int(np.log2(matrix.shape[0])),
            tolerance=tolerance,
            evolution_time=evolution_time,
            name=name,
        )

        # store parameters
        self.matrix = matrix

    @property
    def matrix(self) -> np.ndarray:
        """Return the matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        """Set the matrix.

        Args:
            matrix: The new matrix.
        """
        self._matrix = matrix
        self._invalidate()

    def eigs_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        matrix_array = self.matrix
        lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        return lambda_min, lambda_max

    def condition_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        matrix_array = self.matrix
        kappa = np.linalg.cond(matrix_array)
        return kappa, kappa

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self.matrix.shape[0] != self.matrix.shape[1]:
            if raise_on_failure:
                raise AttributeError("Input matrix must be square!")
            return False
        if np.log2(self.matrix.shape[0]) % 1 != 0:
            if raise_on_failure:
                raise AttributeError("Input matrix dimension must be 2^n!")
            return False
        if not np.allclose(self.matrix, self.matrix.conj().T):
            if raise_on_failure:
                raise AttributeError("Input matrix must be hermitian!")
            return False

        return valid

    def _registers(self, num_state_qubits: int) -> List[Register]:
        """The state register for ``num_state_qubits`` qubits (no ancillas)."""
        return [QuantumRegister(num_state_qubits, "state")]

    def inverse(self, annotated: bool = False) -> QuantumCircuit:
        return NumPyMatrix(self.matrix, evolution_time=-1 * self.evolution_time)

    def power(self, power: int, matrix_power: bool = False,
        annotated: bool = False) -> QuantumCircuit:
        """Build powers of the circuit.

        Args:
            power: The power to raise this circuit to.
            matrix_power: If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.

        Returns:
            The quantum circuit implementing powers of the unitary.
        """
        qc = QuantumCircuit(self.num_state_qubits)
        evolved = sp.linalg.expm(1j * self.matrix * self.evolution_time)
        # pylint: disable=no-member
        qc.unitary(evolved, qc.qubits)
        return qc.power(power)
