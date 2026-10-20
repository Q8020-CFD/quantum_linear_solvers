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

"""An abstract class for matrices input to the linear systems solvers in Qiskit."""

import copy as _copy
from abc import ABC, abstractmethod
from typing import List, Tuple

from qiskit.circuit import QuantumCircuit, Register


class LinearSystemMatrix(QuantumCircuit, ABC):
    """Base class for linear system matrices.

    A ``QuantumCircuit`` implementing ``exp(i A t)``, plus the hooks HHL uses:
    ``eigs_bounds``, ``condition_bounds`` and ``power`` for phase estimation.
    The body (``power(1)``) is built lazily on first read or mutation and
    dropped by any setter that changes it, as ``BlueprintCircuit`` (removed in
    Qiskit 3.0) did.
    """

    def __init__(
        self,
        num_state_qubits: int,
        tolerance: float,
        evolution_time: float,
        name: str = "ls_matrix",
    ) -> None:
        """
        Args:
            num_state_qubits: the number of qubits where the unitary acts.
            tolerance: the accuracy desired for the approximation
            evolution_time: the time of the Hamiltonian simulation
            name: The name of the object.
        """
        self._is_built = False
        self._num_state_qubits = num_state_qubits
        super().__init__(*self._registers(num_state_qubits), name=name)

        # store parameters
        self.tolerance = tolerance
        self.evolution_time = evolution_time

    @property
    def num_state_qubits(self) -> int:
        r"""The number of state qubits representing the state :math:`|x\rangle`.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the number of state qubits.

        Note that this may change the underlying quantum register, if the number of
        state qubits changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        if num_state_qubits != self._num_state_qubits:
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @property
    def tolerance(self) -> float:
        """Return the error tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """Set the error tolerance
        Args:
            tolerance: The new error tolerance.
        """
        self._tolerance = tolerance
        self._invalidate()

    @property
    def evolution_time(self) -> float:
        """Return the time of the evolution."""
        return self._evolution_time

    @evolution_time.setter
    def evolution_time(self, evolution_time: float) -> None:
        """Set the time of the evolution.

        Args:
            evolution_time: The new time of the evolution.
        """
        self._evolution_time = evolution_time
        self._invalidate()

    @abstractmethod
    def eigs_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        raise NotImplementedError

    @abstractmethod
    def condition_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        raise NotImplementedError

    @abstractmethod
    def _registers(self, num_state_qubits: int) -> List[Register]:
        """The circuit registers for ``num_state_qubits`` state qubits.

        Args:
            num_state_qubits: The number of qubits to represent the matrix.

        Returns:
            The registers, state register first, then any ancilla register.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check that the current configuration allows the circuit body to be built.

        Args:
            raise_on_failure: If True, raise on an invalid configuration, otherwise
                return False.

        Returns:
            True if the configuration is valid.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Replace all registers for ``num_state_qubits`` and drop the circuit body."""
        QuantumCircuit.__init__(
            self, *self._registers(num_state_qubits), name=self.name
        )
        self._is_built = False

    def _invalidate(self) -> None:
        """Drop a built circuit body; it is rebuilt on the next access."""
        # getattr: subclass setters can run before __init__ sets _is_built
        if not getattr(self, "_is_built", False):
            return
        self.clear()
        self.global_phase = 0
        self._is_built = False

    def _build(self) -> None:
        """Build the circuit body, ``power(1)``, once; no-op if already built."""
        if self._is_built:
            return
        self._check_configuration()
        body = self.power(1)
        # Set the flag before composing: compose re-enters the hooks below. On a
        # failed compose drop whatever landed so the next access retries.
        self._is_built = True
        try:
            self.compose(body, inplace=True)
        except BaseException:
            self._invalidate()
            raise

    # QuantumCircuit entry points that read or mutate the body build it first,
    # so a user's own instructions always follow the power(1) body.

    @property
    def data(self):
        self._build()
        return super().data

    @property
    def parameters(self):
        self._build()
        return super().parameters

    @property
    def num_parameters(self) -> int:
        self._build()
        return super().num_parameters

    def _append(self, instruction, *args, **kwargs):
        self._build()
        return super()._append(instruction, *args, **kwargs)

    def compose(self, other, *args, **kwargs):
        self._build()
        return super().compose(other, *args, **kwargs)

    def copy_empty_like(self, name=None, *, vars_mode="alike") -> QuantumCircuit:
        """Empty copy as a plain ``QuantumCircuit`` with the same registers.

        Plain so that helpers which rebuild from an empty copy (``reverse_ops``,
        ``reverse_bits``, ...) do not trigger the lazy build on it.
        """
        cpy = QuantumCircuit(
            *self.qregs, *self.cregs, name=name, global_phase=self.global_phase
        )
        cpy.metadata = _copy.deepcopy(self.metadata)
        return cpy

    def copy(self, name=None):
        """Built copy of the same class (parameters, registers and body)."""
        self._build()
        cpy = _copy.copy(self)
        # Fresh body on the same registers; a shallow copy shares the data list
        QuantumCircuit.__init__(
            cpy, *self.qregs, *self.cregs, name=self.name if name is None else name
        )
        cpy.metadata = _copy.deepcopy(self.metadata)
        cpy.compose(self, inplace=True)
        return cpy

    def __len__(self) -> int:
        self._build()
        return super().__len__()

    def __getitem__(self, item):
        self._build()
        return super().__getitem__(item)

    def decompose(self, gates_to_decompose=None, reps: int = 1) -> QuantumCircuit:
        self._build()
        return super().decompose(gates_to_decompose, reps)

    def draw(self, *args, **kwargs):
        self._build()
        return super().draw(*args, **kwargs)

    def size(self, *args, **kwargs) -> int:
        self._build()
        return super().size(*args, **kwargs)

    def depth(self, *args, **kwargs) -> int:
        self._build()
        return super().depth(*args, **kwargs)

    def count_ops(self):
        self._build()
        return super().count_ops()

    def num_nonlocal_gates(self) -> int:
        self._build()
        return super().num_nonlocal_gates()

    def num_connected_components(self, unitary_only: bool = False) -> int:
        self._build()
        return super().num_connected_components(unitary_only=unitary_only)

    def to_instruction(self, parameter_map=None, label=None):
        self._build()
        return super().to_instruction(parameter_map, label=label)

    def to_gate(self, parameter_map=None, label=None):
        self._build()
        return super().to_gate(parameter_map, label=label)
