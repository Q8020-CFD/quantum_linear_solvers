# Quantum linear solvers
Contains classical and quantum algorithms to solve systems of linear equations such as the HHL algorithm.

## Installation
```python
git clone https://github.com/jw676/quantum_linear_solvers.git
cd quantum_linear_solvers
pip install .

Or use this fork, ported to Qiskit >= 2.2: https://github.com/Q8020-CFD/quantum_linear_solvers
```

This fork (Q8020-CFD/quantum_linear_solvers) requires Qiskit >= 2.2 and uses
only APIs that survive Qiskit 3.0: `phase_estimation()`, `ExactReciprocalGate`,
`PiecewiseChebyshevGate`, and a plain `QuantumCircuit` base for the matrix
classes (no `PhaseEstimation`, `ExactReciprocal`, `PiecewiseChebyshev`, `QFT`
or `BlueprintCircuit`). The phase-estimation step is isolated in
`HHL.qpe_circuit()` so another QPE implementation can be substituted.

## Documentation
This tutorial is based on an older version of this package, which used an older version of Qiskit. The current version of this library requires Qiskit >= 2.2, so these examples won't work as-is. But the gist of the approach remains the same, only the syntax changes.

Tutorial: https://qiskit.org/textbook/ch-applications/hhl_tutorial.html

Other updated examples can be found here: https://github.com/agallojr/wciscc2025