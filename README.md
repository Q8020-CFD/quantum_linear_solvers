# Quantum linear solvers
Contains classical and quantum algorithms to solve systems of linear equations such as the HHL algorithm.

## Installation
```python
git clone https://github.com/jw676/quantum_linear_solvers.git
cd quantum_linear_solvers
pip install .

Or use this fork, ported to Qiskit 2.x (requires Qiskit >=2.6,<3): https://github.com/Q8020-CFD/quantum_linear_solvers
```

This fork uses only APIs that survive Qiskit 3.0 (`phase_estimation()`,
`ExactReciprocalGate`, `PiecewiseChebyshevGate` and a plain `QuantumCircuit`
base for the matrix classes). The QPE step is isolated in `HHL.qpe_circuit()`
so another QPE can be substituted.

### Supplied spectral bounds

`HHL.construct_circuit` and `HHL.solve` take two optional bounds for the
encoded (Hermitian) matrix:

- `condition_bound`: condition-number bound kappa (>= 1); sizes the clock register.
- `eigenvalue_bounds`: `(lambda_min, lambda_max)` of the absolute eigenvalues;
  set delta and the evolution time.

A supplied value replaces the matrix's own `condition_bounds()` /
`eigs_bounds()` call (for a `NumPyMatrix`, a dense SVD / eigendecomposition).
Supplying the internal values gives the identical circuit. The values used and
their source are recorded in `HHL.spectral_info` and
`LinearSolverResult.spectral_info` (a `SpectralInfo`).

## Documentation
This tutorial is based on an older version of this package, which used an older version of Qiskit. The current version of this library requires Qiskit >=2.6,<3, so these examples won't work as-is. But the gist of the approach remains the same, only the syntax changes.

Tutorial: https://qiskit.org/textbook/ch-applications/hhl_tutorial.html

Other updated examples can be found here: https://github.com/agallojr/wciscc2025