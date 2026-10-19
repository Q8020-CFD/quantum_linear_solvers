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

### Supplied spectral bounds

`HHL.construct_circuit` and `HHL.solve` take two optional keyword arguments
for the matrix actually encoded (the Hermitian matrix handed to HHL):

- `condition_bound`: a condition-number bound kappa (finite, >= 1). It sizes
  the clock register, `nl = max(nb + 1, ceil(log2(kappa + 1))) + neg_vals`.
- `eigenvalue_bounds`: `(lambda_min, lambda_max)` of the absolute
  eigenvalues (finite, `0 <= lambda_min <= lambda_max`, `lambda_max > 0`).
  They set the reciprocal constant delta and the evolution time.

When a value is supplied, the matrix circuit's own `condition_bounds()` or
`eigs_bounds()` is not called. For a `NumPyMatrix` those are a full dense
SVD (`np.linalg.cond`) and a full eigendecomposition (`np.linalg.eigvals`).
With neither argument the circuit is the one built before (each internal
method is now called once, not twice, with the same values). Supplying
exactly the internal values gives the identical circuit.

After each build, `HHL.spectral_info` (and `LinearSolverResult.spectral_info`
from `solve`) is a `linear_solvers.SpectralInfo` with the values used
(`kappa`, `lambda_min` after the near-zero substitution, `lambda_max`),
their sources (`kappa_source`, `eigs_source`: `internal`, `supplied` or
`none`), the combined `source` (`internal`, `supplied` or `mixed`) and
`wall_s`, the wall time of the internal bound calls (0.0 when both are
supplied). Tests: `test/test_linear_solvers.py`, class
`TestSuppliedSpectrum`.

## Documentation
This tutorial is based on an older version of this package, which used an older version of Qiskit. The current version of this library requires Qiskit >= 2.2, so these examples won't work as-is. But the gist of the approach remains the same, only the syntax changes.

Tutorial: https://qiskit.org/textbook/ch-applications/hhl_tutorial.html

Other updated examples can be found here: https://github.com/agallojr/wciscc2025