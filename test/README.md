# Lorentzian–Landau falsification sweep: run 1 and corrected run 2

## Corrected run 2 (current default)

Run in the pinned Python 3.12 container; never install the numerical
dependencies on the host:

```sh
podman run --rm -v "$PWD":/work:z docker.io/library/python:3.12-slim bash -c "pip install -q 'sympy==1.14.*' 'numpy==2.*' && python /work/test/sweep.py --seed 20260811"
```

The current script writes `test/results-run2-seed20260811.json`. It never writes
the frozen run-1 artifact `test/results-seed20260811.json`.

Run 2 corrects two decisive methodological errors in run 1:

- Experiment A now draws conserved Euclidean three-vectors and constructs all
  invariants from them, guaranteeing a negative-semidefinite external Gram
  matrix. Negativity of independent invariants alone was not sufficient.
- Cubic forms now use the exact coordinate-partial test: every Hessian of
  `dF/dx_i` must have at most one positive eigenvalue. The 64 seeded interior
  directions are retained only as a supplementary K-Lorentzian probe.

Run 2 also uses nondegenerate four-point path masses, an explicit Euclidean
path start, exhaustive candidate catalogs, symbolic sunrise and Källén/Gram
checks, M-convex support assertions, and full per-sample logging.

## Historical run 1 (frozen audit artifact)

`test/results-seed20260811.json` was produced by the pre-correction version of
`sweep.py` using this exact command:

```sh
podman run --rm -v "$PWD":/work:z docker.io/library/python:3.12-slim bash -c "pip install -q 'sympy==1.14.*' 'numpy==2.*' && python /work/test/sweep.py --seed 20260811"
```

The pre-correction `sweep.py` is **not preserved in Git**: the repository's
`lead-*/` ignore rule excludes this entire lead directory, so that source file
was never tracked and has no repository history. The available run-1 audit
trail is the frozen JSON itself, including its embedded conventions, constants,
and methodology fields, together with the orchestrator's `ADDENDUMS.md` log
entry. The frozen JSON must not be regenerated with the corrected script.

Both runs print the seed and actual Python, SymPy, and NumPy versions. Run 2's
JSON contains graph conventions and polynomials, validation results,
Experiments A–D, all per-sample momenta and flags, candidate loci, criterion
mapping, and wall-clock time.
