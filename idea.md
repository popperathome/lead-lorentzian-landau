# The second Symanzik polynomial is K-Lorentzian, and losing that predicts Landau pinches

**Status:** ⏳ QUEUED

## Claim

For distinguished Euclidean kinematic regions, the second Symanzik polynomial F
of a Feynman graph is K-Lorentzian in the sense of Blekherman–Dey
(arXiv:2405.12973), with K the relevant copositive/kinematic cone of
Sturmfels–Telek (arXiv:2504.01628); and the failure of Lorentzianity or
hyperbolicity as kinematics vary is a detectable precursor of Landau
singularities (pinches), via a Hessian-signature transition. Background on
graph polynomials: Bogner–Weinzierl (arXiv:1002.3458). Plausible because the
first Symanzik polynomial is matroidal (already in Lorentzian territory), F
degenerates by sign cancellation exactly in the nonplanar multi-loop cases
where pinches appear, and both the K-Lorentzian and copositive-geometry papers
study cones attached to the same polynomials without citing each other.

## Decisive test

Symbolic/numeric sweep on small graphs (sympy + numpy in a container; a day):

1. One-loop bubble, triangle, box; two-loop sunrise and double box; one known
   nonplanar example. Compute F symbolically with masses/momenta.
2. For each graph, sample Euclidean kinematics and test K-Lorentzianity on the
   positive orthant: positivity of the relevant derivatives and Hessian
   signature (at most one positive eigenvalue) of log F / of F's Hessian on the
   cone, per the Blekherman–Dey criteria.
3. Track the Hessian signature along kinematic paths approaching known Landau
   loci (e.g. normal thresholds s → (m₁+m₂)²) and record whether a signature
   transition occurs at or before the Landau locus, and never far from one.

## Kill criteria

- F already fails K-Lorentzianity generically deep inside the Euclidean region
  for the bubble or triangle (no distinguished region where the property holds
  at all).
- Hessian-signature transitions occur at loci uncorrelated with the Landau
  discriminant across the tested graphs (precursor claim dead even if
  Lorentzianity holds somewhere).
- Literature collision: the comparison is already made (search "Lorentzian
  Symanzik", "hyperbolic polynomial Landau singularity", "copositive
  Lorentzian Feynman").

## Log

See ADDENDUMS.md (no compute spent yet).
