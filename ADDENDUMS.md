# Addendums — lead-lorentzian-landau

## 2026-08-11 · Entry 1 — Literature-collision check (kill criterion 3)

- **Who:** Claude (Fable 5) orchestrator, Claude Code session; web search.
- **What:** searched "Lorentzian Symanzik", "hyperbolic polynomial Landau
  singularity", "K-Lorentzian copositive Feynman", "Lorentzian polynomial
  Feynman Symanzik log-concave" (arXiv + web, 2026-08-11).
- **Result:** no collision. Closest work is Sturmfels–Telek arXiv:2504.01628
  (boundary of the copositive cone of F = product of Landau discriminants) —
  already cited in the claim as motivation, and about copositivity, not
  Lorentzianity/hyperbolicity. No hits for a Lorentzian/Hessian-signature
  version.
- **Bears on:** kill criterion 3 — **NOT FIRED**. Scope caveat discovered later
  (entries 2–4): the one-loop half of the claim reduces to classical facts
  (F = xᵀYx with Y the modified Cayley matrix; det Y = 0 is the leading Landau
  surface), so any novelty claim must rest on the framing (Lorentzian signature
  + cone membership) and on the multi-loop case.

## 2026-08-11 · Entry 2 — Run 1 (flawed; superseded by entry 4)

- **Who:** codex (gpt-5.6-sol) subagent implementing `test/sweep.py`, directed
  by Claude orchestrator; independently rerun by the wrapper agent.
- **Command:**
  `podman run --rm -v <lead-dir>:/work:z docker.io/library/python:3.12-slim bash -c "pip install -q 'sympy==1.14.*' 'numpy==2.*' && python /work/test/sweep.py --seed 20260811"`
  (Python 3.12.13, sympy 1.14.0, numpy 2.5.2). Seed 20260811.
- **Results (raw):** in a region defined as "all independent invariants in
  [−10,−0.1]", Hessian-signature failure fractions (N=200/graph): bubble 0%,
  triangle 10.5%, box 45.0%, sunrise 0%, planar double box 34.0%, crossed
  double box 33.5% (+4.5% negative coefficients). Threshold paths: bubble
  s\*=0.99999974 vs pseudo-threshold (m₁−m₂)²=1 (2.6e-7); triangle
  s\*=0.038822608 vs leading modified-Cayley root (2.3e-7); sunrise
  s\*=1.4335873 "vs pseudo-threshold 1.44" (later retracted, entry 3); box and
  double-box paths "already failing at s=−10" (later shown vacuous, entry 3).
- **Follow-up (Experiment D, same session):** for one-loop graphs,
  **Hessian(F) = 2Y exactly**, Y the modified Cayley matrix (algebraic
  identity: off-diagonal coefficient of F is mᵢ²+mⱼ²−qᵢⱼ², diagonal mᵢ²).
  Triangle/box "deep Euclidean" failures classified by sign(det Y) with 0/400
  misclassified — i.e. every failure lay beyond the leading Landau surface at
  its own kinematics.
- **Flaws (found by adversarial review, entry 3):** the sampled region is not
  the Euclidean region for ≥3-point kinematics; the degree-3 Lorentzian test
  (32 random interior directions) is unsound. **Raw kill-criterion-1 "fired"
  verdict is an artifact.**
- **Audit trail:** `test/results-seed20260811.json` frozen (SHA-256
  `f743927a65538ff32acb96331ed11a5ec6d48dd611d23b4bed50be3abd2fac4f`). The
  run-1 version of `sweep.py` was overwritten before the lead repo's first
  commit of `test/` and is not recoverable; the JSON's embedded
  `conventions`/`constants` fields are its methodological record.

## 2026-08-11 · Entry 3 — Adversarial review of run 1

- **Who:** Claude Opus subagent, independent reimplementation from the
  closed-form polynomials (own 2-forest enumeration; q_F² computed directly
  from explicit four-momenta). Same container image/pins as entry 2; review
  scripts lived in session scratchpad and are recorded here by their findings.
- **Findings:**
  1. **Region mis-specification.** "All invariants negative" ≠ Euclidean
     region; the correct condition is external Gram negative-semidefinite
     (for the triangle, Källén λ(p₁²,p₂²,p₃²) ≤ 0; identity
     det Gram = −λ/4 proven symbolically in run 2). Cross-tab over run-1's own
     samples: **all** signature failures were non-Euclidean (triangle 21/21,
     box 90/90, pdb 68/68, xdb 67/67); Euclidean ∧ non-Lorentzian count = 0
     for every graph. Fresh sampling of genuine Euclidean momenta: 0/1500
     failures. One-loop **proof**: Y = ½(r𝟙ᵀ+𝟙rᵀ) + G with G the Gram of loop
     offsets; Euclidean ⇒ G ⪯ 0 ⇒ (Weyl) at most one positive eigenvalue of Y.
  2. **Degree-3 test unsound.** Random strictly-interior directions miss
     vertex-localized failures. The Brändén–Huh criterion uses coordinate
     partials. Corrected sunrise onset: s\* = min mᵢ² = 1 exactly, from the
     symbolic factorization det Hess(∂F/∂x₁) = 8m₁⁴(m₁²−s). Run-1's sunrise
     "0.0064 from pseudo-threshold 1.44" is **retracted** (true distance to
     nearest standard candidate: 0.36).
  3. Run-1 four-point paths started outside the Euclidean region (Gram had a
     positive eigenvalue at the start; u=+9), masses were degenerate (all 1),
     and their "no transition" controls were vacuous (already failing
     everywhere). Equal-mass "pseudo-threshold 0" matches were degenerate with
     the Gram locus and non-discriminating.
  4. Positive checks: 2-forest enumeration and s/t/u reduction confirmed
     symbolically (difference exactly 0, all six graphs, incl. u appearing
     10× in the crossed and 0× in the planar double box); eigenvalue tolerance
     safe (min |λ₂|/|λmax| = 3.59e-5 vs tol 1e-9); M-convexity of supp(F)
     holds for all six graphs.
- **Bears on:** kill criterion 1 — **NOT FIRED** (the "distinguished region"
  of the criterion's own text is the true Euclidean region, where failures are
  0 with a one-loop proof). Kill criterion 2 — not fired.

## 2026-08-11 · Entry 4 — Run 2 (corrected; authoritative)

- **Who:** codex subagent rewriting `test/sweep.py` per entry-3 review;
  independently rerun and verified by the wrapper agent.
- **Command:** identical podman command and seed as entry 2 (the script now
  performs run 2); output `test/results-run2-seed20260811.json` (includes
  per-sample momenta/invariants/masses/flags so no RNG replay is needed).
- **Method fixes:** kinematics drawn as explicit Euclidean momenta (Gram-NSD
  by construction, asserted per sample; invariants = −Euclidean squares;
  q_F² cross-checked against direct momentum sums to <1e-9); primary degree-3
  test = exact Brändén–Huh coordinate-partial criterion (deterministic), with
  a labeled 64-direction supplementary probe; M-convexity asserted; four-point
  paths use non-degenerate masses (box m=(1.0,1.2,1.4,1.6); double boxes
  m=1.0…1.6) and Euclidean-verified starts; candidate catalog extended to
  Gram-boundary roots, all 2-/3-particle normal and pseudo-thresholds, leading
  and subdiagram Cayley roots, and mᵢ² loci.
- **Results:**
  - Deep-Euclidean sweep (true region, N=200/graph): **0 failures** on all six
    graphs, for coefficient nonnegativity, the primary coordinate-partial
    test, and the supplementary direction probe. **Kill criterion 1 NOT
    FIRED.**
  - Threshold paths (s\*, nearest candidate, identity, |distance|):
    | path | s\* | candidate | identity | dist |
    |---|---:|---:|---|---:|
    | bubble | 0.99999974 | 1 | pseudo-threshold (m₁−m₂)² | 2.6e-7 |
    | triangle (p₃²) | 0.038823063 | 0.038822841 | leading Cayley root (anomalous threshold) | 2.2e-7 |
    | sunrise | 0.99999997 | 1 | **mᵢ² locus — non-standard, see open problems** | 3.0e-8 |
    | box | −0.1746568 | −0.17465701 | leading Cayley root | 2.0e-7 |
    | planar double box | −0.70547279 | −0.7054732 | external Gram boundary (second-type) | 4.1e-7 |
    | crossed double box | −0.70547279 | −0.7054732 | external Gram boundary (second-type) | 4.1e-7 |
    Controls within the true Euclidean interior: no transitions. **Kill
    criterion 2 NOT FIRED** (5/6 transitions on identified Landau-type loci to
    ~1e-7; the sixth is the open sunrise question below).
  - Symbolic proofs inside the validation gate: sunrise factorization
    det Hess(∂F/∂x₁) = 8m₁⁴(m₁²−s); Källén/Gram identity det Gram = −λ/4.
- **Verdict:** no kill criterion fired → **status SURVIVING**, in a sharpened
  form: at one loop the claim is a theorem (Hessian(F)=2Y; Euclidean ⇒
  Lorentzian; signature transition = leading Landau surface, with sheet
  discrimination available via cone membership of the null vector — bubble
  kernel is (m₂,−m₁) at the pseudo-threshold vs (m₂,m₁) at the normal
  threshold); at two loops it is supported numerically.
- **Open problems for the next contributor (in priority order):**
  1. **Sunrise onset at s = m₁².** Exactly reproducible, symbolically proven,
     and not on the standard normal/pseudo-threshold list. Either it lies on
     the sunrise's extended (second-type/subleading) Landau variety — which
     would vindicate the strong conjecture "F is Lorentzian exactly on the
     Landau-free region" — or it is a genuine counterexample to that strong
     form at two loops. Decide which.
  2. **Sheet discrimination.** Formalize and test: at a signature transition,
     the Hessian null vector lies in the positive orthant iff the pinch is
     α-positive (physical sheet). Verified by hand for the bubble only.
  3. **Massless/degenerate kinematics.** Untested (m² sampled in [0.25,4]).
     Second-type/Gram loci and IR-relevant massless corners are exactly where
     the practical interest is; expect boundary degenerations (the M-convex
     support condition is the right tool).
  4. **Scope limit:** all of this is per-topology scalar-denominator analysis;
     numerator zeros/cancellations can render a detected pinch harmless in an
     actual amplitude. The method is a necessary-condition locator, not a
     certifier.
