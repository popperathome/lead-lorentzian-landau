#!/usr/bin/env python3
"""Deterministic falsification sweep for Lorentzian second Symanzik forms."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp


COEFF_TOL = 1.0e-12
EIGEN_TOL_FACTOR = 1.0e-9
ORTHANT_DIRECTIONS = 64
DEEP_SAMPLES = 200
PATH_GRID = 400
BISECT_TOL = 1.0e-6
CAYLEY_SIGN_TOL = 1.0e-10


@dataclass(frozen=True)
class GraphSpec:
    name: str
    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    leg_attachments: tuple[int, ...]  # vertex for p1, p2, ...
    kinematic_kind: str
    invariant_names: tuple[str, ...]
    description: str = ""


@dataclass
class CompiledGraph:
    spec: GraphSpec
    x: tuple[sp.Symbol, ...]
    invariants: tuple[sp.Symbol, ...]
    mass_sq: tuple[sp.Symbol, ...]
    U: sp.Expr
    F0: sp.Expr
    F: sp.Expr
    degree: int
    coefficient_fn: Callable[..., object]
    hessian_fn: Callable[..., object] | None
    third_fn: Callable[..., object] | None


def graph_specs() -> tuple[GraphSpec, ...]:
    return (
        GraphSpec(
            "bubble", (0, 1), ((0, 1), (0, 1)), (0, 1), "two_point", ("s",),
            "Two parallel propagators; p1=p enters v1 and p2=-p enters v2.",
        ),
        GraphSpec(
            "triangle", (0, 1, 2), ((1, 2), (0, 2), (0, 1)), (0, 1, 2),
            "three_point", ("p1_sq", "p2_sq", "p3_sq"),
            "e1=(v2,v3), e2=(v1,v3), e3=(v1,v2).",
        ),
        GraphSpec(
            "box", (0, 1, 2, 3), ((0, 1), (1, 2), (2, 3), (3, 0)),
            (0, 1, 2, 3), "four_point",
            ("p1_sq", "p2_sq", "p3_sq", "p4_sq", "s", "t"),
            "Cyclic box, ei=(vi,v{i+1}) and pi attached at vi.",
        ),
        GraphSpec(
            "sunrise", (0, 1), ((0, 1), (0, 1), (0, 1)), (0, 1),
            "two_point", ("s",), "Three parallel propagators.",
        ),
        GraphSpec(
            "planar_double_box", (0, 1, 2, 3, 4, 5),
            ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)),
            (0, 3, 5, 4), "four_point",
            ("p1_sq", "p2_sq", "p3_sq", "p4_sq", "s", "t"),
            "Squares (v1,v2,v3,v4) and (v2,v5,v6,v3) share e2=(v2,v3); "
            "legs (p1,p2,p3,p4) attach to (v1,v4,v6,v5).",
        ),
        GraphSpec(
            "crossed_double_box", (0, 1, 2, 3, 4, 5),
            ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)),
            (0, 5, 3, 4), "four_point",
            ("p1_sq", "p2_sq", "p3_sq", "p4_sq", "s", "t"),
            "Same seven internal edges as the planar double box, with p2 and p3 "
            "crossed: legs (p1,p2,p3,p4) attach to (v1,v6,v4,v5).",
        ),
    )


def components(vertices: tuple[int, ...], edges: tuple[tuple[int, int], ...], chosen: tuple[int, ...]) -> list[set[int]]:
    parent = {v: v for v in vertices}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge_index in chosen:
        union(*edges[edge_index])
    result: dict[int, set[int]] = {}
    for vertex in vertices:
        result.setdefault(find(vertex), set()).add(vertex)
    return list(result.values())


def q_square(spec: GraphSpec, component: set[int], kin: dict[str, sp.Symbol]) -> sp.Expr:
    legs = frozenset(i for i, vertex in enumerate(spec.leg_attachments) if vertex in component)
    nlegs = len(spec.leg_attachments)
    if not legs or len(legs) == nlegs:
        return sp.Integer(0)
    complement = frozenset(range(nlegs)) - legs
    if len(complement) < len(legs):
        legs = complement
    if spec.kinematic_kind == "two_point":
        return kin["s"]
    if spec.kinematic_kind == "three_point":
        if len(legs) == 1:
            return kin[f"p{next(iter(legs)) + 1}_sq"]
        missing = next(iter(frozenset(range(3)) - legs))
        return kin[f"p{missing + 1}_sq"]
    if spec.kinematic_kind == "four_point":
        if len(legs) == 1:
            return kin[f"p{next(iter(legs)) + 1}_sq"]
        pair = frozenset(legs)
        if pair in (frozenset((0, 1)), frozenset((2, 3))):
            return kin["s"]
        if pair in (frozenset((1, 2)), frozenset((0, 3))):
            return kin["t"]
        if pair in (frozenset((0, 2)), frozenset((1, 3))):
            return sum(kin[f"p{i}_sq"] for i in range(1, 5)) - kin["s"] - kin["t"]
    raise ValueError(f"Cannot reduce momentum subset {sorted(legs)} for {spec.name}")


def build_symanzik(spec: GraphSpec) -> CompiledGraph:
    edge_count, vertex_count = len(spec.edges), len(spec.vertices)
    x = sp.symbols(f"x1:{edge_count + 1}")
    mass_sq = sp.symbols(f"mu1:{edge_count + 1}", nonnegative=True)
    invariants = tuple(sp.Symbol(name, real=True) for name in spec.invariant_names)
    kin = dict(zip(spec.invariant_names, invariants, strict=True))

    U = sp.Integer(0)
    for chosen in itertools.combinations(range(edge_count), vertex_count - 1):
        if len(components(spec.vertices, spec.edges, chosen)) == 1:
            U += sp.prod(x[e] for e in range(edge_count) if e not in chosen)

    F0 = sp.Integer(0)
    for chosen in itertools.combinations(range(edge_count), vertex_count - 2):
        comps = components(spec.vertices, spec.edges, chosen)
        if len(comps) == 2:
            q2 = q_square(spec, comps[0], kin)
            F0 += -q2 * sp.prod(x[e] for e in range(edge_count) if e not in chosen)

    F = sp.expand(F0 + U * sum(mass_sq[e] * x[e] for e in range(edge_count)))
    U, F0 = sp.expand(U), sp.expand(F0)
    degree = edge_count - vertex_count + 2
    assert sp.Poly(F, x).is_homogeneous
    assert sp.Poly(F, x).total_degree() == degree

    parameters = invariants + mass_sq
    coefficients = tuple(sp.Poly(F, x).coeffs())
    coefficient_fn = sp.lambdify(parameters, coefficients, "numpy")
    hessian_fn: Callable[..., object] | None = None
    third_fn: Callable[..., object] | None = None
    if degree == 2:
        hessian_fn = sp.lambdify(parameters, sp.hessian(F, x), "numpy")
    elif degree == 3:
        tensor = [[[sp.diff(F, x[i], x[j], x[k]) for k in range(edge_count)]
                   for j in range(edge_count)] for i in range(edge_count)]
        third_fn = sp.lambdify(parameters, tensor, "numpy")
    else:
        raise AssertionError(f"Unexpected degree {degree} for {spec.name}")
    return CompiledGraph(spec, x, invariants, mass_sq, U, F0, F, degree,
                         coefficient_fn, hessian_fn, third_fn)


def evaluate(compiled: CompiledGraph, invariant_values: dict[str, float], masses_sq: list[float],
             directions: np.ndarray | None = None) -> dict[str, object]:
    args = [invariant_values[name] for name in compiled.spec.invariant_names] + list(masses_sq)
    coeffs = np.asarray(compiled.coefficient_fn(*args), dtype=float).reshape(-1)
    coefficient_pass = bool(np.all(coeffs >= -COEFF_TOL))
    primary_matrices: list[np.ndarray]
    supplementary_matrices: list[np.ndarray] = []
    if compiled.degree == 2:
        assert compiled.hessian_fn is not None
        primary_matrices = [np.asarray(compiled.hessian_fn(*args), dtype=float)]
    else:
        assert compiled.third_fn is not None
        tensor = np.asarray(compiled.third_fn(*args), dtype=float)
        primary_matrices = [tensor[:, :, coordinate] for coordinate in range(len(compiled.spec.edges))]
        if directions is not None:
            supplementary_matrices = [np.einsum("ijk,k->ij", tensor, direction) for direction in directions]

    def signatures_for(matrices: list[np.ndarray], at_most_one: bool) -> tuple[bool, list[dict[str, object]]]:
        signatures: list[dict[str, object]] = []
        all_pass = True
        for matrix in matrices:
            eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
            scale = float(np.max(np.abs(eigenvalues)))
            tolerance = EIGEN_TOL_FACTOR * scale
            positive_count = int(np.count_nonzero(eigenvalues > tolerance))
            passed = positive_count <= 1 if at_most_one else positive_count == 1
            all_pass = all_pass and passed
            signatures.append({
                "positive_count": positive_count,
                "zero_count": int(np.count_nonzero(np.abs(eigenvalues) <= tolerance)),
                "eigenvalues": eigenvalues.tolist(),
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
                "tolerance": tolerance,
                "pass": passed,
            })
        return all_pass, signatures

    primary_pass, primary_signatures = signatures_for(primary_matrices, compiled.degree == 3)
    supplementary_pass, supplementary_signatures = signatures_for(supplementary_matrices, True)
    return {
        "coefficient_nonnegative": coefficient_pass,
        "minimum_coefficient": float(np.min(coeffs)),
        "hessian_signature": primary_pass,
        "primary_coordinate_signature": primary_pass,
        "primary_test_kind": ("constant quadratic Hessian: exactly one positive eigenvalue"
                              if compiled.degree == 2 else
                              "all coordinate partial Hessians: at most one positive eigenvalue"),
        "primary_tested_matrices": len(primary_matrices),
        "signatures": primary_signatures,
        "supplementary_random_direction_signature": supplementary_pass if supplementary_matrices else None,
        "supplementary_tested_matrices": len(supplementary_matrices),
        "supplementary_signatures": supplementary_signatures,
    }


def bisect_boolean_boundary(predicate: Callable[[float], bool], lo: float, hi: float) -> float:
    assert predicate(lo) and not predicate(hi)
    while hi - lo > BISECT_TOL:
        mid = (lo + hi) / 2.0
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def determinant_roots(function: Callable[[float], float], brackets: list[tuple[float, float]]) -> list[float]:
    roots = []
    for lo, hi in brackets:
        flo, fhi = function(lo), function(hi)
        assert flo * fhi < 0.0
        while hi - lo > 1.0e-10:
            mid = (lo + hi) / 2.0
            fm = function(mid)
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        roots.append((lo + hi) / 2.0)
    return roots


def m_convex_support_check(graph: CompiledGraph) -> dict[str, object]:
    support = set(sp.Poly(graph.F, graph.x).monoms())
    for alpha in support:
        for beta in support:
            for i in range(len(alpha)):
                if alpha[i] <= beta[i]:
                    continue
                exchange_found = False
                for j in range(len(alpha)):
                    if alpha[j] >= beta[j]:
                        continue
                    alpha_exchange = list(alpha)
                    beta_exchange = list(beta)
                    alpha_exchange[i] -= 1
                    alpha_exchange[j] += 1
                    beta_exchange[i] += 1
                    beta_exchange[j] -= 1
                    if tuple(alpha_exchange) in support and tuple(beta_exchange) in support:
                        exchange_found = True
                        break
                assert exchange_found, f"M-convex exchange failed for {graph.spec.name}: {alpha}, {beta}, i={i}"
    return {"passed": True, "support_size": len(support),
            "support": [list(monomial) for monomial in sorted(support)]}


def validation_gate(graphs: dict[str, CompiledGraph]) -> dict[str, object]:
    bubble = graphs["bubble"]
    x1, x2 = bubble.x
    s = bubble.invariants[0]
    mu1, mu2 = bubble.mass_sq
    bubble_closed = -s * x1 * x2 + (x1 + x2) * (mu1 * x1 + mu2 * x2)
    assert sp.simplify(bubble.F - bubble_closed) == 0

    triangle = graphs["triangle"]
    x1, x2, x3 = triangle.x
    p1, p2, p3 = triangle.invariants
    mu1, mu2, mu3 = triangle.mass_sq
    triangle_closed = (-p1 * x2 * x3 - p2 * x1 * x3 - p3 * x1 * x2
                       + (x1 + x2 + x3) * (mu1 * x1 + mu2 * x2 + mu3 * x3))
    assert sp.simplify(triangle.F - triangle_closed) == 0

    sunrise = graphs["sunrise"]
    x1, x2, x3 = sunrise.x
    s = sunrise.invariants[0]
    mu1, mu2, mu3 = sunrise.mass_sq
    sunrise_closed = (-s * x1 * x2 * x3
                      + (x1 * x2 + x1 * x3 + x2 * x3) * (mu1 * x1 + mu2 * x2 + mu3 * x3))
    assert sp.simplify(sunrise.F - sunrise_closed) == 0
    sunrise_partial_hessian_det = sp.factor(sp.hessian(sp.diff(sunrise.F, x1), sunrise.x).det())
    sunrise_expected_factor = 8 * mu1**2 * (mu1 - s)
    assert sp.simplify(sunrise_partial_hessian_det - sunrise_expected_factor) == 0

    a, b, c = sp.symbols("a b c", real=True)
    triangle_external_gram = sp.Matrix([[a, (c - a - b) / 2], [(c - a - b) / 2, b]])
    kallen = a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c
    assert sp.simplify(triangle_external_gram.det() + kallen / 4) == 0

    determinant = lambda value: 4.0 * 1.0**2 * 2.0**2 - (1.0**2 + 2.0**2 - value) ** 2
    roots = determinant_roots(determinant, [(0.0, 2.0), (8.0, 10.0)])
    assert max(abs(roots[0] - 1.0), abs(roots[1] - 9.0)) < 1.0e-6
    bubble_at_minus_five = evaluate(bubble, {"s": -5.0}, [1.0, 4.0])
    signature = bubble_at_minus_five["signatures"][0]
    assert signature["positive_count"] == 1
    assert signature["zero_count"] == 0
    return {
        "closed_forms": {"bubble": True, "triangle": True, "sunrise": True},
        "bubble_masses": [1.0, 2.0],
        "bubble_hessian_determinant": "4*m1^2*m2^2 - (m1^2+m2^2-s)^2",
        "located_zero_crossings": roots,
        "expected_zero_crossings": [1.0, 9.0],
        "signature_at_s_minus_5": signature,
        "triangle_gram_kallen_identity": "det Gram(p1,p2) = -lambda(p1^2,p2^2,p3^2)/4; with p1^2,p2^2<0, Gram NSD iff lambda<=0",
        "sunrise_coordinate_partial_factorization": {
            "coordinate": "x1",
            "symbolic_determinant": str(sunrise_partial_hessian_det),
            "identity": "det Hess(dF/dx1) = 8*(m1^2)^2*(m1^2-s)",
            "passed": True,
        },
        "m_convex_support": {name: m_convex_support_check(graph) for name, graph in graphs.items()},
        "passed": True,
    }


def random_directions(rng: np.random.Generator, edge_count: int) -> np.ndarray:
    # np.random.random is in [0,1); 1-random therefore lies in (0,1].
    return 1.0 - rng.random((ORTHANT_DIRECTIONS, edge_count))


def invariants_from_euclidean_momenta(kind: str, momenta: np.ndarray) -> dict[str, float]:
    squares = [-float(momentum @ momentum) for momentum in momenta]
    if kind == "two_point":
        return {"s": squares[0]}
    if kind == "three_point":
        return {f"p{i + 1}_sq": value for i, value in enumerate(squares)}
    if kind == "four_point":
        return {
            **{f"p{i + 1}_sq": value for i, value in enumerate(squares)},
            "s": -float((momenta[0] + momenta[1]) @ (momenta[0] + momenta[1])),
            "t": -float((momenta[1] + momenta[2]) @ (momenta[1] + momenta[2])),
        }
    raise ValueError(kind)


def sample_euclidean_kinematics(spec: GraphSpec, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    nlegs = len(spec.leg_attachments)
    while True:
        independent = rng.normal(size=(nlegs - 1, 3))
        momenta = np.vstack((independent, -np.sum(independent, axis=0)))
        raw = invariants_from_euclidean_momenta(spec.kinematic_kind, momenta)
        magnitudes = [-value for value in raw.values()]
        if spec.kinematic_kind == "four_point":
            u_value = sum(raw[f"p{i}_sq"] for i in range(1, 5)) - raw["s"] - raw["t"]
            magnitudes.append(-u_value)
        minimum, maximum = min(magnitudes), max(magnitudes)
        if minimum > 1.0e-12 and maximum / minimum <= 100.0:
            lower_scale_sq = 0.1 / minimum
            upper_scale_sq = 10.0 / maximum
            scale_sq = float(rng.uniform(lower_scale_sq, upper_scale_sq))
            momenta *= np.sqrt(scale_sq)
            invariants = invariants_from_euclidean_momenta(spec.kinematic_kind, momenta)
            values = list(invariants.values())
            if spec.kinematic_kind == "four_point":
                values.append(sum(invariants[f"p{i}_sq"] for i in range(1, 5))
                              - invariants["s"] - invariants["t"])
            assert all(-10.0 - 1.0e-10 <= value <= -0.1 + 1.0e-10 for value in values)
            return momenta, invariants


def external_gram(momenta: np.ndarray) -> np.ndarray:
    independent = momenta[:-1]
    return -(independent @ independent.T)


def q_square_crosscheck(graph: CompiledGraph, momenta: np.ndarray,
                        invariants: dict[str, float]) -> float:
    vertex_count = len(graph.spec.vertices)
    maximum_error = 0.0
    for chosen in itertools.combinations(range(len(graph.spec.edges)), vertex_count - 2):
        comps = components(graph.spec.vertices, graph.spec.edges, chosen)
        if len(comps) != 2:
            continue
        enumerated = float(q_square(graph.spec, comps[0], invariants))
        legs = [index for index, vertex in enumerate(graph.spec.leg_attachments) if vertex in comps[0]]
        summed = np.sum(momenta[legs], axis=0) if legs else np.zeros(3)
        direct = -float(summed @ summed)
        maximum_error = max(maximum_error, abs(enumerated - direct))
    assert maximum_error < 1.0e-9
    return maximum_error


def deep_euclidean_sweep(graphs: dict[str, CompiledGraph], rng: np.random.Generator,
                         retained_samples: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, graph in graphs.items():
        coefficient_failures = 0
        signature_failures = 0
        supplementary_failures = 0
        minimum_coefficient = float("inf")
        positive_count_histogram: dict[str, int] = {}
        retained_samples[name] = []
        for sample_index in range(DEEP_SAMPLES):
            momenta, invariants = sample_euclidean_kinematics(graph.spec, rng)
            masses_sq = rng.uniform(0.25, 4.0, len(graph.spec.edges)).tolist()
            directions = random_directions(rng, len(graph.spec.edges)) if graph.degree == 3 else None
            result = evaluate(graph, invariants, masses_sq, directions)
            gram = external_gram(momenta)
            gram_eigenvalues = np.linalg.eigvalsh(gram).tolist()
            assert max(gram_eigenvalues) <= 1.0e-9
            q_error = q_square_crosscheck(graph, momenta, invariants)
            logged_invariants = dict(invariants)
            if graph.spec.kinematic_kind == "four_point":
                logged_invariants["u"] = (sum(invariants[f"p{i}_sq"] for i in range(1, 5))
                                           - invariants["s"] - invariants["t"])
            retained_samples[name].append({
                "sample_index": sample_index,
                "invariants": logged_invariants,
                "masses_sq": masses_sq,
                "masses": [float(np.sqrt(value)) for value in masses_sq],
                "euclidean_momenta": momenta.tolist(),
                "external_gram_matrix": gram.tolist(),
                "external_gram_eigenvalues": gram_eigenvalues,
                "gram_negative_semidefinite": True,
                "q_square_crosscheck_max_abs_error": q_error,
                "coefficient_nonnegative": result["coefficient_nonnegative"],
                "signature_pass": result["hessian_signature"],
                "primary_coordinate_signature_pass": result["primary_coordinate_signature"],
                "primary_signatures": result["signatures"],
                "supplementary_random_direction_signature_pass": result["supplementary_random_direction_signature"],
            })
            coefficient_failures += not result["coefficient_nonnegative"]
            signature_failures += not result["hessian_signature"]
            supplementary_failures += result["supplementary_random_direction_signature"] is False
            minimum_coefficient = min(minimum_coefficient, result["minimum_coefficient"])
            for item in result["signatures"]:
                key = str(item["positive_count"])
                positive_count_histogram[key] = positive_count_histogram.get(key, 0) + 1
        output[name] = {
            "samples": DEEP_SAMPLES,
            "invariant_interval": [-10.0, -0.1],
            "kinematic_sampling": "Gaussian Euclidean 3-vectors for n-1 legs, last leg by conservation, followed by one common rescaling conditioned so all p_i^2 and channel invariants (including u for four legs) lie in [-10,-0.1].",
            "mass_squared_interval": [0.25, 4.0],
            "coefficient_nonnegativity_failures": coefficient_failures,
            "coefficient_nonnegativity_failure_fraction": coefficient_failures / DEEP_SAMPLES,
            "hessian_signature_failures": signature_failures,
            "hessian_signature_failure_fraction": signature_failures / DEEP_SAMPLES,
            "primary_coordinate_signature_failures": signature_failures,
            "primary_coordinate_signature_failure_fraction": signature_failures / DEEP_SAMPLES,
            "supplementary_random_direction_failures": supplementary_failures,
            "supplementary_random_direction_failure_fraction": supplementary_failures / DEEP_SAMPLES,
            "minimum_coefficient_seen": minimum_coefficient,
            "positive_eigenvalue_count_histogram": positive_count_histogram,
            "per_sample_records": retained_samples[name],
        }
    return output


def sign_label(value: float) -> str:
    if value > CAYLEY_SIGN_TOL:
        return "positive"
    if value < -CAYLEY_SIGN_TOL:
        return "negative"
    return "zero"


def pair_q_squares(graph: CompiledGraph, invariants: dict[str, float]) -> dict[tuple[int, int], float]:
    """Reuse the graph's 2-forest/q_square convention for one-loop cut pairs."""
    edge_count = len(graph.spec.edges)
    result: dict[tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(edge_count), 2):
        surviving = tuple(edge for edge in range(edge_count) if edge not in (i, j))
        comps = components(graph.spec.vertices, graph.spec.edges, surviving)
        assert len(comps) == 2
        result[(i, j)] = float(q_square(graph.spec, comps[0], invariants))
    return result


def cayley_matrix(masses_sq: list[float], q2: dict[tuple[int, int], float],
                  subset: tuple[int, ...] | None = None) -> np.ndarray:
    indices = tuple(range(len(masses_sq))) if subset is None else subset
    matrix = np.zeros((len(indices), len(indices)), dtype=float)
    for local_i, edge_i in enumerate(indices):
        matrix[local_i, local_i] = masses_sq[edge_i]
        for local_j in range(local_i + 1, len(indices)):
            edge_j = indices[local_j]
            key = (min(edge_i, edge_j), max(edge_i, edge_j))
            value = (masses_sq[edge_i] + masses_sq[edge_j] - q2[key]) / 2.0
            matrix[local_i, local_j] = matrix[local_j, local_i] = value
    return matrix


def cayley_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    leading_minors = [float(np.linalg.det(matrix[:size, :size]))
                      for size in range(1, matrix.shape[0] + 1)]
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0).tolist()
    return {
        "matrix": matrix.tolist(),
        "determinant": leading_minors[-1],
        "determinant_sign": sign_label(leading_minors[-1]),
        "leading_principal_minors": leading_minors,
        "leading_principal_minor_signs": [sign_label(value) for value in leading_minors],
        "eigenvalues": eigenvalues,
        "eigenvalue_signs": [sign_label(value) for value in eigenvalues],
    }


def hessian_eigenvalues(graph: CompiledGraph, invariants: dict[str, float], masses_sq: list[float]) -> list[float]:
    assert graph.degree == 2 and graph.hessian_fn is not None
    args = [invariants[name] for name in graph.spec.invariant_names] + masses_sq
    matrix = np.asarray(graph.hessian_fn(*args), dtype=float)
    return np.linalg.eigvalsh((matrix + matrix.T) / 2.0).tolist()


def category_cross_tab(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row[key]).lower() if isinstance(row[key], bool) else str(row[key])
        bucket = table.setdefault(category, {"signature_pass": 0, "signature_fail": 0, "total": 0})
        bucket["signature_pass" if row["signature_pass"] else "signature_fail"] += 1
        bucket["total"] += 1
    return dict(sorted(table.items()))


def binary_confusion(rows: list[dict[str, object]], feature: str, failure_sign: str) -> dict[str, int]:
    counts = {"true_fail": 0, "false_fail": 0, "true_pass": 0, "false_pass": 0}
    for row in rows:
        predicted_fail = row["feature_signs"][feature] == failure_sign
        actual_fail = not row["signature_pass"]
        if predicted_fail and actual_fail:
            counts["true_fail"] += 1
        elif predicted_fail:
            counts["false_fail"] += 1
        elif actual_fail:
            counts["false_pass"] += 1
        else:
            counts["true_pass"] += 1
    counts["misclassified"] = counts["false_fail"] + counts["false_pass"]
    return counts


def feature_sign_cross_tab(rows: list[dict[str, object]], feature: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        sign = row["feature_signs"][feature]
        bucket = table.setdefault(sign, {"signature_pass": 0, "signature_fail": 0, "total": 0})
        bucket["signature_pass" if row["signature_pass"] else "signature_fail"] += 1
        bucket["total"] += 1
    return dict(sorted(table.items()))


def prefixed_feature_cross_tabs(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    features = [name for name in rows[0]["feature_signs"] if name.startswith(prefix)]
    return {feature.removeprefix(prefix): feature_sign_cross_tab(rows, feature) for feature in features}


def best_single_sign_predictor(rows: list[dict[str, object]], feature_names: list[str]) -> dict[str, object]:
    candidates = []
    for feature in feature_names:
        for failure_sign in ("negative", "zero", "positive"):
            confusion = binary_confusion(rows, feature, failure_sign)
            candidates.append({"feature": feature, "predict_failure_when_sign_is": failure_sign,
                               "confusion_matrix": confusion,
                               "misclassified": confusion["misclassified"]})
    # min() is stable, so equally accurate rules prefer the caller's feature
    # order (det_Y is deliberately first).
    return min(candidates, key=lambda item: item["misclassified"])


def subset_name(subset: tuple[int, ...]) -> str:
    return "_".join(f"e{edge + 1}" for edge in subset)


def one_loop_diagnostic_row(graph: CompiledGraph, sample: dict[str, object]) -> dict[str, object]:
    invariants = dict(sample["invariants"])
    masses_sq = list(sample["masses_sq"])
    q2 = pair_q_squares(graph, invariants)
    full = cayley_diagnostics(cayley_matrix(masses_sq, q2))
    feature_signs = {"det_Y": full["determinant_sign"]}
    for i, sign in enumerate(full["leading_principal_minor_signs"], 1):
        feature_signs[f"leading_minor_{i}"] = sign
    for i, sign in enumerate(full["eigenvalue_signs"], 1):
        feature_signs[f"Y_eigenvalue_{i}"] = sign

    triangle_subdeterminants: dict[str, dict[str, object]] = {}
    for subset in itertools.combinations(range(len(masses_sq)), 3):
        determinant = float(np.linalg.det(cayley_matrix(masses_sq, q2, subset)))
        sign = sign_label(determinant)
        name = subset_name(subset)
        triangle_subdeterminants[name] = {"determinant": determinant, "sign": sign}
        feature_signs[f"triangle_subdet_{name}"] = sign

    bubble_subdeterminants: dict[str, dict[str, object]] = {}
    for subset in itertools.combinations(range(len(masses_sq)), 2):
        determinant = float(np.linalg.det(cayley_matrix(masses_sq, q2, subset)))
        sign = sign_label(determinant)
        name = subset_name(subset)
        q_value = q2[(min(subset), max(subset))]
        mass_i, mass_j = np.sqrt(masses_sq[subset[0]]), np.sqrt(masses_sq[subset[1]])
        bubble_subdeterminants[name] = {
            "determinant": determinant, "sign": sign, "q_squared": q_value,
            "pseudo_threshold": float((mass_i - mass_j) ** 2),
            "normal_threshold": float((mass_i + mass_j) ** 2),
        }
        feature_signs[f"bubble_subdet_{name}"] = sign

    triangle_pattern = ",".join(item["sign"] for item in triangle_subdeterminants.values())
    bubble_pattern = ",".join(item["sign"] for item in bubble_subdeterminants.values())
    # At equal large negative pair invariants, size-k Cayley determinants have
    # sign (-1)^(k-1): positive for k=3 and negative for k=2.
    any_triangle_nonreference = any(item["sign"] != "positive" for item in triangle_subdeterminants.values())
    any_bubble_nonreference = any(item["sign"] != "negative" for item in bubble_subdeterminants.values())
    return {
        "sample_index": sample["sample_index"], "signature_pass": sample["signature_pass"],
        "invariants": invariants, "masses_sq": masses_sq,
        "masses": [float(np.sqrt(value)) for value in masses_sq],
        "q_ij_squared": {subset_name(pair): value for pair, value in q2.items()},
        "Y": full, "feature_signs": feature_signs,
        "Y_det_sign": full["determinant_sign"],
        "leading_minor_sign_pattern": ",".join(full["leading_principal_minor_signs"]),
        "Y_eigenvalue_sign_pattern": ",".join(full["eigenvalue_signs"]),
        "triangle_subdeterminants": triangle_subdeterminants,
        "triangle_subdeterminant_sign_pattern": triangle_pattern,
        "bubble_subdeterminants": bubble_subdeterminants,
        "bubble_subdeterminant_sign_pattern": bubble_pattern,
        "any_triangle_subdiagram_nonreference_or_surface": any_triangle_nonreference,
        "any_bubble_subdiagram_nonreference_or_surface": any_bubble_nonreference,
        "any_subdiagram_nonreference_or_surface": any_triangle_nonreference or any_bubble_nonreference,
    }


def triangle_experiment_d(graph: CompiledGraph, samples: list[dict[str, object]]) -> dict[str, object]:
    rows = [one_loop_diagnostic_row(graph, sample) for sample in samples]
    for row in rows:
        row["hessian_eigenvalues"] = hessian_eigenvalues(graph, row["invariants"], row["masses_sq"])
    features = [name for name in rows[0]["feature_signs"] if not name.startswith("triangle_subdet_")]
    det_predictors = [
        {"predict_failure_when_sign_is": sign, "confusion_matrix": binary_confusion(rows, "det_Y", sign)}
        for sign in ("negative", "positive")
    ]
    best_det = min(det_predictors, key=lambda item: item["confusion_matrix"]["misclassified"])
    fully_explained = (best_det["predict_failure_when_sign_is"] == "negative"
                       and best_det["confusion_matrix"]["misclassified"] == 0)
    failures = [row for row in rows if not row["signature_pass"]]
    failures_on_det_or_bubble_nonreference = sum(
        row["Y_det_sign"] != "positive" or row["any_bubble_subdiagram_nonreference_or_surface"]
        for row in failures
    )
    spot_checks = [
        {key: row[key] for key in ("sample_index", "masses", "masses_sq", "invariants",
                                    "Y", "hessian_eigenvalues")}
        for row in rows if not row["signature_pass"]
    ][:3]
    return {
        "samples": len(rows),
        "signature_counts": category_cross_tab(rows, "signature_pass"),
        "det_Y_sign_confusion": category_cross_tab(rows, "Y_det_sign"),
        "leading_principal_minor_sign_pattern_confusion": category_cross_tab(rows, "leading_minor_sign_pattern"),
        "Y_eigenvalue_sign_pattern_confusion": category_cross_tab(rows, "Y_eigenvalue_sign_pattern"),
        "bubble_subdiagram_sign_pattern_confusion": category_cross_tab(rows, "bubble_subdeterminant_sign_pattern"),
        "bubble_subdiagram_det_sign_confusions": prefixed_feature_cross_tabs(rows, "bubble_subdet_"),
        "any_bubble_subdiagram_nonreference_confusion": category_cross_tab(rows, "any_bubble_subdiagram_nonreference_or_surface"),
        "det_Y_binary_predictors": det_predictors,
        "best_single_sign_predictor": best_single_sign_predictor(rows, features),
        "failures_fully_explained_by_det_Y_nonreference_sign": fully_explained,
        "failures_on_det_Y_or_bubble_nonreference_side": failures_on_det_or_bubble_nonreference,
        "total_signature_failures": len(failures),
        "failing_sample_spot_checks": spot_checks,
    }


def box_experiment_d(graph: CompiledGraph, samples: list[dict[str, object]]) -> dict[str, object]:
    rows = [one_loop_diagnostic_row(graph, sample) for sample in samples]
    features = list(rows[0]["feature_signs"])
    det_predictors = [
        {"predict_failure_when_sign_is": sign, "confusion_matrix": binary_confusion(rows, "det_Y", sign)}
        for sign in ("negative", "positive")
    ]
    best_det = min(det_predictors, key=lambda item: item["confusion_matrix"]["misclassified"])
    fully_explained_by_leading = (best_det["predict_failure_when_sign_is"] == "positive"
                                  and best_det["confusion_matrix"]["misclassified"] == 0)
    path_config = next(config for config in path_configurations() if config["graph"] == "box")
    path_invariants = dict(path_config["fixed"])
    path_invariants[str(path_config["raised"])] = float(path_config["range"][0])
    path_masses_sq = [float(mass)**2 for mass in path_config["masses"]]
    path_evaluation = evaluate(graph, path_invariants, path_masses_sq)
    path_sample = {
        "sample_index": "experiment_B_path_start",
        "invariants": path_invariants,
        "masses_sq": path_masses_sq,
        "signature_pass": path_evaluation["hessian_signature"],
    }
    path_start = one_loop_diagnostic_row(graph, path_sample)
    path_start["hessian_eigenvalues"] = hessian_eigenvalues(
        graph, path_start["invariants"], path_start["masses_sq"])
    path_start["leading_Landau_nonreference_or_surface"] = path_start["Y"]["determinant_sign"] != "negative"
    path_start["external_Gram_eigenvalues"] = np.linalg.eigvalsh(
        external_gram(np.asarray(path_config["explicit_euclidean_start_momenta"], dtype=float))).tolist()
    path_start["external_Gram_negative_semidefinite"] = max(path_start["external_Gram_eigenvalues"]) <= 1.0e-10
    failures = [row for row in rows if not row["signature_pass"]]
    failures_on_any_subdiagram_nonreference = sum(
        row["any_subdiagram_nonreference_or_surface"] for row in failures)
    return {
        "samples": len(rows),
        "det_Y_sign_confusion": category_cross_tab(rows, "Y_det_sign"),
        "leading_principal_minor_sign_pattern_confusion": category_cross_tab(rows, "leading_minor_sign_pattern"),
        "Y_eigenvalue_sign_pattern_confusion": category_cross_tab(rows, "Y_eigenvalue_sign_pattern"),
        "triangle_subdiagram_sign_pattern_confusion": category_cross_tab(rows, "triangle_subdeterminant_sign_pattern"),
        "bubble_subdiagram_sign_pattern_confusion": category_cross_tab(rows, "bubble_subdeterminant_sign_pattern"),
        "triangle_subdiagram_det_sign_confusions": prefixed_feature_cross_tabs(rows, "triangle_subdet_"),
        "bubble_subdiagram_det_sign_confusions": prefixed_feature_cross_tabs(rows, "bubble_subdet_"),
        "any_triangle_subdiagram_nonreference_confusion": category_cross_tab(rows, "any_triangle_subdiagram_nonreference_or_surface"),
        "any_bubble_subdiagram_nonreference_confusion": category_cross_tab(rows, "any_bubble_subdiagram_nonreference_or_surface"),
        "any_subdiagram_nonreference_confusion": category_cross_tab(rows, "any_subdiagram_nonreference_or_surface"),
        "det_Y_binary_predictors": det_predictors,
        "best_single_sign_predictor": best_single_sign_predictor(rows, features),
        "failures_fully_explained_by_det_Y_nonreference_sign": fully_explained_by_leading,
        "failures_on_any_subdiagram_nonreference_side": failures_on_any_subdiagram_nonreference,
        "total_signature_failures": len(failures),
        "experiment_B_path_start": path_start,
    }


def u_channel_audit(samples_by_graph: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for graph_name in ("box", "planar_double_box", "crossed_double_box"):
        groups = {"u_negative_fully_euclidean": {"samples": 0, "signature_failures": 0},
                  "u_nonnegative": {"samples": 0, "signature_failures": 0}}
        sample_classifications = []
        for sample in samples_by_graph[graph_name]:
            invariants = sample["invariants"]
            u_value = sum(invariants[f"p{i}_sq"] for i in range(1, 5)) - invariants["s"] - invariants["t"]
            group = groups["u_negative_fully_euclidean" if u_value < 0.0 else "u_nonnegative"]
            group["samples"] += 1
            group["signature_failures"] += not sample["signature_pass"]
            sample_classifications.append({
                "sample_index": sample["sample_index"], "u": u_value, "u_negative": u_value < 0.0,
                "signature_pass": sample["signature_pass"],
            })
        total_failures = sum(group["signature_failures"] for group in groups.values())
        for group in groups.values():
            group["signature_failure_fraction"] = (group["signature_failures"] / group["samples"]
                                                    if group["samples"] else None)
        nonnegative_failures = groups["u_nonnegative"]["signature_failures"]
        fraction_carried = nonnegative_failures / total_failures if total_failures else 0.0
        output[graph_name] = {
            **groups,
            "total_signature_failures": total_failures,
            "fraction_of_failures_in_u_nonnegative_subsample": fraction_carried,
            "u_nonnegative_subsample_carries_most_failures": nonnegative_failures > total_failures / 2,
            "sample_classifications": sample_classifications,
        }
    return output


def experiment_d(graphs: dict[str, CompiledGraph], samples: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    triangle = triangle_experiment_d(graphs["triangle"], samples["triangle"])
    box = box_experiment_d(graphs["box"], samples["box"])
    return {
        "documented_choices": {
            "pair_invariants": "For triangle and box, q_ij^2 is recomputed with the existing components() and q_square() logic after cutting propagators ei and ej.",
            "subdiagrams": "Pinched one-loop subdiagrams are principal Cayley submatrices on surviving propagator subsets, retaining the full box q_ij^2 values.",
            "sign_tolerance": CAYLEY_SIGN_TOL,
            "nonreference_side": "At a canonical far-Euclidean point with all pair invariants large negative, size-k Cayley determinant sign is (-1)^(k-1): bubbles negative, triangles positive, boxes negative. A different sign or zero is labeled nonreference_or_surface. This sign label diagnoses connected components but does not by itself prove a unique real path crossing.",
            "best_single_sign_predictor": "Among one Cayley determinant, leading-minor, or sorted-eigenvalue sign at a time, choose the rule 'signature failure iff that sign equals negative/zero/positive' with the fewest in-sample errors.",
        },
        "triangle": triangle,
        "box": box,
        "u_channel_audit": u_channel_audit(samples),
    }


def triangle_cayley_candidates() -> list[dict[str, object]]:
    masses = [1.0, 1.3, 1.7]
    z = sp.Symbol("p3_sq", real=True)
    q12, q13, q23 = z, sp.Float(-3.0), sp.Float(-3.0)
    q = {(0, 1): q12, (0, 2): q13, (1, 2): q23}
    Y = sp.zeros(3)
    for i in range(3):
        Y[i, i] = masses[i] ** 2
        for j in range(i + 1, 3):
            Y[i, j] = Y[j, i] = (masses[i] ** 2 + masses[j] ** 2 - q[(i, j)]) / 2
    roots = sp.nroots(sp.Poly(sp.expand(Y.det()), z))
    real_roots = sorted(float(sp.re(root)) for root in roots if abs(float(sp.im(root))) < 1.0e-10)
    return [{"label": "triangle_modified_Cayley_det_root", "value": root,
             "cut_edges": ["e1", "e2"]} for root in real_roots]


def path_configurations() -> list[dict[str, object]]:
    four_momenta = np.asarray([
        [1.0, 0.0, 0.0],
        [0.2, 1.1, 0.0],
        [-0.4, 0.3, 0.9],
        [-0.8, -1.4, -0.9],
    ])
    four_invariants = invariants_from_euclidean_momenta("four_point", four_momenta)
    four_fixed = {name: value for name, value in four_invariants.items() if name != "s"}
    four_start = four_invariants["s"]
    return [
        {
            "name": "bubble_s", "graph": "bubble", "raised": "s", "fixed": {}, "masses": [1.0, 2.0],
            "range": [-10.0, 13.5], "control_range": [-10.0, -2.0],
        },
        {
            "name": "triangle_p3_sq", "graph": "triangle", "raised": "p3_sq",
            "fixed": {"p1_sq": -3.0, "p2_sq": -3.0}, "masses": [1.0, 1.3, 1.7],
            "range": [-10.0, 8.0], "control_range": [-10.0, -2.0],
        },
        {
            "name": "sunrise_s", "graph": "sunrise", "raised": "s", "fixed": {},
            "masses": [1.0, 1.1, 1.3], "range": [-10.0, 17.34], "control_range": [-10.0, -2.0],
        },
        {
            "name": "box_s", "graph": "box", "raised": "s",
            "fixed": four_fixed, "masses": [1.0, 1.2, 1.4, 1.6],
            "range": [four_start, 15.0], "control_range": [four_start, -2.0],
            "explicit_euclidean_start_momenta": four_momenta.tolist(),
        },
        {
            "name": "planar_double_box_s", "graph": "planar_double_box", "raised": "s",
            "fixed": four_fixed, "masses": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "range": [four_start, 20.0], "control_range": [four_start, -2.0],
            "explicit_euclidean_start_momenta": four_momenta.tolist(),
        },
        {
            "name": "crossed_double_box_s", "graph": "crossed_double_box", "raised": "s",
            "fixed": four_fixed, "masses": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "range": [four_start, 20.0], "control_range": [four_start, -2.0],
            "explicit_euclidean_start_momenta": four_momenta.tolist(),
        },
    ]


def real_roots(expression: sp.Expr, variable: sp.Symbol) -> list[float]:
    polynomial = sp.Poly(sp.expand(expression), variable)
    if polynomial.degree() <= 0:
        return []
    roots = []
    for root in sp.nroots(polynomial, maxsteps=200):
        if abs(float(sp.im(root))) < 1.0e-9:
            roots.append(float(sp.re(root)))
    return sorted(set(round(root, 12) for root in roots))


def path_invariant_expressions(config: dict[str, object]) -> tuple[sp.Symbol, dict[str, sp.Expr]]:
    variable = sp.Symbol(str(config["raised"]), real=True)
    invariants = {name: sp.Float(value) for name, value in config["fixed"].items()}
    invariants[str(config["raised"])] = variable
    return variable, invariants


def four_point_gram_from_invariants(invariants: dict[str, sp.Expr]) -> sp.Matrix:
    p1, p2, p3, p4 = (invariants[f"p{i}_sq"] for i in range(1, 5))
    dot12 = (invariants["s"] - p1 - p2) / 2
    dot23 = (invariants["t"] - p2 - p3) / 2
    dot13 = (p4 - p1 - p2 - p3) / 2 - dot12 - dot23
    return sp.Matrix([[p1, dot12, dot13], [dot12, p2, dot23], [dot13, dot23, p3]])


def symbolic_pair_q_squares(graph: CompiledGraph,
                            invariants: dict[str, sp.Expr]) -> dict[tuple[int, int], sp.Expr]:
    edge_count = len(graph.spec.edges)
    assert edge_count == len(graph.spec.vertices)
    result = {}
    for i, j in itertools.combinations(range(edge_count), 2):
        surviving = tuple(edge for edge in range(edge_count) if edge not in (i, j))
        comps = components(graph.spec.vertices, graph.spec.edges, surviving)
        assert len(comps) == 2
        result[(i, j)] = sp.sympify(q_square(graph.spec, comps[0], invariants))
    return result


def symbolic_cayley(masses_sq: list[float], q2: dict[tuple[int, int], sp.Expr],
                    subset: tuple[int, ...]) -> sp.Matrix:
    matrix = sp.zeros(len(subset))
    for local_i, edge_i in enumerate(subset):
        matrix[local_i, local_i] = sp.Float(masses_sq[edge_i])
        for local_j in range(local_i + 1, len(subset)):
            edge_j = subset[local_j]
            key = (min(edge_i, edge_j), max(edge_i, edge_j))
            value = (masses_sq[edge_i] + masses_sq[edge_j] - q2[key]) / 2
            matrix[local_i, local_j] = matrix[local_j, local_i] = value
    return matrix


def candidate_catalog(config: dict[str, object], graph: CompiledGraph) -> list[dict[str, object]]:
    variable, invariants = path_invariant_expressions(config)
    masses = [float(value) for value in config["masses"]]
    masses_sq = [value**2 for value in masses]
    candidates: list[dict[str, object]] = []

    def add(label: str, category: str, value: float | None, **details: object) -> None:
        candidates.append({"label": label, "category": category, "path_value": value, **details})

    if graph.spec.kinematic_kind == "two_point":
        gram_determinant = invariants["s"]
    elif graph.spec.kinematic_kind == "three_point":
        a, b, c = (invariants[f"p{i}_sq"] for i in range(1, 4))
        gram_determinant = a * b - ((c - a - b) / 2) ** 2
    else:
        gram_determinant = four_point_gram_from_invariants(invariants).det()
    for root in real_roots(gram_determinant, variable):
        add("external_Gram_det_zero", "second_type_or_Euclidean_boundary", root,
            polynomial=str(sp.factor(gram_determinant)))

    edge_count = len(graph.spec.edges)
    for cut_size in (2, 3):
        for cut in itertools.combinations(range(edge_count), cut_size):
            surviving = tuple(edge for edge in range(edge_count) if edge not in cut)
            comps = components(graph.spec.vertices, graph.spec.edges, surviving)
            if len(comps) != 2:
                continue
            if any(len(components(graph.spec.vertices, graph.spec.edges, surviving + (edge,))) != 1
                   for edge in cut):
                continue
            q2 = sp.expand(q_square(graph.spec, comps[0], invariants))
            cut_masses = [masses[edge] for edge in cut]
            signed_sums = []
            if cut_size == 2:
                signed_sums = [("pseudo", cut_masses[0] - cut_masses[1]),
                               ("normal", cut_masses[0] + cut_masses[1])]
            else:
                for signs in itertools.product((1.0, -1.0), repeat=2):
                    label = "normal" if signs == (1.0, 1.0) else "pseudo"
                    signed_sums.append((label, cut_masses[0] + signs[0] * cut_masses[1]
                                        + signs[1] * cut_masses[2]))
            for threshold_kind, signed_sum in signed_sums:
                threshold = signed_sum**2
                roots = real_roots(q2 - threshold, variable)
                cut_name = subset_name(cut)
                if roots:
                    for root in roots:
                        add(f"{cut_size}_particle_{threshold_kind}_threshold_{cut_name}",
                            f"{cut_size}_particle_{threshold_kind}_threshold", root,
                            cut_edges=[f"e{edge + 1}" for edge in cut], q_squared=str(q2),
                            threshold=threshold)
                else:
                    add(f"{cut_size}_particle_{threshold_kind}_threshold_{cut_name}",
                        f"{cut_size}_particle_{threshold_kind}_threshold", None,
                        cut_edges=[f"e{edge + 1}" for edge in cut], q_squared=str(q2),
                        threshold=threshold, path_relation="no isolated intersection on this one-parameter path")

    if edge_count == len(graph.spec.vertices):
        q2_pairs = symbolic_pair_q_squares(graph, invariants)
        full_subset = tuple(range(edge_count))
        for subset_size in range(2, edge_count + 1):
            for subset in itertools.combinations(range(edge_count), subset_size):
                determinant = sp.factor(symbolic_cayley(masses_sq, q2_pairs, subset).det())
                category = "leading_modified_Cayley" if subset == full_subset else "subdiagram_modified_Cayley"
                for root in real_roots(determinant, variable):
                    add(f"{category}_det_zero_{subset_name(subset)}", category, root,
                        propagators=[f"e{edge + 1}" for edge in subset], polynomial=str(determinant))

    for edge, mass_sq in enumerate(masses_sq):
        add(f"mass_squared_locus_e{edge + 1}", "mass_squared_type", mass_sq,
            propagator=f"e{edge + 1}", mass_squared=mass_sq)
    return candidates


def bisect_status_change(predicate: Callable[[float], bool], lo: float, hi: float,
                         left_status: bool) -> float:
    assert predicate(lo) == left_status and predicate(hi) != left_status
    while hi - lo > BISECT_TOL:
        mid = (lo + hi) / 2
        if predicate(mid) == left_status:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_path(config: dict[str, object], graph: CompiledGraph, rng: np.random.Generator,
             lower: float, upper: float, assert_control: bool,
             candidates: list[dict[str, object]]) -> dict[str, object]:
    directions = random_directions(rng, len(graph.spec.edges)) if graph.degree == 3 else None
    masses_sq = [float(mass) ** 2 for mass in config["masses"]]
    fixed = dict(config["fixed"])

    def full_evaluation(value: float) -> dict[str, object]:
        invariants = dict(fixed)
        invariants[str(config["raised"])] = value
        return evaluate(graph, invariants, masses_sq, directions)

    def predicate(value: float) -> bool:
        return bool(full_evaluation(value)["hessian_signature"])

    if "explicit_euclidean_start_momenta" in config and not assert_control:
        momenta = np.asarray(config["explicit_euclidean_start_momenta"], dtype=float)
        assert np.max(np.abs(np.sum(momenta, axis=0))) < 1.0e-12
        start_invariants = invariants_from_euclidean_momenta("four_point", momenta)
        assert abs(start_invariants[str(config["raised"])] - lower) < 1.0e-12
        assert all(abs(start_invariants[name] - value) < 1.0e-12 for name, value in fixed.items())
        start_gram_eigenvalues = np.linalg.eigvalsh(external_gram(momenta)).tolist()
        assert max(start_gram_eigenvalues) <= 1.0e-10
    else:
        start_gram_eigenvalues = None

    grid = np.linspace(lower, upper, PATH_GRID)
    statuses = [predicate(float(value)) for value in grid]
    transitions = []
    for index in range(1, len(grid)):
        if statuses[index] == statuses[index - 1]:
            continue
        boundary = bisect_status_change(predicate, float(grid[index - 1]), float(grid[index]),
                                        statuses[index - 1])
        transitions.append({"s": boundary,
                            "orientation": "pass_to_fail" if statuses[index - 1] else "fail_to_pass"})
    transition_boundary = next((item["s"] for item in transitions if item["orientation"] == "pass_to_fail"), None)
    first_failure: float | None = None
    if not statuses[0]:
        first_failure = lower
    elif transition_boundary is not None:
        first_failure = transition_boundary
    if assert_control:
        assert not transitions, f"Control path {config['name']} has a signature transition"

    numeric_candidates = [candidate for candidate in candidates if candidate["path_value"] is not None]
    nearest = None
    distance = None
    nearest_ties = []
    if first_failure is not None and numeric_candidates:
        nearest = min(numeric_candidates, key=lambda candidate: abs(first_failure - float(candidate["path_value"])))
        distance = abs(first_failure - float(nearest["path_value"]))
        nearest_ties = [candidate for candidate in numeric_candidates
                        if abs(abs(first_failure - float(candidate["path_value"])) - distance) < 1.0e-7]
    transition_nearest = None
    transition_distance = None
    if transition_boundary is not None and numeric_candidates:
        transition_nearest = min(numeric_candidates,
                                 key=lambda candidate: abs(transition_boundary - float(candidate["path_value"])))
        transition_distance = abs(transition_boundary - float(transition_nearest["path_value"]))
    start_eval, end_eval = full_evaluation(lower), full_evaluation(upper)
    standard_landau_candidates = [candidate for candidate in numeric_candidates
                                  if "threshold" in candidate["category"]]
    nearest_standard_landau = None
    nearest_standard_landau_distance = None
    if first_failure is not None and standard_landau_candidates:
        nearest_standard_landau = min(
            standard_landau_candidates,
            key=lambda candidate: abs(first_failure - float(candidate["path_value"])))
        nearest_standard_landau_distance = abs(
            first_failure - float(nearest_standard_landau["path_value"]))
    return {
        "graph": config["graph"], "raised_invariant": config["raised"], "fixed_invariants": fixed,
        "masses": config["masses"], "mass_squared_values": masses_sq,
        "range": [lower, upper], "grid_points": PATH_GRID, "orthant_directions": len(directions) if directions is not None else 0,
        "candidate_loci": candidates, "first_signature_failure": first_failure,
        "failure_at_scan_start": not statuses[0],
        "first_pass_to_fail_transition": transition_boundary,
        "nearest_candidate": nearest, "absolute_distance_to_nearest_candidate": distance,
        "nearest_candidate_to_transition": transition_nearest,
        "absolute_transition_distance_to_nearest_candidate": transition_distance,
        "nearest_candidate_ties": nearest_ties,
        "nearest_standard_normal_or_pseudo_threshold": nearest_standard_landau,
        "absolute_distance_to_nearest_standard_threshold": nearest_standard_landau_distance,
        "all_signature_transitions": transitions,
        "grid_signature_transitions": len(transitions), "transition_free": len(transitions) == 0,
        "all_grid_points_pass": all(statuses),
        "start_coefficient_nonnegative": start_eval["coefficient_nonnegative"],
        "end_coefficient_nonnegative": end_eval["coefficient_nonnegative"],
        "start_signature_pass": start_eval["hessian_signature"],
        "end_signature_pass": end_eval["hessian_signature"],
        "explicit_start_Gram_eigenvalues": start_gram_eigenvalues,
        "sunrise_nonstandard_onset_flag": ("onset at s = min m_i^2; not a standard normal/pseudo-threshold; "
                                            "candidate counterexample to the strong 'Lorentzian exactly on Landau-free region' "
                                            "conjecture at two loops; needs theory follow-up"
                                            if graph.spec.name == "sunrise" else None),
    }


def threshold_and_control_paths(graphs: dict[str, CompiledGraph], rng: np.random.Generator) -> tuple[dict[str, object], dict[str, object]]:
    thresholds: dict[str, object] = {}
    controls: dict[str, object] = {}
    for config in path_configurations():
        graph = graphs[str(config["graph"])]
        candidates = candidate_catalog(config, graph)
        lower, upper = (float(value) for value in config["range"])
        control_lower, control_upper = (float(value) for value in config["control_range"])
        thresholds[str(config["name"])] = run_path(config, graph, rng, lower, upper, False, candidates)
        controls[str(config["name"])] = run_path(config, graph, rng, control_lower, control_upper, True, candidates)
    return thresholds, controls


def verify_reviewer_legacy_double_box_transitions(graphs: dict[str, CompiledGraph]) -> dict[str, object]:
    fixed = {"p1_sq": -1.0, "p2_sq": -1.0, "p3_sq": -1.0, "p4_sq": -1.0, "t": -3.0}
    output = {
        "setup": {"masses": [1.0] * 7, "fixed_invariants": fixed, "scan_range": [-2.0, 1.0],
                  "warning": "Reviewer-requested verification only; this equal-mass path is not the corrected Experiment-B path and is outside the Euclidean region over part of the scan."}
    }
    for graph_name in ("planar_double_box", "crossed_double_box"):
        graph = graphs[graph_name]
        masses_sq = [1.0] * 7

        def predicate(value: float) -> bool:
            return bool(evaluate(graph, {**fixed, "s": value}, masses_sq)["hessian_signature"])

        grid = np.linspace(-2.0, 1.0, 1201)
        statuses = [predicate(float(value)) for value in grid]
        transitions = []
        for index in range(1, len(grid)):
            if statuses[index] == statuses[index - 1]:
                continue
            boundary = bisect_status_change(predicate, float(grid[index - 1]), float(grid[index]),
                                            statuses[index - 1])
            transitions.append({"s": boundary,
                                "orientation": "pass_to_fail" if statuses[index - 1] else "fail_to_pass"})
        assert len(transitions) == 2
        assert transitions[0]["orientation"] == "fail_to_pass" and abs(transitions[0]["s"] + 1.0) < 1.0e-6
        assert transitions[1]["orientation"] == "pass_to_fail" and abs(transitions[1]["s"]) < 1.0e-6
        output[graph_name] = {"transitions": transitions, "passed": True}
    return output


def graph_metadata(graph: CompiledGraph) -> dict[str, object]:
    return {
        "vertices": [f"v{v + 1}" for v in graph.spec.vertices],
        "edges": [{"name": f"e{i + 1}", "vertices": [f"v{a + 1}", f"v{b + 1}"]}
                  for i, (a, b) in enumerate(graph.spec.edges)],
        "leg_attachments": {f"p{i + 1}": f"v{vertex + 1}"
                            for i, vertex in enumerate(graph.spec.leg_attachments)},
        "description": graph.spec.description,
        "loop_number": len(graph.spec.edges) - len(graph.spec.vertices) + 1,
        "F_degree": graph.degree,
        "U": str(graph.U), "F0": str(graph.F0), "F": str(graph.F),
    }


def classify_kill_criteria(deep: dict[str, object], thresholds: dict[str, object], controls: dict[str, object]) -> dict[str, object]:
    simple_fractions = {name: deep[name]["hessian_signature_failure_fraction"] for name in ("bubble", "triangle")}
    criterion1_fired = any(float(value) >= 0.10 for value in simple_fractions.values())
    transition_items = [item for item in thresholds.values() if item["first_pass_to_fail_transition"] is not None]
    distances = [item["absolute_transition_distance_to_nearest_candidate"] for item in transition_items]
    missing = [name for name, item in thresholds.items() if item["first_pass_to_fail_transition"] is None]
    far = [float(distance) for distance in distances if float(distance) > 0.10]
    control_bad = [name for name, item in controls.items() if not item["transition_free"]]
    failed_at_start = [name for name, item in thresholds.items() if item["failure_at_scan_start"]]
    if far and len(far) >= max(1, len(thresholds) // 2):
        criterion2_status = "fired"
    elif far or missing or control_bad or failed_at_start:
        criterion2_status = "partially"
    else:
        criterion2_status = "not fired"
    return {
        "criterion_1": {
            "status": "fired" if criterion1_fired else "not fired",
            "rule": "Fires when bubble or triangle signature-failure fraction is at least 0.10.",
            "bubble_and_triangle_failure_fractions": simple_fractions,
        },
        "criterion_2": {
            "status": criterion2_status,
            "rule": "Fires when at least half the tested paths have a detected transition farther than 0.10 "
                    "from every listed candidate; partial if any path is far, lacks a pass-to-fail transition, "
                    "has a control transition, or already fails at the lower endpoint. A path already failing "
                    "at its lower endpoint is not called a transition.",
            "detected_transition_distances": distances,
            "paths_without_detected_transition": missing,
            "far_transition_distances": far,
            "bad_controls": control_bad,
            "paths_failing_at_scan_start": failed_at_start,
        },
        "literature_collision": {"status": "partially", "reason": "Not assessed by this computational test."},
    }


def reassess_criterion_1(criteria: dict[str, object], diagnostic: dict[str, object]) -> None:
    criterion = criteria["criterion_1"]
    raw_status = criterion["status"]
    triangle = diagnostic["triangle"]
    if triangle["total_signature_failures"] == 0:
        status = "not fired"
        reason = "No bubble or triangle primary-signature failures occur among the correctly sampled Gram-NSD Euclidean points."
    elif triangle["failures_fully_explained_by_det_Y_nonreference_sign"]:
        status = "not fired"
        reason = ("All sampled triangle failures, and no passes, lie on the negative-det(Y) side of the "
                  "intruding leading-Landau surface; bubble has no failures.")
    elif triangle["failures_on_det_Y_or_bubble_nonreference_side"] == triangle["total_signature_failures"]:
        status = "partially"
        reason = "All triangle failures lie on a leading or bubble-subdiagram nonreference side, but the sign predictor is not exact."
    else:
        status = raw_status
        reason = "Experiment D leaves triangle failures not accounted for by the tested Cayley sign diagnostics."
    criterion["experiment_A_raw_status"] = raw_status
    criterion["status"] = status
    criterion["experiment_D_reassessment_reason"] = reason


def print_summary(validation: dict[str, object], deep: dict[str, object], thresholds: dict[str, object],
                  controls: dict[str, object], diagnostic: dict[str, object], criteria: dict[str, object],
                  elapsed: float, output_path: Path) -> None:
    print(f"Validation gate: {'PASS' if validation['passed'] else 'FAIL'}")
    print("\nDeep-Euclidean sweep (coefficient / primary coordinate / supplementary random failure fractions):")
    print(f"{'graph':25s} {'coeff':>10s} {'primary':>12s} {'random-64':>12s}")
    for name, item in deep.items():
        print(f"{name:25s} {item['coefficient_nonnegativity_failure_fraction']:10.3f} "
              f"{item['hessian_signature_failure_fraction']:12.3f} "
              f"{item['supplementary_random_direction_failure_fraction']:12.3f}")
    print("\nThreshold paths:")
    print(f"{'path':25s} {'first fail':>14s} {'nearest value':>15s} {'distance':>12s}  identity")
    for name, item in thresholds.items():
        first = item["first_signature_failure"]
        nearest = item["nearest_candidate"]
        distance = item["absolute_distance_to_nearest_candidate"]
        first_text = "none" if first is None else f"{first:.8g}"
        nearest_text = "none" if nearest is None else f"{nearest['path_value']:.8g}"
        distance_text = "none" if distance is None else f"{distance:.3g}"
        identity = "none" if nearest is None else nearest["label"]
        print(f"{name:25s} {first_text:>14s} {nearest_text:>15s} {distance_text:>12s}  {identity}")
    print("  sunrise note: onset at s=min_i(m_i^2)=1 is not a standard normal/pseudo-threshold; candidate two-loop counterexample needing theory follow-up.")
    print("  legacy equal-mass double-box verification: fail->pass at s=-1 and pass->fail at s=0 for planar and crossed graphs.")
    print("\nM-convex support checks:")
    for name, item in validation["m_convex_support"].items():
        print(f"  {name:25s} PASS (support size {item['support_size']})")
    controls_transition_free = all(item["transition_free"] for item in controls.values())
    print("\nControl paths (no signature transition): " + ("PASS" if controls_transition_free else "FAIL"))
    print("\nexperiment_D: Cayley/Landau audit")
    for graph_name in ("triangle", "box"):
        item = diagnostic[graph_name]
        print(f"  {graph_name} det(Y) sign cross-tab (pass/fail):")
        for sign, counts in item["det_Y_sign_confusion"].items():
            print(f"    {sign:8s} {counts['signature_pass']:3d}/{counts['signature_fail']:3d}")
        best = item["best_single_sign_predictor"]
        print(f"    best sign predictor: {best['feature']}={best['predict_failure_when_sign_is']}; "
              f"misclassified={best['misclassified']}")
    path_start = diagnostic["box"]["experiment_B_path_start"]
    print(f"  box path start det(Y)={path_start['Y']['determinant']:.8g} "
          f"({path_start['Y']['determinant_sign']}); leading nonreference="
          f"{path_start['leading_Landau_nonreference_or_surface']}")
    print("  u-channel restricted signature-failure fractions:")
    for graph_name, item in diagnostic["u_channel_audit"].items():
        negative = item["u_negative_fully_euclidean"]
        nonnegative = item["u_nonnegative"]
        negative_fraction = "none" if negative["signature_failure_fraction"] is None else f"{negative['signature_failure_fraction']:.3f}"
        nonnegative_fraction = "none" if nonnegative["signature_failure_fraction"] is None else f"{nonnegative['signature_failure_fraction']:.3f}"
        print(f"    {graph_name:22s} u<0 {negative_fraction} "
              f"({negative['signature_failures']}/{negative['samples']}), u>=0 "
              f"{nonnegative_fraction} "
              f"({nonnegative['signature_failures']}/{nonnegative['samples']}), "
              f"u>=0 carries most={item['u_nonnegative_subsample_carries_most_failures']}")
    print("\nKill-criterion mapping:")
    print(f"  criterion 1 (generic Euclidean failure): {criteria['criterion_1']['status']}")
    print(f"  criterion 2 (uncorrelated transitions):  {criteria['criterion_2']['status']}")
    print(f"  literature collision:                   {criteria['literature_collision']['status']}")
    print(f"\nWall-clock time: {elapsed:.3f} s")
    print(f"JSON: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    print(f"seed={args.seed}")
    print(f"python={platform.python_version()} sympy={sp.__version__} numpy={np.__version__}")

    rng = np.random.default_rng(args.seed)
    graphs = {spec.name: build_symanzik(spec) for spec in graph_specs()}
    validation = validation_gate(graphs)
    retained_samples: dict[str, list[dict[str, object]]] = {}
    deep = deep_euclidean_sweep(graphs, rng, retained_samples)
    assert all(item["primary_coordinate_signature_failures"] == 0 for item in deep.values()), \
        "Nonzero primary failure fraction in the true Gram-NSD Euclidean region; stop and investigate"
    thresholds, controls = threshold_and_control_paths(graphs, rng)
    sunrise_onset = thresholds["sunrise_s"]["first_signature_failure"]
    assert sunrise_onset is not None and abs(sunrise_onset - 1.0) < 1.0e-6
    legacy_transition_check = verify_reviewer_legacy_double_box_transitions(graphs)
    diagnostic = experiment_d(graphs, retained_samples)
    criteria = classify_kill_criteria(deep, thresholds, controls)
    reassess_criterion_1(criteria, diagnostic)
    elapsed = time.perf_counter() - started

    output = {
        "run_label": "run 2: corrected Euclidean sampling and coordinate-partial Lorentzian test",
        "seed": args.seed,
        "package_versions": {"python": platform.python_version(), "sympy": sp.__version__, "numpy": np.__version__},
        "constants": {
            "coefficient_tolerance": COEFF_TOL, "eigenvalue_tolerance_factor": EIGEN_TOL_FACTOR,
            "supplementary_orthant_directions_per_cubic_test": ORTHANT_DIRECTIONS, "deep_samples_per_graph": DEEP_SAMPLES,
            "path_grid_points": PATH_GRID, "bisection_tolerance": BISECT_TOL,
            "cayley_sign_tolerance": CAYLEY_SIGN_TOL,
        },
        "conventions": {
            "metric": "Minkowski (+,-,-,-)",
            "euclidean_region": "external momenta realizable as Euclidean vectors; equivalently their Minkowski Gram matrix is negative semidefinite",
            "F0_sign": "each 2-forest contributes (-q_F^2) times the complementary edge monomial",
            "four_point_u": "u=sum_i p_i^2-s-t",
        },
        "graphs": {name: graph_metadata(graph) for name, graph in graphs.items()},
        "validation_gate": validation,
        "reviewer_legacy_double_box_transition_verification": legacy_transition_check,
        "experiment_A_deep_euclidean": deep,
        "experiment_B_threshold_paths": thresholds,
        "experiment_C_control_paths": controls,
        "experiment_D": diagnostic,
        "candidate_catalog_conventions": {
            "standard_cut_loci": "All minimal 2- and 3-edge cuts are enumerated; normal and signed pseudo-threshold equations q_cut^2=(signed mass sum)^2 are solved on each path.",
            "modified_Cayley_scope": "Leading and all principal subdiagram modified-Cayley roots are included for one-loop graphs (E=V). No artificial single-loop Cayley matrix is assigned to a genuine two-loop double box; its candidate catalog instead contains external-Gram, graph-cut threshold, and m_e^2-type loci.",
            "ambiguity_resolution": "The reviewer-requested -1 and 0 double-box transitions are verified separately on the old equal-mass/non-Euclidean setup; corrected Experiment B uses nondegenerate masses and an explicit Gram-NSD start, so its transition values need not remain -1 and 0.",
        },
        "kill_criteria": criteria,
        "wall_clock_seconds": elapsed,
    }
    output_path = Path(__file__).resolve().parent / f"results-run2-seed{args.seed}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(validation, deep, thresholds, controls, diagnostic, criteria, elapsed, output_path)


if __name__ == "__main__":
    main()
