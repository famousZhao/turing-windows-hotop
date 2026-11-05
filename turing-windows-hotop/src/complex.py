"""Simplicial complexes: construction, boundary operators, Laplacian variants.

Conventions
    Nodes 0..n-1; edges are ascending 2-tuples; triangles are ascending 3-tuples.
    Boundary operator B_k: C_k -> C_{k-1}, shape (dim C_{k-1}) x (dim C_k).

    B1: (n, n_e), edge (i,j), i<j -> B1[i,e] = -1, B1[j,e] = +1
        so L0 = B1 @ B1.T is the graph Laplacian (degree on the diagonal, -1 off it).
        Note: this is the "covariant" convention, same sign as the usual graph Laplacian.

    B2: (n_e, n_t), boundary of triangle (i,j,k) = [j,k] - [i,k] + [i,j]

    The topological identity B1 @ B2 = 0 must hold; it is the core unit-test assertion.

Laplacians
    L0 = B1 B1^T              (n x n)     graph (down) Laplacian, PSD
    L1 = B1^T B1 + B2 B2^T    (n_e x n_e) Hodge Laplacian on edges, PSD
    L2 = B2^T B2              (n_t x n_t)
    L_tri (node-level higher-order Laplacian):
        L_tri[i,i] = 2 * (number of triangles containing i)
        L_tri[i,j] = -(number of triangles containing both i and j)
    This operator is the sum of the clique Laplacians of all triangles, hence PSD
    with zero row sums.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import sparse


class SimplicialComplex:
    def __init__(self, n_nodes: int = 0):
        self.n = n_nodes
        self.edges: list[tuple[int, int]] = []
        self.triangles: list[tuple[int, int, int]] = []
        self._eidx: dict[tuple[int, int], int] = {}
        self._tset: set[tuple[int, int, int]] = set()

    @classmethod
    def from_triangles(cls, triangles, n_nodes: int | None = None) -> "SimplicialComplex":
        """Build a complex from a triangle list, filling in edges and nodes (closure)."""
        tris = sorted({tuple(sorted(t)) for t in triangles})
        nodes = set()
        edges = set()
        for t in tris:
            nodes.update(t)
            edges.update(combinations(t, 2))
        n = n_nodes if n_nodes is not None else (max(nodes) + 1 if nodes else 0)
        c = cls(n)
        for e in sorted(edges):
            c.add_edge(*e)
        for t in tris:
            c.add_triangle(*t)
        return c

    @classmethod
    def from_edges(cls, n_nodes: int, edges) -> "SimplicialComplex":
        c = cls(n_nodes)
        for e in sorted({tuple(sorted(e)) for e in edges}):
            c.add_edge(*e)
        return c

    def add_node(self, i: int) -> None:
        if i >= self.n:
            self.n = i + 1

    def add_edge(self, i: int, j: int) -> int:
        if i == j:
            raise ValueError("self-loops are not allowed")
        e = (i, j) if i < j else (j, i)
        if e not in self._eidx:
            self._eidx[e] = len(self.edges)
            self.edges.append(e)
        self.add_node(max(e))
        return self._eidx[e]

    def add_triangle(self, i: int, j: int, k: int) -> int:
        t = tuple(sorted((i, j, k)))
        if t not in self._tset:
            self._tset.add(t)
            for a, b in combinations(t, 2):
                self.add_edge(a, b)
            self.triangles.append(t)
        return len(self.triangles) - 1

    def B1(self) -> sparse.csr_matrix:
        """(n, n_e) node-edge incidence matrix."""
        ne = len(self.edges)
        if ne == 0:
            return sparse.csr_matrix((self.n, 0))
        rows, cols, vals = [], [], []
        for e, (i, j) in enumerate(self.edges):
            rows += [i, j]
            cols += [e, e]
            vals += [-1.0, 1.0]
        return sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, ne))

    def B2(self) -> sparse.csr_matrix:
        """(n_e, n_t) edge-triangle incidence matrix."""
        nt = len(self.triangles)
        ne = len(self.edges)
        if nt == 0:
            return sparse.csr_matrix((ne, 0))
        rows, cols, vals = [], [], []
        for t, (i, j, k) in enumerate(self.triangles):
            rows += [self._eidx[(j, k)], self._eidx[(i, k)], self._eidx[(i, j)]]
            cols += [t, t, t]
            vals += [1.0, -1.0, 1.0]
        return sparse.csr_matrix((vals, (rows, cols)), shape=(ne, nt))

    def L0(self) -> sparse.csr_matrix:
        B1 = self.B1()
        return (B1 @ B1.T).tocsr()

    def L1(self) -> sparse.csr_matrix:
        B1, B2 = self.B1(), self.B2()
        return (B1.T @ B1 + B2 @ B2.T).tocsr()

    def L2(self) -> sparse.csr_matrix:
        B2 = self.B2()
        return (B2.T @ B2).tocsr()

    def L_tri(self) -> sparse.csr_matrix:
        """Node-level higher-order (triangle) Laplacian. More triangles, stronger term."""
        if not self.triangles:
            return sparse.csr_matrix((self.n, self.n))
        cnt_node = np.zeros(self.n)
        cnt_pair: dict[tuple[int, int], float] = {}
        for (i, j, k) in self.triangles:
            cnt_node[[i, j, k]] += 1
            for a, b in ((i, j), (i, k), (j, k)):
                cnt_pair[(a, b)] = cnt_pair.get((a, b), 0.0) + 1.0
        diag = 2.0 * cnt_node
        rows, cols, vals = [], [], []
        for (a, b), c in cnt_pair.items():
            rows += [a, b]
            cols += [b, a]
            vals += [-c, -c]
        Lp = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return (Lp + sparse.diags(diag)).tocsr()

    def laplacian(self, order_strength: float = 0.0) -> sparse.csr_matrix:
        """Operator for node-level reaction-diffusion: L = L0 + order_strength * L_tri.

        order_strength = 0 reduces to the classic graph Turing problem (comparable to the literature).
        """
        L = self.L0()
        if order_strength > 0:
            L = L + order_strength * self.L_tri()
        return L.tocsr()

    def stats(self) -> dict:
        B1, B2 = self.B1(), self.B2()
        L0, L1 = self.L0(), self.L1()
        ev0 = np.linalg.eigvalsh(L0.toarray()) if self.n <= 1200 else None
        ev1 = np.linalg.eigvalsh(L1.toarray()) if L1.shape[0] <= 1200 else None
        # topological quantities: Betti numbers from matrix ranks
        b0 = self.n - np.linalg.matrix_rank(B1.toarray()) if self.n <= 2000 else -1
        b1 = (len(self.edges) - np.linalg.matrix_rank(B1.toarray())
              - np.linalg.matrix_rank(B2.toarray())) if self.n <= 2000 else -1
        return {
            "n_nodes": self.n,
            "n_edges": len(self.edges),
            "n_triangles": len(self.triangles),
            "b0": int(b0) if b0 >= 0 else -1,
            "b1": int(b1) if b1 >= 0 else -1,
            "eig_L0_min": float(ev0.min()) if ev0 is not None else np.nan,
            "eig_L0_max": float(ev0.max()) if ev0 is not None else np.nan,
            "eig_L1_min": float(ev1.min()) if ev1 is not None else np.nan,
            "eig_L1_max": float(ev1.max()) if ev1 is not None else np.nan,
            "ker_L1_dim": int((np.abs(ev1) < 1e-8).sum()) if ev1 is not None else -1,
        }


def check_boundary_identity(c: SimplicialComplex) -> float:
    """Return max|B1 @ B2|, which must be 0 (topological identity)."""
    if not c.triangles:
        return 0.0
    return float(abs(c.B1() @ c.B2()).max())


if __name__ == "__main__":
    # Verification cases: a ring and a triangle, results checkable by hand
    print("=== Simplicial complex verification ===")
    # triangle made of 3 nodes
    c = SimplicialComplex.from_triangles([(0, 1, 2)])
    print("single triangle:", c.stats())
    print("B1@B2 max|·| =", check_boundary_identity(c), "(must be 0)")
    print("L0 =", c.L0().toarray())
    print("L_tri =", c.L_tri().toarray())
    print("L0 @ 1 =", c.L0().toarray() @ np.ones(3), "(must be 0)")
    print("L_tri @ 1 =", c.L_tri().toarray() @ np.ones(3), "(must be 0)")
    print("L_tri eigenvalues =", np.linalg.eigvalsh(c.L_tri().toarray()), "(must all be >= 0)")

    # two triangles sharing an edge => B1 B2 = 0
    c2 = SimplicialComplex.from_triangles([(0, 1, 2), (1, 2, 3)])
    print("\ntwo triangles:", c2.stats())
    print("B1@B2 max|·| =", check_boundary_identity(c2), "(must be 0)")

    # ring C4: b1 should be 1
    c3 = SimplicialComplex.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    print("\nring C4:", c3.stats(), "(b1 should be 1)")
