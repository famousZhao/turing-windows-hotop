"""Reaction-diffusion system: Schnakenberg kinetics, linear stability/dispersion, time integration.

Schnakenberg model
    du/dt = a - u + u^2 v + Du * (L u)
    dv/dt = b - u^2 v     + Dv * (L v)
    homogeneous equilibrium: u* = a+b,  v* = b/(a+b)^2

Linear stability (essential; every case has an analytic reference)
    For each Laplacian eigenvalue Λ (eigenvector v_k) the perturbation growth rate is
        J(Λ) = [[f_u - Du*Λ,  f_v],
                [g_u,         g_v - Dv*Λ]]
        μ(Λ) = real part of the leading eigenvalue
    Turing instability criterion: the uniform state is stable (tr J(0)<0, det J(0)>0),
    yet some Λ gives det J(Λ) < 0.

With the default parameters (a,b)=(0.1,0.9), Du=1, Dv=10,
det J(Λ) = 10Λ² - 7Λ + 1 < 0  is equivalent to  0.2 < Λ < 0.5
This band is the project's yardstick; every numerical result must reproduce it.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class Schnakenberg:
    def __init__(self, a: float = 0.1, b: float = 0.9,
                 Du: float = 1.0, Dv: float = 10.0):
        self.a, self.b, self.Du, self.Dv = a, b, Du, Dv

    # kinetics
    @property
    def ustar(self) -> float:
        return self.a + self.b

    @property
    def vstar(self) -> float:
        return self.b / (self.a + self.b) ** 2

    def reaction(self, u: np.ndarray, v: np.ndarray):
        uvv = u * u * v
        return (self.a - u + uvv, self.b - uvv)

    def jacobian(self) -> np.ndarray:
        u, v = self.ustar, self.vstar
        return np.array([
            [-1.0 + 2.0 * u * v, u * u],
            [-2.0 * u * v, -u * u],
        ])

    # linear analysis
    def homogeneous_stable(self) -> tuple[bool, float, float]:
        J = self.jacobian()
        tr, det = float(np.trace(J)), float(np.linalg.det(J))
        return (tr < 0 and det > 0), tr, det

    def growth_rate(self, lam: float) -> float:
        """For a given Laplacian eigenvalue Λ, return the growth rate of that mode (Re μ)."""
        J = self.jacobian()
        M = J.copy()
        M[0, 0] -= self.Du * lam
        M[1, 1] -= self.Dv * lam
        return float(np.max(np.linalg.eigvals(M).real))

    def det(self, lam: float) -> float:
        J = self.jacobian()
        M = J.copy()
        M[0, 0] -= self.Du * lam
        M[1, 1] -= self.Dv * lam
        return float(np.linalg.det(M))

    def turing_band(self) -> tuple[float, float] | None:
        """Analytically solve for the Λ bounds of the Turing unstable band (roots of det(Λ)=0)."""
        J = self.jacobian()
        # det = (f_u - Du Λ)(g_v - Dv Λ) - f_v g_u  = Du*Dv Λ² - (f_u Dv + g_v Du)Λ + (f_u g_v - f_v g_u)
        A = self.Du * self.Dv
        B = -(J[0, 0] * self.Dv + J[1, 1] * self.Du)
        C = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        disc = B * B - 4 * A * C
        if disc <= 0 or A <= 0:
            return None
        r1 = (-B - np.sqrt(disc)) / (2 * A)
        r2 = (-B + np.sqrt(disc)) / (2 * A)
        if r2 <= 0:
            return None
        return (max(0.0, r1), r2)

    def spectrum_analysis(self, L) -> dict:
        """Full dispersion analysis for a given Laplacian."""
        dense = L.toarray() if sparse.issparse(L) else np.asarray(L)
        lam, vecs = np.linalg.eigh(dense)
        mu = np.array([self.growth_rate(l) for l in lam])
        order = np.argsort(-mu)
        k = int(order[0])
        band = self.turing_band()
        unstable = mu > 1e-10
        return {
            "lam": lam,
            "vecs": vecs,
            "mu": mu,
            "lam_star": float(lam[k]),
            "mu_star": float(mu[k]),
            "k_star": k,
            "n_unstable": int(unstable.sum()),
            "unstable_lam": lam[unstable],
            "turing_band": band,
            "hom_stable": self.homogeneous_stable(),
        }


# time integration
def rk4_step(L, f, u, v, dt, Du, Dv):
    """
    Fixed-step RK4. L is the standard graph Laplacian, positive semi-definite.

    The diffusion term carries a minus sign, du_i/dt = f(u_i, v_i) - Du (L u)_i, which
    equals f + Du sum_j A_ij (u_j - u_i), genuine diffusion. A uniform field satisfies
    L 1 = 0, so diffusion does not destroy the uniform state.
    """
    def rhs(uu, vv):
        ru, rv = f(uu, vv)
        return ru - Du * (L @ uu), rv - Dv * (L @ vv)

    k1u, k1v = rhs(u, v)
    k2u, k2v = rhs(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v)
    k3u, k3v = rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
    k4u, k4v = rhs(u + dt * k3u, v + dt * k3v)
    return (u + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u),
            v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))


def integrate(L, model: Schnakenberg, u0, v0, t_end, dt, record_every=0, seed=0):
    """Integrate on a fixed structure. Return (u, v, trajectory)."""
    n_steps = int(round(t_end / dt))
    rec = []
    u, v = u0.copy(), v0.copy()
    for s in range(n_steps):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        if record_every and (s + 1) % record_every == 0:
            rec.append((s * dt, u.copy(), v.copy()))
    return u, v, rec


def perturbed_ic(n: int, model: Schnakenberg, amp: float = 0.01, seed: int = 0):
    """Add small random perturbations around the uniform state."""
    rng = np.random.default_rng(seed)
    u = model.ustar + amp * rng.standard_normal(n)
    v = model.vstar + amp * rng.standard_normal(n)
    return u, v


def modal_energy(x: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Project field x onto the Laplacian eigenvectors; return the modal energy fraction."""
    c = vecs.T @ x
    e = c**2
    return e / (e.sum() + 1e-300)


if __name__ == "__main__":
    m = Schnakenberg()
    print("=== Schnakenberg linear stability check ===")
    print(f"u* = {m.ustar:.4f}, v* = {m.vstar:.4f}")
    st, tr, det = m.homogeneous_stable()
    print(f"uniform state stable? {st}  (tr={tr:.4f}, det={det:.4f})  [expected True]")
    band = m.turing_band()
    print(f"Turing unstable band Λ ∈ {band}  [expected (0.2, 0.5)]")
    print(f"  check: growth(0.1)={m.growth_rate(0.1):.5f} (expected <0), "
          f"growth(0.35)={m.growth_rate(0.35):.5f} (expected >0), "
          f"growth(0.8)={m.growth_rate(0.8):.5f} (expected <0)")

    # analytic vs numerical: verified on a 1D ring
    from complex import SimplicialComplex

    for n in (20, 40, 80):
        edges = [(i, (i + 1) % n) for i in range(n)]
        sc = SimplicialComplex.from_edges(n, edges)
        L = sc.L0()
        an = m.spectrum_analysis(L)
        lam_in = [l for l in an["unstable_lam"]]
        print(f"ring n={n}: unstable eigenvalues={an['n_unstable']}, "
              f"Λ*={an['lam_star']:.4f}, μ*={an['mu_star']:.4f}")
