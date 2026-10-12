"""
EN3330 Assignment 03 -- Lagrangian duality, saddle points and weak duality.

Boralugoda M.S. (220074B)

Problem:  min_x x^2  subject to  h(x) = 1 - x <= 0
Lagrangian:  L(mu, x) = x^2 + mu * (1 - x),  mu >= 0

Running this script regenerates every figure used in the report
(saved as PDF in ./figures/) and prints the numerical checks.

Requirements: numpy, matplotlib
Usage:        python en3330_a03_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "savefig.bbox": "tight",
})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------- model
def f(x):
    """Objective f(x) = x^2."""
    return x ** 2


def h(x, a=1.0):
    """Constraint h(x) = a - x <= 0  (i.e. x >= a). a = 1 for the main problem."""
    return a - x


def lagrangian(mu, x, a=1.0):
    """L(mu, x) = f(x) + mu * h(x)."""
    return f(x) + mu * h(x, a)


def x_min(mu):
    """argmin_x L(mu, x): from dL/dx = 2x - mu = 0."""
    return mu / 2.0


def dual(mu, a=1.0):
    """d(mu) = min_x L(mu, x), evaluated at the analytic minimiser."""
    return lagrangian(mu, x_min(mu), a)


def dual_numeric(mu, a=1.0, grid=np.linspace(-10, 10, 200001)):
    """d(mu) found by brute-force minimisation over a fine x-grid (cross-check)."""
    return np.min(lagrangian(mu, grid, a))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved figures/{name}.pdf")


# ------------------------------------------------------- Q1(a)
def fig_q1a():
    x = np.linspace(-1.5, 3.2, 400)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, mu in enumerate([0, 1, 2, 3]):
        ax.plot(x, lagrangian(mu, x), color=COLORS[i], lw=1.8, label=rf"$\mu={mu}$")
        xm = x_min(mu)
        ax.plot(xm, lagrangian(mu, xm), "o", color=COLORS[i], ms=7, zorder=5)
        ax.annotate(rf"$x({mu})={xm:g}$", (xm, lagrangian(mu, xm)),
                    textcoords="offset points", xytext=(4, -15), fontsize=9,
                    color=COLORS[i])
    mus = np.linspace(0, 3, 100)
    ax.plot(x_min(mus), dual(mus), "k--", lw=1.2,
            label=r"path of $(x(\mu),\,d(\mu))$")
    ax.axvline(1, color="gray", ls=":", lw=1.2)
    ax.text(1.05, 5.5, r"$x=1$ (constraint)", color="gray", fontsize=9)
    ax.set(xlim=(-1.5, 3.2), ylim=(-0.5, 6), xlabel=r"$x$",
           ylabel=r"$L(\mu,x)$")
    ax.legend(loc="upper left")
    save(fig, "q1a_lagrangian_vs_x")


# ------------------------------------------------------- Q1(b)
def fig_q1b():
    mu = np.linspace(0, 5.2, 400)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(mu, dual(mu), color=COLORS[0], lw=2.2, label=r"$d(\mu)=\mu-\mu^2/4$")
    ax.axhline(1, color=COLORS[3], ls="--", lw=1.4, label=r"$p^*=1$")
    pts = np.array([0, 1, 3])
    ax.plot(pts, dual(pts), "o", mfc="white", mec=COLORS[1], mew=1.6, ms=7,
            label=r"$\mu=0,1,3$")
    ax.plot(2, 1, "ko", ms=8, zorder=5)
    ax.annotate(r"$(\mu^*,d^*)=(2,1)$", (2, 1), xytext=(2.25, 1.15), fontsize=10)
    ax.vlines(2, -1.5, 1, colors="gray", linestyles=":")
    ax.set(xlim=(0, 5.2), ylim=(-1.5, 1.6), xlabel=r"$\mu$", ylabel=r"$d(\mu)$")
    ax.legend(loc="lower left")
    save(fig, "q1b_dual_function")


# ------------------------------------------------------- Q1(c)
def fig_q1c():
    x = np.linspace(-1, 3, 600)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvspan(-1, 1, color=COLORS[3], alpha=0.08)
    for i, mu in enumerate([0, 1, 2, 5, 10, 50]):
        ax.plot(x, lagrangian(mu, x), color=COLORS[i], lw=1, label=rf"$\mu={mu}$")
    # numerical max over a large mu grid -> approximates g(x)
    mu_grid = np.linspace(0, 1e4, 2001)
    g_num = np.max(lagrangian(mu_grid[:, None], x[None, :]), axis=0)
    feas = x >= 1
    ax.plot(x[feas], g_num[feas], "k", lw=3, label=r"$g(x)$")
    ax.annotate("", xy=(1, 11.7), xytext=(1, 1),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
    for xa in (-0.5, 0.5):
        ax.annotate("", xy=(xa, 11.5), xytext=(xa, 1.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, ls="--", color="k"))
    ax.plot(1, 1, "ko", ms=8, zorder=5)
    ax.text(1.08, 0.2, r"$\min g = g(1)=1$", fontsize=10)
    ax.text(0, 11, r"infeasible: $g=+\infty$", ha="center", color="darkred")
    ax.text(2, 11, r"feasible: $g=x^2$", ha="center", color="darkgreen")
    ax.set(xlim=(-1, 3), ylim=(-2, 12), xlabel=r"$x$", ylabel="value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "q1c_primal_function")


# ------------------------------------------------------- Q2(a)(iv)
def fig_q2a():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    mu = np.linspace(0, 5, 200)
    a1.plot(mu, lagrangian(mu, 1.0), color=COLORS[0], lw=2)
    a1.plot(2, 1, "o", color=COLORS[3], ms=8)
    a1.annotate(r"$L(\mu^*,x^*)=1$", (2, 1), xytext=(1.3, 1.3))
    a1.set(xlim=(0, 5), ylim=(0, 3), xlabel=r"$\mu$", ylabel=r"$L(\mu,x^*)$",
           title=r"$L(\mu,1)=1$ (flat in $\mu$)")
    x = np.linspace(-1, 3, 200)
    a2.plot(x, lagrangian(2, x), color=COLORS[2], lw=2)
    a2.plot(1, 1, "o", color=COLORS[3], ms=8)
    a2.annotate(r"$L(\mu^*,x^*)=1$", (1, 1), xytext=(0.6, 0.35))
    a2.set(xlim=(-1, 3), ylim=(0, 5), xlabel=r"$x$", ylabel=r"$L(\mu^*,x)$",
           title=r"$L(2,x)=(x-1)^2+1$")
    fig.tight_layout()
    save(fig, "q2a_saddle_slices")


# ------------------------------------------------------- Q2(c)
def fig_q2c():
    mu = np.linspace(0, 4, 200)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(mu, dual(mu, a=-1.0), color=COLORS[0], lw=2,
            label=r"$d(\mu)=-\mu-\mu^2/4$")
    ax.plot(0, 0, "o", color=COLORS[3], ms=8, zorder=5)
    ax.annotate(r"$\mu^*=0,\ d^*=0$", (0, 0), xytext=(0.15, 0.3))
    ax.set(xlim=(0, 4), ylim=(-8.5, 1), xlabel=r"$\mu$", ylabel=r"$d(\mu)$")
    ax.legend(loc="lower left")
    save(fig, "q2c_dual_inactive")


# ------------------------------------------------------- Q3(b)
def fig_q3b():
    mu = np.linspace(0, 4.5, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    lines = []
    for i, xv in enumerate([0, 1, 2]):
        Lx = lagrangian(mu, xv)
        lines.append(Lx)
        expr = {0: r"\mu", 1: "1", 2: r"4-\mu"}[xv]
        ax.plot(mu, Lx, color=COLORS[i], lw=1.5, label=rf"$x={xv}$: $L={expr}$")
    env = np.min(lines, axis=0)
    ax.plot(mu, env, color="k", lw=4, alpha=0.45,
            label=r"lower envelope $\tilde d(\mu)$")
    ax.plot(mu, dual(mu), color=COLORS[3], ls="--", lw=2,
            label=r"true $d(\mu)=\mu-\mu^2/4$")
    ax.plot(2, 1, "ko", ms=6)
    ax.set(xlim=(0, 4.5), ylim=(-0.6, 4.2), xlabel=r"$\mu$", ylabel=r"$L(\mu,x)$")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "q3b_lower_envelope")


# ------------------------------------------------------- numerical checks
def checks():
    print("\nNumerical checks")
    print(f"{'mu':>4} {'x(mu)':>7} {'d analytic':>11} {'d numeric':>10} "
          f"{'d<=p*=1':>8} {'gap':>6}")
    for mu in [0, 1, 2, 3]:
        da, dn = dual(mu), dual_numeric(mu)
        print(f"{mu:>4} {x_min(mu):>7.2f} {da:>11.4f} {dn:>10.4f} "
              f"{str(da <= 1 + 1e-12):>8} {1 - da:>6.2f}")
    mu = np.linspace(0, 10, 1001)
    star = mu[np.argmax(dual(mu))]
    print(f"\nmax of d over mu in [0,10]: mu* = {star:.3f}, d* = {dual(star):.4f}")
    xs = np.linspace(-5, 5, 10001)
    print(f"L(mu,1) for mu in 0..10 : all equal 1 -> "
          f"{np.allclose(lagrangian(np.arange(11), 1.0), 1)}")
    print(f"min_x L(2,x) = {lagrangian(2, xs).min():.4f} at x = "
          f"{xs[np.argmin(lagrangian(2, xs))]:.3f}")
    print(f"Concavity check: max second difference of d = "
          f"{np.max(np.diff(dual(mu), 2)):.2e} (<= 0)")


if __name__ == "__main__":
    print("Generating figures ...")
    fig_q1a()
    fig_q1b()
    fig_q1c()
    fig_q2a()
    fig_q2c()
    fig_q3b()
    checks()
