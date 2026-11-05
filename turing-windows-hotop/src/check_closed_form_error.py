"""Compare the closed form with the exact minimal size, both continuously and as integers.

For every banded parameter set of Table 4 and all three structures:
  exact n_min      -- scanned from the exact window condition, f(2*pi*k/n) in (lam-, lam+)
  exact continuous -- 2*pi/theta_+, with theta_+ the exact root of f(theta) = lam+
  closed form      -- 2*pi*sqrt(C/lam+)
and the two possible notions of "error" are reported separately:
  |diff_cont| = |closed form - exact continuous|
  diff_int    = ceil(closed form) - exact n_min
"""
import math

import numpy as np

SYMBOLS = {
    "ring1": (lambda t: 2 - 2 * np.cos(t), 1),
    "ring2": (lambda t: 4 - 2 * np.cos(t) - 2 * np.cos(2 * t), 5),
    "simpl": (lambda t: 10 - 6 * np.cos(t) - 4 * np.cos(2 * t), 11),
}
SYMBOLS["ring3"] = (
    lambda t: sum(2 * (1 - np.cos(d * t)) for d in (1, 2, 3)),
    sum(d * d for d in (1, 2, 3)),
)

BANDS = [
    ("set 1", 0.200, 0.500),
    ("set 2", 0.074, 0.676),
    ("set 3", 0.179, 0.615),
    ("set 4", 0.148, 1.352),
    ("set 5", 0.100, 0.250),
]


def exact_nmin(f, lo, hi, nmax=4000):
    for n in range(3, nmax + 1):
        k = np.arange(1, n)
        vals = f(2 * np.pi * k / n)
        if np.any((vals > lo) & (vals < hi)):
            return n
    return None


def theta_plus(f, lam):
    a, b = 0.0, np.pi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(np.array([m]))[0] < lam:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


print(f"{'case':<10}{'lam+':>7}{'exact':>7}{'closed':>9}{'ceil':>6}"
      f"{'diff_int':>9}{'exact_cont':>11}{'diff_cont':>10}")
print("-" * 70)

records = []
for label, lo, hi in BANDS:
    for name, (f, C) in SYMBOLS.items():
        if name == "ring3" and label != "set 1":
            continue
        n_ex = exact_nmin(f, lo, hi)
        closed = 2 * np.pi * math.sqrt(C / hi)
        cont = 2 * np.pi / theta_plus(f, hi)
        di = math.ceil(closed - 1e-12) - n_ex
        dc = closed - cont
        records.append((label, name, hi, n_ex, closed, di, cont, dc))
        print(f"{label + ' ' + name:<10}{hi:>7.3f}{n_ex:>7}{closed:>9.3f}"
              f"{math.ceil(closed - 1e-12):>6}{di:>9}{cont:>11.3f}{dc:>10.3f}")

print("\n=== summary ===")
ints = [r[5] for r in records]
conts = [abs(r[7]) for r in records]
print(f"  integer differences: {sorted(set(ints))}, "
      f"nonzero in {sum(1 for d in ints if d)}/{len(ints)} cases")
print(f"  continuous |diff| max = {max(conts):.3f}  "
      f"(all < 1: {all(c < 1 for c in conts)})")
print(f"  sign of continuous diff: closed form always larger = "
      f"{all(r[7] > 0 for r in records)}")
