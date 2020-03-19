from os import listdir

import numpy as np

from random import choice
from scipy.signal import correlate, find_peaks
from scipy.interpolate import interp1d


def load_datafile(fn):
    with open(fn, "rb") as file:
        lines = file.read().replace(b",", b".").split(b"\n")[2:-1]

    a = np.empty((len(lines), 3))
    for row, line in enumerate(lines):
        a[row] = line.split(b"\t")

    return a


def find_shift(a, b, col=0):
    idx = col + 1

    a_peaks, _ = find_peaks(-a[idx])
    b_peaks, _ = find_peaks(-b[idx])

    if not (a_peaks.size and b_peaks.size):
        return None, None, None

    a_peak = a_peaks[a[idx][a_peaks].argmin()]
    b_peak = b_peaks[b[idx][b_peaks].argmin()]

    xshift = b[0][b_peak] - a[0][a_peak]
    yshift = b[idx][b_peak] - a[idx][a_peak]

    return xshift, yshift, a_peak, b_peak


def normalize(tests, fake_missing=True):
    tests = [test.T for test in tests]

    if fake_missing:
        if len(tests) < 10:
            n_missing = 10 - len(tests)
            for _ in range(n_missing):
                fake = choice(tests).copy()
                fake += np.random.randn(*fake.shape) * 0.0005
                tests.append(fake)

    t_max = min(a[0][-1] for a in tests)
    t = np.linspace(0, t_max, 200)

    ref_idx = 0

    ref = tests[ref_idx]
    new_tests = []

    for i, a in enumerate(tests):
        xshift, yshift, ap, refp = find_shift(a, ref, col=1)

        if xshift != None:
            a[0] += xshift
            a[1] += yshift

        f = interp1d(a[0], a[1], fill_value="extrapolate", kind="linear")
        g = interp1d(a[0], a[2], fill_value="extrapolate", kind="linear")

        b = np.vstack([t, f(t), g(t)])
        new_tests.append(b)

    return np.array(new_tests)
