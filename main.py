from os import listdir

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from parse import load_datafile, normalize

tests = {}
for folder in "disk", "ring":
    tmp = []
    for fn in [fn for fn in listdir(folder) if fn.endswith(".txt")]:
        fn = f"{folder}/{fn}"
        data = load_datafile(fn)
        tmp.append(data)
    tests[folder] = normalize(tmp).mean(axis=0)


# TODO: calibrate h for each object type based on measurement
h = 200.0

xmin, xmax = 0.0, 1.401
dx = 1e-3

xfast = np.arange(8.0) * h / 1000
yfast = np.array([0.281, 0.231, 0.157, 0.087, 0.085, 0.157, 0.127, 0.095])

cs = CubicSpline(xfast, yfast, bc_type="natural")

x = np.arange(xmin, xmax, dx)

# fig, axes = plt.subplots(3, 2)

y = cs(x)
dy = cs(x, 1)
d2y = cs(x, 2)

g = 9.81


params = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False
}
mpl.rcParams.update(params)
mpl.use("pgf")

exts = ("pgf", "png")

for obj_type in "disk", "ring":

    print(obj_type)

    if obj_type == "disk":
        m = 27.7 / 1000.0
        c = 0.5
    elif obj_type == "ring":
        m = 13.1 / 1000.0
        r = 46.4
        R = 47.8
        c = 1/2 * (1 + r**2 / R**2)

    fig, ax = plt.subplots()
    ax.set_title("Høyde")
    ax.plot(x, y, xfast, yfast, "*")
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$y ~ (m)$")
    ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-height.{ext}")


    fig, ax = plt.subplots()
    v = np.sqrt((2 * g * (y[0] - y) / (1 + c)))
    ax.set_title("Hastighet")
    ax.plot(x, v)
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$v ~ (m/s)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-velocity_pos.{ext}")


    kappa = d2y / (1 + dy**2) ** (3/2)
    fig, ax = plt.subplots()
    ax.set_title("Kurvatur")
    ax.plot(x, kappa)
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$\kappa ~ (m)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-curvature.{ext}")


    sentripetal = v**2 * kappa
    fig, ax = plt.subplots()
    ax.set_title("Sentripetal akselerasjon")
    ax.plot(x, sentripetal)
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$a_\perp ~ (m/s^2)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-sentripetal_acceleration.{ext}")


    beta = np.arctan(dy)
    fig, ax = plt.subplots()
    normal = m * (g * np.cos(beta) + sentripetal)
    ax.set_title("Normal kraft")
    ax.plot(x, normal)
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$N ~ (N)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-normal_force.{ext}")

    F = (c * m * g * np.sin(beta)) / (1 + c)
    fig, ax = plt.subplots()
    ax.set_title("Kraft")
    ax.plot(x, F)
    ax.set_xlabel("$x ~ (m)$")
    ax.set_ylabel("$F ~ (N)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-force.{ext}")


    v_x = v * np.cos(beta)
    v_bx = (v_x[:-1] + v_x[1:]) / 2

    dxn = 1e-3

    # t = np.concatenate(([0], dxn * np.arange(1, len(v_bx)+1) / v_bx))

    t = np.empty(1401)
    t[0] = 0
    for n in range(1, 1401):
        t[n] = t[n-1] + dxn / v_bx[n-1]

    
    m_t, m_x, m_y = tests[obj_type]
    m_v = np.diff(m_x) / np.diff(m_t)

    fig, ax = plt.subplots()
    ax.set_title("Posisjon")
    ax.plot(t, x, label="beregnet")
    ax.plot(m_t, m_x, label="målt")
    ax.set_xlabel("$t ~ (s)$")
    ax.set_ylabel("$x ~ (m)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    ax.legend()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-position.{ext}")


    fig, ax = plt.subplots()
    ax.set_title("Hastighet")
    ax.plot(t, v_x, label="beregnet")
    ax.plot(m_t[1:], m_v, label="målt")
    ax.set_xlabel("$t ~ (s)$")
    ax.set_ylabel("$v ~ (m/s)$")
    # ax.set_ylim(0, 0.35)
    ax.grid()
    ax.legend()
    for ext in exts:
        fig.savefig(f"{ext}/{obj_type}-velocity_time.{ext}")
