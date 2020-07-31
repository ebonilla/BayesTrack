import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

RHO = 28.0
SIGMA = 10.0
BETA = 8.0 / 3.0


def f(state, t, sigma, beta, rho):
    x, y, z = state  # Unpack the state vector
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz  # Derivatives


def get_data_lorenz_single(state0, t0, T, delta_t):
    t = np.arange(t0, T, delta_t)
    states = odeint(f, state0, t, args=(SIGMA, BETA, RHO))
    return states, t


def main():
    state0 = [1.0, 1.0, 1.0]
    states, _ = get_data_lorenz_single(state0=state0, t0=0.0, T=40.0, delta_t=0.01)
    print(states)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()

