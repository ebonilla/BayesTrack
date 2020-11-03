import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

RHO = 28.0
SIGMA = 10.0
BETA = 8.0 / 3.0


def f_single(state, t, sigma, beta, rho):
    x, y, z = state  # Unpack the state vector
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz  # Derivatives


def f_coupled(state, t, sigma, beta, rho, c_e, c, c_z, tau, S, k1, k2):
    """

    :param state: 9D state (list)
    :param t:
    :param sigma: Lorenz param
    :param beta: Lorenz param
    :param rho: Lorenz param
    :param c_e: Coupling coefficient (extratropical to tropical)
    :param c: Coupling coefficient (tropical to ocean)
    :param c_z: Coupling coefficient (tropical to ocean)
    :param tau: Scaling parameter
    :param S: Scaling parameter
    :param k1: Uncentering parameter
    :param k2: Uncentering parameter
    :return:
    """
    x_e, y_e, z_e, x_t, y_t, z_t, X, Y, Z = state  # Unpack state vector

    # Fast extratropical atmosphere
    dx_e = sigma * (y_e - x_e) - c_e * (S * x_t + k1)
    dy_e = rho * x_e - y_e - x_e * z_e + c_e * (S * y_t + k1)
    dz_e = x_e * y_e - beta * z_e

    # Fast tropical atmosphere
    dx_t = sigma * (y_t - x_t) - c * (S * X + k2) - c_e * (S * x_e + k1)
    dy_t = rho * x_t - y_t - x_t * z_t + c * (S * Y + k2) + c_e * (S * y_e + k1)
    dz_t = x_t * y_t - beta * z_t + c_z * Z

    # Slow tropical ocean
    dX = tau * sigma * (Y - X) - c * (x_t + k2)
    dY = tau * rho * X - tau * Y - tau * S * X * Z + c * (y_t + k2)
    dZ = tau * S * X * Y - tau * beta * Z - c_z * z_t

    return dx_e, dy_e, dz_e, dx_t, dy_t, dz_t, dX, dY, dZ


def get_data_lorenz_single(state0, t0, T, delta_t):
    t = np.arange(t0, T, delta_t)
    states = odeint(f_single, state0, t, args=(SIGMA, BETA, RHO))
    return states, t


def get_data_lorenz_coupled(state0, t0, T, delta_t, c_e=0.08, c=1, c_z=1, tau=0.1, S=1, k1=10, k2=-11):
    t = np.arange(t0, T, delta_t)
    states = odeint(f_coupled, state0, t, args=(SIGMA, BETA, RHO, c_e, c, c_z, tau, S, k1, k2))
    return states, t


def main_single():
    state0 = [1.0, 1.0, 1.0]
    states, _ = get_data_lorenz_single(state0=state0, t0=0.0, T=1.0, delta_t=0.01)
    print(states.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()


def unpack_state_coupled(states):
    x_e = states[:, 0]
    y_e = states[:, 1]
    z_e = states[:, 2]

    x_t = states[:, 3]
    y_t = states[:, 4]
    z_t = states[:, 5]

    X = states[:, 6]
    Y = states[:, 7]
    Z = states[:, 8]

    return x_e, y_e, z_e, x_t, y_t, z_t, X, Y, Z

def main_coupled():
    delta_t = 0.01
    state0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    states, _ = get_data_lorenz_coupled(state0=state0, t0=0.0, T=100.0, delta_t=delta_t)
    print(states.shape)

    x_e, y_e, z_e, x_t, y_t, z_t, X, Y, Z = unpack_state_coupled(states)

    # 3d systems
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x_e, y_e, z_e)
    plt.draw()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x_t, y_t, z_t)
    plt.draw()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X, Y, Z)
    plt.draw()
    plt.show()

    # x trajectories
    T = 40
    t = np.arange(start=0, stop=T, step=delta_t)
    L_t = t.size

    plt.plot(t, x_e[:L_t])
    plt.show()

    plt.plot(t, x_t[:L_t])
    plt.show()

    plt.plot(t, X[:L_t])
    plt.show()







if __name__ == "__main__":
    # execute only if run as a script
    # main_single()
    main_coupled()






