import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import animation


def main():
    # physical constants (set ħ = 1, m = 1 for simplicity)
    hbar = 1.0
    m = 1.0

    # spatial grid
    L = 200.0
    N = 2000
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # initial gaussian wavepacket (sigma is standard deviation)
    x0 = -30.0           # initial center
    k0 = 3.0             # initial momentum
    sigma = 5.0          # width (standard deviation)
    psi0 = np.exp(-(x - x0)**2/(2.0 * sigma**2)) * np.exp(1j * k0 * x)
    # normalize
    psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

    # potential: example a barrier
    V0 = 1.0
    V = np.zeros_like(x)
    V[np.logical_and(x > -5, x < 5)] = V0

    # Crank-Nicolson setup
    dt = 0.05
    steps = 800

    # discrete laplacian (sparse, second-order central)
    diag = np.full(N, -2.0)
    off = np.ones(N - 1)
    lap = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csc') / dx**2

    # Hamiltonian H = - (ħ^2 / 2m) d2/dx2 + V
    H = - (hbar**2 / (2.0 * m)) * lap + sp.diags(V, 0, format='csc')

    I = sp.identity(N, format='csc')
    A = (I + 1j * dt * H / (2.0 * hbar)).tocsc()
    B = (I - 1j * dt * H / (2.0 * hbar)).tocsc()

    # factorized solver for A (fast solves for many timesteps)
    lu = spla.factorized(A)

    psi = psi0.copy()

    # prepare plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    line_psi, = ax.plot(x, np.abs(psi)**2, lw=1.5, label='|ψ|^2')

    V_max = V.max() if V.max() > 0 else 1.0
    scaled_V = V / V_max * np.max(np.abs(psi0)**2) * 0.9
    line_V, = ax.plot(x, scaled_V, lw=1, label='V (scaled)')

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, np.max(np.abs(psi0)**2) * 1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability density')
    ax.legend()

    def update(i):
        nonlocal psi
        b = B.dot(psi)
        psi = lu(b)
        if i % 1 == 0:
            line_psi.set_ydata(np.abs(psi)**2)
        return line_psi,

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=20)
    plt.show()


if __name__ == '__main__':
    main()
