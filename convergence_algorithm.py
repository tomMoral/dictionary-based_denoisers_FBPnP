import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Plot convergence depending on the number of layers/iterations.')

parser.add_argument('--algorithm', type=str, help='The algorithm to be used [analysis | synthesis]', default='analysis')

parser.add_argument('--dimension_observation', type=int, help='Dimension of the observations', default=20)

parser.add_argument('--dimension_signal', type=int, help='Dimension of the signal', default=50)

parser.add_argument('--dimension_codes', type=int, help='Dimension of the sparse codes', default=100)

parser.add_argument('--seed', type=int, help='Random seed', default=123456)


args = parser.parse_args()
RNG = np.random.default_rng(args.seed)

m = args.dimension_observation
n = args.dimension_signal
L = args.dimension_codes

A = RNG.normal(size=(m, n))
D = RNG.normal(size=(n, L))


def st(u, lmbd):
    return np.sign(u) * np.maximum(0, np.abs(u) - lmbd)


def relu(u):
    return np.maximum(0, u)


def prox_gstar(u, lmbd, sigma):
    return u - sigma * (
        relu(u / sigma - lmbd / sigma)
        - relu(- u / sigma - lmbd / sigma)
    )


def iterate_analysis(x, z, y, lmbd, tau, sigma, n_prox):
    x_tilde = x - tau * A.T @ (A @ x - y)
    z_new = z.copy()
    for i in range(n_prox):
        z_new = z_new - sigma * D.T @ (D @ z_new - x_tilde)
        z_new = prox_gstar(z_new, lmbd * tau, sigma)
    x_new = x_tilde - D @ z_new
    return x_new, z_new


def iterate_synthesis(x, z, y, lmbd, tau, sigma, n_prox):
    x_tilde = x - tau * A.T @ (A @ x - y)
    z_new = z.copy()
    for i in range(n_prox):
        z_new = st(z_new - sigma * D.T @ (D @ z_new - x_tilde), sigma * tau * lmbd)
    x_new = D @ z_new
    return x_new, z_new


def lasso_synthesis(x, z, y, lmbd, tau, sigma, n_prox):
    z_new = st(z - tau * sigma * D.T @ A.T @ (A @ D @ z - y), tau * sigma * lmbd)
    return D @ z_new, z_new


def algo(x0, z0, y, lmbd, tau, sigma, n_prox, n_iter, algorithm,
         x_truth=None, z_truth=None):
    x_old, z_old = x0.copy(), z0.copy()
    conv_x = []
    conv_z = []
    times = []
    if x_truth is not None and z_truth is not None:
        conv_x.append(np.linalg.norm(x0 - x_truth))
        conv_z.append(np.linalg.norm(z0 - z_truth))
        times.append(0)
    start = time.time()
    for j in tqdm(range(n_iter)):
        x, z = algorithm(x_old, z_old, y, lmbd, tau, sigma, n_prox)
        times.append(time.time() - start)
        if x_truth is not None and z_truth is not None:
            conv_x.append(np.linalg.norm(x - x_truth))
            conv_z.append(np.linalg.norm(z - z_truth))
        x_old, z_old = x.copy(), z.copy()
    return x, z, conv_x, conv_z, times


def power_iteration(M, n_iter=50):
    u = RNG.random(M.shape[1])
    u /= np.linalg.norm(u)
    for i in range(n_iter):
        u = M.T @ M @ u
        norm = np.linalg.norm(u)
        u /= norm
    return norm


algorithm = iterate_analysis if args.algorithm == "analysis" else iterate_synthesis

tau = 1 / power_iteration(A)
sigma = 1 / power_iteration(D)

x0 = np.zeros(n)
z0 = np.zeros(L)

y = A @ RNG.random(n)

lmbd_max = np.abs(D.T @ A.T @ y).max()

n_iter = 10000

if args.algorithm == "analysis":
    lmbd = 0.01 * lmbd_max
else:
    lmbd = 0.1 * lmbd_max

print("Computing reference solution...")
x_truth, z_truth, _, _, _ = algo(
    x0, z0, y, lmbd, tau, sigma,
    n_prox=1000, n_iter=n_iter, algorithm=algorithm
)

print(np.sum(np.abs(x_truth)))

fig, axs = plt.subplots(1, 4, figsize=(10, 2))

range_n_prox = [1, 20, 50, 100]
handles = []
labels = []
print(f"Computing solution for n_prox in {range_n_prox}")
for n_prox in range_n_prox:
    if n_prox == 1:
        n_iter_current = int(1e5)
    else:
        n_iter_current = int(1e4)
    x, z, conv_x, conv_z, times = algo(
        x0, z0, y, lmbd, tau, sigma, n_prox=n_prox, n_iter=n_iter_current,
        algorithm=algorithm, x_truth=x_truth, z_truth=z_truth
    )

    line1, = axs[0].plot(conv_x, alpha=0.5)
    axs[1].plot(conv_z, alpha=0.5)
    axs[2].plot(times, conv_x, alpha=0.5)
    axs[3].plot(times, conv_z, alpha=0.5)

    handles.append(line1)
    labels.append(rf"L={n_prox}")

if args.algorithm == "analysis":
    dual_variable = "u"
else:
    dual_variable = "z"

axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].set_xticks([1, 1e2, 1e4])
axs[0].set_xlabel(r"Iterations $k$")
axs[0].set_ylabel(r"$||x_k - x^*||$")
axs[0].grid(True)

axs[1].set_yscale("log")
axs[1].set_xscale("log")
axs[1].set_xticks([1, 1e2, 1e4])
axs[1].set_xlabel(r"Iterations $k$")
axs[1].set_ylabel(rf"$||{dual_variable}_k - {dual_variable}^*||$")
axs[1].grid(True)

axs[2].set_yscale("log")
axs[2].set_xscale("log")
axs[2].set_xlabel("Time (s)")
axs[2].set_xticks([1e-4, 1e-2, 1])
axs[2].set_ylabel(r"$||x_k - x^*||$")
axs[2].grid(True)

axs[3].set_yscale("log")
axs[3].set_xscale("log")
axs[3].set_xlabel("Time (s)")
axs[3].set_xticks([1e-4, 1e-2, 1])
axs[3].set_ylabel(rf"$||{dual_variable}_k - {dual_variable}^*||$")
axs[3].grid(True)

fig.legend(handles, labels, loc='upper center', ncol=len(labels))

plt.subplots_adjust(wspace=1.0, top=0.8)  # Increase top margin

plt.savefig(f"convergence_{args.algorithm}.pdf", bbox_inches='tight', dpi=300)
