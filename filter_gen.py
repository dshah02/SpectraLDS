import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import math

### CONSTANTS

H = 10000
K = 24
OMP = 10 #this loosely controls the resulting dimension h
seq_len = 8192

distribution_skew = 4 #even number
assert distribution_skew %2 == 0

use_hankel_L = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## For generating the spectral_filters
def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    """Find the nearest power of 2 to x."""
    if not round_up:
        return 1 << math.floor(math.log2(x))
    else:
        return 1 << math.ceil(math.log2(x))

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    """Generate Hankel matrix for spectral filters."""
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z

# Note this is slow (usually we just run this once and save the result)
def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    dtype: np.dtype = np.float64
) -> np.ndarray:
    """Generate spectral filters using Hankel matrix eigendecomposition."""
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    return phi_k.astype(dtype)
import os

#caching for speed
phi_path = "./phi.npy"
if os.path.exists(phi_path):
    phi = np.load(phi_path)
    print(f"Loaded phi from {phi_path}")
else:
    phi = get_spectral_filters(seq_len=seq_len, K=K, use_hankel_L=use_hankel_L, dtype=np.float64)
    np.save(phi_path, phi)
    print(f"Generated and saved phi to {phi_path}")

lds_samples = (1 - np.random.rand(H) ** (distribution_skew)) * (np.random.randint(0, 2, H) * 2 - 1)
np.save("lds_samples.npy", lds_samples)

# Each row: [1, alpha_i, alpha_i^2, ..., alpha_i^{seq_len-1}]
exponents = np.arange(seq_len, dtype=np.float64)
lds_filters = lds_samples[:, None] ** exponents[None, :]

# Plot the distribution
plt.figure(figsize=(6,4))
plt.hist(lds_samples, bins=50, color='skyblue', edgecolor='k')
plt.title('Distribution of alpha')
plt.xlabel('alpha')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# OMP-based sparse approximation
A = lds_filters.T.astype(np.float64)  # (seq_len, H)
B = phi.astype(np.float64)            # (seq_len, K)

# Column-normalize A
col_norms = np.linalg.norm(A, axis=0)
col_norms[col_norms == 0] = 1.0
A_n = A / col_norms

# Scale B to A's range
alpha = np.mean(np.abs(A_n)) / (np.mean(np.abs(B)) + 1e-300)
B_n = alpha * B

# OMP: select features with shared support across K targets
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=OMP, fit_intercept=False, precompute='auto')
omp.fit(A_n, B_n)
Wn = omp.coef_.T  # (H, K)

# Undo scaling
W_sparse = (Wn / col_norms[:, None]) / alphawa

# De-bias: refit LS on selected support
support = np.flatnonzero(np.linalg.norm(W_sparse, axis=1) > 0)
X, *_ = np.linalg.lstsq(A[:, support], B, rcond=None)
W_debias = np.zeros_like(W_sparse)
W_debias[support, :] = X

# Evaluate MSE
B_reconstructed = A @ W_debias
mse_np = np.mean((B_reconstructed - B)**2)

h = len(support)
print(f"Hidden Dimension {h}")

# Simpler plot: show fewer filters and clearer display
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

num_filters_to_plot = min(6, K)  # only show up to 6 filters for clarity
colors = ['#FF3D3D', '#4B71EA']

plt.figure(figsize=(14, 8), dpi=100)
for i in range(num_filters_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.plot(B[:200, i], color=colors[0], linewidth=2, label='Original')
    plt.plot(B_reconstructed[:200, i], '--', color=colors[1], linewidth=2, label='Approximated')
    plt.title(f'Filter {i+1}', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    if i == 0:
        plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# Per-filter MSE distribution
filter_mses = [np.mean((B_reconstructed[:, i] - B[:, i])**2) for i in range(K)]


print(f"Mean MSE: {np.mean(filter_mses):.2e}, Median: {np.median(filter_mses):.2e}")
print(f"Min MSE: {np.min(filter_mses):.2e}, Max MSE: {np.max(filter_mses):.2e}")

#Now load this into an LDS

from flash_stu.utils.lds import LDS
# Load an LDS with input dim 1 and output dim 24
lds_model = LDS(2 * h, input_dim=1, output_dim=2*K, dtype =torch.float64, device = torch.device('cpu'))

A_tensor = torch.tensor(lds_samples[support], dtype=torch.float64).flatten()
lds_model.A.data = torch.cat([A_tensor.to(torch.float64), -1 * A_tensor.to(torch.float64)], dim = -1)

lds_model.B.data = torch.ones_like(lds_model.B.data)

combined = torch.zeros((h*2, K*2)).to(torch.float64)
combined[:h, :K] = torch.tensor(W_debias[support, :K], dtype=torch.float64)
combined[h:, K:] = torch.tensor(W_debias[support, :K], dtype=torch.float64)

lds_model.C.data = combined
lds_model.h0.data = torch.zeros(lds_model.A.shape[0]).to(torch.float64)


checkpoint = {
    'state_dim': lds_model.A.shape[0],
    'input_dim': 1,
    'output_dim': 2 *h,
    'kx': 0,  # as set in the model
    'dtype': 'torch.float64',  # as used in the model
    'model_state_dict': lds_model.state_dict()
}

# Save the checkpoint
torch.save(checkpoint, f'./{len(support)}_phi_lds.pt')
