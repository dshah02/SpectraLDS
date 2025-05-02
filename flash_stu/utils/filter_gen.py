import torch
import torch.nn.functional as F
import numpy as np
import json
from flash_stu.modules.stu import STU
from flash_stu.utils.lds import LDS
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.utils.filter_gen_utils import load_lds_stu_pairs, compute_mse_for_pairs, from_stored
import yaml
import tqdm
import copy
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--sc', type=int, default=50, help='Number of shuffled models to use')
parser.add_argument('--gc', type=int, default=100, help='Number of models for greedy selection')
parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for filtering')
parser.add_argument('--S_threshold', type=float, default=1e-5, help='S matrix threshold')
parser.add_argument('--shuffle_steps', type=int, default=500000, help='Number of shuffle optimization steps')
parser.add_argument('--gradient_descent_steps', type=int, default=50000, help='Number of gradient descent steps')

args = parser.parse_args()

shuffled_count = args.sc
greedy_count = args.gc  
threshold = args.threshold
S_threshold = args.S_threshold
shuffle_steps = args.shuffle_steps
gradient_descent_steps = args.gradient_descent_steps

print("___________________________________________")
print(f"Shuffled count: {shuffled_count}")
print(f"Greedy count: {greedy_count}")
print(f"Threshold: {threshold}")
print(f"S matrix threshold: {S_threshold}")
print(f"Shuffle steps: {shuffle_steps}")
print(f"Gradient descent steps: {gradient_descent_steps}")


dtype = torch.float32
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
seq_len, num_eigh = 8192, 24

# phi= get_spectral_filters(seq_len = seq_len, K = num_eigh,  use_hankel_L= False, device  = device,  dtype = torch.float32)
# print('filters generated')
#(below is same up to similarity)
phi = torch.tensor(np.load('../stu_distill_2/experiments/convex_hull/spectral_filters.npy')).to(device).to(dtype) #download at _____
n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

class Config:
    def __init__(self):
        self.num_eigh = num_eigh
        self.use_hankel_L = False
        self.torch_dtype = dtype
        self.dim = 1
        self.seq_len = seq_len
        self.k_u = 0
        self.use_flash_fft = False
        self.use_approx = False

stu_config = Config()
lds_stu_pairs = load_lds_stu_pairs(directory = '../../../scratch/gpfs/ds6237/pairs')

# Convert state dicts into actual model pairs
model_pairs = []
for lds_state, stu_state in lds_stu_pairs:
    # Skip if used flash_fft
    if 'flash_fft.f_16_fft' in stu_state:
        continue
        
    lds = LDS(1, 1, 1, device = device).to(device)
    stu = STU(stu_config, phi, n).to(device)
    
    lds.load_state_dict(lds_state, strict = False)
    lds.h0.data = lds_state['h0'].to(device)
    stu.load_state_dict(stu_state)
    
    model_pairs.append((lds, stu))

lds_stu_pairs = model_pairs




# print("Loaded models", len(lds_stu_pairs))
#precomputed for time
mse_results = compute_mse_for_pairs(lds_stu_pairs, 8192)


# # Perform unnecessary GPU operations cause otherwise cluster will kick me off :(
# print("Performing unnecessary GPU operations...")

# # Generate large random tensors
# random_tensor_1 = torch.randn(10000, 10000, device=device).cuda()
# random_tensor_2 = torch.randn(10000, 10000, device=device).cuda()

# # Matrix multiplications
# for _ in range(10):
#     result = torch.matmul(random_tensor_1, random_tensor_2)
#     result = torch.matmul(result, random_tensor_1.T)
    
# print("Finished unnecessary GPU operations")

#filters pairs with high MSE Loss
filtered_pairs = []
for result in mse_results:
    if result['mse'] < threshold:
        filtered_pairs.append((result['lds'], result['stu']))

phi_n = phi.data.cpu().numpy() #baseline

def gen_lds_impulse(lds, seq_len=seq_len):
    a = torch.ones(seq_len)
    a[1:] = lds.A.item()
    a_powers = torch.cumprod(a, dim=0) 
    lds_impulse = lds.C.item() * a_powers * lds.B.item()
    return lds_impulse.cpu()

def gen_stu_impulse(stu, seq_len = seq_len):
    alt_sign = lambda x: x * np.array([1, -1] * (seq_len//2))
    pos_coef = stu.M_phi_plus.data.cpu().numpy()[:, 0,0]
    neg_coef = stu.M_phi_minus.data.cpu().numpy()[:,0,0]
    impulse = np.sum(phi_n*pos_coef, axis = -1) + alt_sign(np.sum(phi_n*neg_coef, axis = -1))
    return impulse

lds_params = [lds for lds, _ in filtered_pairs]
stu_params = [stu for _, stu in filtered_pairs]


with torch.no_grad():
    lds_impulses = np.array([gen_lds_impulse(lds).detach() for lds in tqdm.tqdm(lds_params, desc="Generating LDS impulses")])
    stu_impulses = np.array([gen_stu_impulse(stu) for stu in tqdm.tqdm(stu_params, desc="Generating STU impulses")])

lds_impulses= torch.tensor(lds_impulses, dtype=torch.float)

#Account for alternating phi
alternating_signs = np.array([1, -1] * (seq_len//2))
phi_n_alternating = phi_n * alternating_signs[:, np.newaxis]
phi_n_combined = np.concatenate([phi_n, phi_n_alternating], axis=1)  # Shape: (8192, 48)

# For each STU, combine M_phi_plus and M_phi_minus
combined_weights = []
for stu in stu_params:
    M_phi_plus = stu.M_phi_plus.detach().cpu().numpy()[:,0,0]
    M_phi_minus = stu.M_phi_minus.detach().cpu().numpy()[:,0,0]
    
    # Concatenate the weights
    combined = np.concatenate([M_phi_plus, M_phi_minus], axis=0)
    combined_weights.append(combined)

# Stack all combined weights into a single array
combined_weights = np.stack(combined_weights)
samples_cnt = lds_impulses.shape[0]

best_mse = float('inf')
best_indices = None

for _ in tqdm.tqdm(range(shuffle_steps)):
    sampled_indices = np.random.choice(samples_cnt, size=shuffled_count, replace=False)
    
    # Subsample from lds_impulses and combined_weights using these indices
    lds_impulses_sampled = lds_impulses[sampled_indices]
    combined_weights_sampled = combined_weights[sampled_indices]

    CWSP = np.linalg.pinv(combined_weights_sampled.T)
    phi_n_approx = np.matmul(lds_impulses_sampled.T.cpu(), CWSP)
    mse = F.mse_loss(torch.tensor(phi_n_combined[:, :24]), torch.tensor(phi_n_approx[:, :24])) #only focusing on first 24 filters
    
    if mse < best_mse:
        best_mse = mse
        best_indices = sampled_indices


indices = copy.deepcopy(best_indices)
best_mse = float('inf')

# Calculate initial MSE
lds_impulses_sampled = lds_impulses[indices]
CWS = combined_weights[indices]
CWTP = np.linalg.pinv(CWS.T)
phi_n_approx = np.matmul(lds_impulses_sampled.T, CWTP)
best_mse = F.mse_loss(torch.tensor(phi_n_combined[:, :24]), torch.tensor(phi_n_approx[:, :24]))

print(f"Initial MSE with {len(indices)} coordinates: {best_mse:.2e}")
init_size = len(indices)
# Greedily add coordinates that minimize MSE
remaining_indices = list(set(range(samples_cnt)) - set(indices))


for i in tqdm.tqdm(range(greedy_count)):
    current_best_mse = float('inf')
    best_new_idx = None
    
    # Try adding each remaining coordinate and keep the best one
    for j, idx in enumerate(np.random.choice(remaining_indices, size=len(remaining_indices), replace=False)):
        test_indices = np.append(indices, idx)
        
        # Calculate MSE with this additional coordinate
        lds_impulses_test = lds_impulses[test_indices]
        CWT = combined_weights[test_indices]
        CWTP = np.linalg.pinv(CWT.T)
        phi_n_test = np.matmul(lds_impulses_test.T, CWTP)
        test_mse = F.mse_loss(torch.tensor(phi_n_combined[:, :24]), torch.tensor(phi_n_test[:, :24]))
        
        if test_mse < current_best_mse:
            current_best_mse = test_mse
            best_new_idx = idx
    
    if best_new_idx is not None:
        indices = np.append(indices, best_new_idx)
        remaining_indices.remove(best_new_idx)
        best_mse = current_best_mse
        print(f'mse data: {init_size} {len(indices)} {best_mse}')
        
lds_impulses_final = lds_impulses[indices]
CWF = combined_weights[indices]
CWFP = np.linalg.pinv(CWF.T)
phi_n_approx = np.matmul(lds_impulses_final.T.cpu(), CWFP.cpu())

print("Starting gradient descent optimization...")

phi_n_combined_tensor = torch.tensor(phi_n_combined, dtype=torch.float32)
CWFP = torch.tensor(CWFP, dtype=torch.float32, requires_grad=True)

learning_rate = 1e-4
optimizer = torch.optim.Adam([CWFP], lr=learning_rate)
print_interval = 5000

losses = []
for step in range(gradient_descent_steps + 1):
    phi_n_approx_tensor = torch.matmul(lds_impulses_final.T, CWFP)

    loss = F.mse_loss(phi_n_approx_tensor[:, :24], phi_n_combined_tensor[:, :24])
    losses.append(loss.cpu().item())
    
    if step % print_interval == 0:
        print(f"Step {step}/{gradient_descent_steps}, Loss: {loss.item():.2e}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('FINAL LOSS:', losses[-1])

# Extract A, B, and C matrices from selected LDS models
A_matrices = []
B_matrices = []
C_matrices = []

for idx in indices:
    lds = lds_params[idx]
    A_matrices.append(lds.A.detach().clone())
    B_matrices.append(lds.B.detach().clone())
    C_matrices.append(lds.C.detach().clone())

stacked_A = torch.stack(A_matrices)
stacked_B = torch.stack(B_matrices)
stacked_C = torch.stack(C_matrices)

print(f"Stacked A shape: {stacked_A.shape}")
print(f"Stacked B shape: {stacked_B.shape}")
print(f"Stacked C shape: {stacked_C.shape}")

BC = (stacked_B * stacked_C).reshape(-1)
A = stacked_A.reshape(-1)

A_tensor = torch.tensor(A, dtype=torch.float64)
BC_tensor = torch.tensor(BC, dtype=torch.float64)
combined_weights = torch.tensor(combined_weights.cpu(), dtype=torch.float64)

lds_model = LDS(A_tensor.shape[0], input_dim=1, output_dim=48, dtype =torch.float64)
lds_model.A.data = A_tensor.to(torch.float64)
lds_model.A.data = torch.cat([lds_model.A.data, -1 * lds_model.A.data], dim = -1)

lds_model.B.data  = torch.cat([BC_tensor.unsqueeze(dim = 0).to(torch.float64), BC_tensor.unsqueeze(dim = 0).to(torch.float64)], dim = -1)
combined = torch.zeros((len(A_tensor)*2, 48)).to(torch.float64)
combined[:len(A_tensor), :24] = CWFP[:, :24].cpu().to(torch.float64)
combined[len(A_tensor):, 24:] = CWFP[:, :24].cpu().to(torch.float64)
lds_model.C.data = combined
lds_model.h0.data = torch.zeros(lds_model.A.shape[0]).to(torch.float64)

checkpoint = {
    'state_dim': lds_model.A.shape[0],
    'input_dim': 1,
    'output_dim': 48,
    'dtype': 'torch.float64',  # as used in the model
    'model_state_dict': lds_model.state_dict()
}

import os
# Generate a random run ID
run_id = hex(hash(str(torch.rand(1).item())))[-6:]
save_dir = f'checkpoints_{run_id}'
os.makedirs(save_dir, exist_ok=True)

torch.save(checkpoint, os.path.join(save_dir, f'{lds_model.A.shape[0]}_phi_lds.pt'))
stats = {
    'state_dim': lds_model.A.shape[0],
    'input_dim': lds_model.B.shape[1], 
    'output_dim': lds_model.C.shape[1],
    'A_norm': float(torch.norm(lds_model.A.data).item()),
    'B_norm': float(torch.norm(lds_model.B.data).item()),
    'C_norm': float(torch.norm(lds_model.C.data).item()),
    'error': losses[-1],
    'checkpoint_path': os.path.join(save_dir, f'{lds_model.A.shape[0]}_phi_lds.pt'),
    'run_id': run_id
}

import json
with open(os.path.join(save_dir, 'model_stats.json'), 'w') as f:
    json.dump(stats, f, indent=4)


