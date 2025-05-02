import torch
import torch.nn.functional as F
import numpy as np
import json
from flash_stu.modules.stu import STU
from flash_stu.modules.distill_stu import DistillSTU
from flash_stu.utils.stu_utils import get_spectral_filters, compute_ar_x_preds
from flash_stu.utils.numerics import nearest_power_of_two
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def generate_lds_params(state_dim, input_dim, output_dim, delta, device=torch.device("cpu"), dtype=torch.float32):
    A = torch.randn(state_dim, dtype=dtype, device=device)
    A = A / torch.max(A.abs()) * (1 - delta) #norm 1
    
    B = torch.randn(input_dim, state_dim, dtype=dtype, device=device) / input_dim
    C = torch.randn(state_dim, output_dim, dtype=dtype, device=device) / state_dim
    h0 = torch.randn(state_dim, dtype=dtype, device=device)
    return A,B,C,h0

def generate_trajectory(A, B, C, us, h0=None):
    state_dim = A.shape[0]
    batch_size = us.shape[1] if len(us.shape) > 2 else 1
    
    if h0 is None:
        h0 = torch.zeros(state_dim, dtype=us.dtype, device=us.device)
    
    # Expand h0 to match batch size if needed
    if len(h0.shape) == 1 and batch_size > 1:
        h0 = h0.unsqueeze(0).expand(batch_size, -1)
    
    h_t, obs = h0, []
    for u in us:
        h_t = A * h_t + (u @ B)
        o_t = h_t @ C 
        obs.append(o_t)
    return torch.stack(obs, dim=0)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d_h', type=int, default=100, help='Hidden state dimension')
parser.add_argument('--d_in', type=int, default=100, help='input/output dimension')
parser.add_argument('--use_approx',type = bool, default = False, help='Whether to use approximation')
parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--delta', type=float, default=0.001, help='Delta parameter for LDS')
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_h = args.d_h
d_in =  args.d_in  #must have d_in = d_out 
d_out = args.d_in
use_approx = args.use_approx
dtype = torch.float32
use_hankel_L = False
use_zero_hidden = True
phi = torch.tensor(np.load('spectral_filters.npy')).to(device).to(dtype)
seq_len, num_eigh = 8192, 24
n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)



class Config:
    def __init__(self):
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.torch_dtype = dtype
        self.dim = d_in
        self.seq_len = seq_len
        self.use_flash_fft = False
        self.use_approx = use_approx

stu_config = Config()

model = STU(stu_config, phi, n).to(device)
lr = 1
steps = args.steps
delta = args.delta
bsz = 32

output_dir = f"synth_results_new/{delta}_{d_in}_{d_h}/"

optimizer = torch.optim.Adagrad([
    {'params': model.parameters(), 'lr': lr},
])

A,B,C,h0 = generate_lds_params(d_h, d_in, d_out, delta, device=device)
training_losses = []
import time
start_time = time.time()

for step in range(steps):
    inputs = torch.randn(bsz, seq_len, d_in).to(device)
    with torch.no_grad():
        targets = generate_trajectory(A,B,C,h0 = h0 if not use_zero_hidden else None, us = inputs)
        
    inputs = inputs.reshape(bsz, seq_len, d_in).to(device).type(dtype)
    targets = targets.reshape(bsz, seq_len, d_out).to(device).type(dtype)
    outputs = model.forward(inputs)
    loss = F.mse_loss(outputs, targets)
    training_losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Step: {step}: {loss.item()} (Time elapsed: {elapsed:.2f}s)")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.2f} seconds")

transfer_start = time.time()
distill_stu = DistillSTU(
    stu = model, 
    lds_path = "../stu_distill_2/experiments/convex_hull/fit_filters_205/250_phi_lds_float32.pt",
    state_dim = 410,
    dtype = torch.float64
)
transfer_time = time.time() - transfer_start
print(f"Transfer time: {transfer_time:.2f} seconds")

with torch.no_grad():
    eval_inputs = torch.randn(bsz,seq_len, d_in).to(device)
    eval_targets = generate_trajectory(A,B,C, h0 = h0 if not use_zero_hidden else None, us=eval_inputs)
    
    eval_inputs = eval_inputs.reshape(bsz, seq_len, d_in).to(device).type(dtype)
    eval_targets = eval_targets.reshape(bsz, seq_len, d_out).to(device).type(dtype)
    
    stu_outputs = model.forward(eval_inputs)
    eval_outputs = distill_stu.forward(eval_inputs, None)
    eval_loss = F.mse_loss(eval_outputs, eval_targets)
    print(f"Evaluation loss on batch size {bsz}: {eval_loss.item()}")
    
    # Convert to numpy for plotting
    test_target = eval_targets[0].cpu().numpy().squeeze()
    stu_output = stu_outputs[0].cpu().numpy().squeeze() 
    lds_output = eval_outputs[0].cpu().numpy().squeeze()
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot target vs STU trajectories
    ax1.plot(test_target[:,0], label='True', color='black')
    ax1.plot(stu_output[:,0], label='STU', color='blue', linestyle='dashed')
    ax1.set_title('Ground Truth vs STU Trajectories')
    ax1.set_ylabel('Output')
    ax1.grid(True)
    ax1.legend()

    # Plot target vs LDS trajectories
    ax2.plot(test_target[:,0], label='True', color='black')
    ax2.plot(lds_output[:,0], label='LDS', color='green', linestyle='dashed')
    ax2.set_title('Ground Truth vs LDS Trajectories')
    ax2.set_ylabel('Output')
    ax2.grid(True)
    ax2.legend()

    # Plot prediction gaps
    stu_gap = stu_output[:] - test_target[:]
    lds_gap = lds_output[:] - test_target[:]
    
    ax3.plot(stu_gap[:,0], label='STU Error', color='blue', linestyle='--')
    ax3.plot(lds_gap[:,0], label='LDS Error', color='green', linestyle=':')
    ax3.set_title('Prediction Errors')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error')
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.close()

    # Save all data to JSON file
    data_to_save = {
        "config": {
            "d_h": d_h,
            "d_in": d_in,
            "d_out": d_out,
            "use_approx": use_approx,
            "dtype": str(dtype),
            "use_hankel_L": use_hankel_L,
            "use_zero_hidden": use_zero_hidden,
            "seq_len": seq_len,
            "num_eigh": num_eigh,
            "n": n,
            "lr": lr,
            "steps": steps,
            "delta": delta
        },
        "training_losses": training_losses,
        "eval_losses": eval_loss.item(),
        "test_target": test_target.tolist(),
        "stu_output": stu_output.tolist(),
        "lds_output": lds_output.tolist(),
        "training_time": total_time
    }

    os.makedirs(output_dir, exist_ok=True)
    existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
    output_file = os.path.join(output_dir, f"res_{existing_files}.json")
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Results saved to {output_file}")

    
