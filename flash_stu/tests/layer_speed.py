import torch
import time
import numpy as np
import gc 
import os
import json
import argparse
import sys

from flash_stu.modules.stu import STU
from flash_stu.modules.distill_stu import DistillSTU
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.modules.cache import Cache


class TestConfig:
    def __init__(self, L=1048576):
        self.dim = 128  # 896
        self.num_eigh = 24
        self.seq_len = L
        self.use_hankel_L = False
        self.use_approx = True
        self.torch_dtype = torch.bfloat16
        self.use_flash_fft = True
        self.state_dim = 100


def create_stu_model(config, model_type="stu", future_fill=False, state_dim=100):
    # Create spectral filters
    filters = torch.randn(config.seq_len, config.num_eigh).cuda().to(config.torch_dtype)
    
    if model_type == "stu":
        # Create STU with approximation
        model = STU(config, filters, future_fill=future_fill).cuda()
        model.setup_phi()
        if future_fill:
            model.setup_ff()
        model.cache = Cache(
            batch_size=1,
            max_seq_len=config.seq_len,
            dim=config.dim,
            dtype=config.torch_dtype,
        ).cuda()
    elif model_type == "stu_no_approx":
        # Create STU without approximation
        config_no_approx = TestConfig(config.seq_len)
        config_no_approx.use_approx = False
        # config_no_approx.seq_len =  config_no_approx.seq_len
        model = STU(config_no_approx, filters, future_fill=False).cuda()
        model.cache = Cache(
            batch_size=1,
            max_seq_len=config.seq_len,
            dim=config.dim,
            dtype=config.torch_dtype,
        ).cuda()
    elif model_type == "distill_stu":
        # Create base STU first
        stu = STU(config, filters, future_fill=False).cuda()
        stu.setup_phi()
        stu.cache = Cache(
            batch_size=1,
            max_seq_len=config.seq_len,
            dim=config.dim,
            dtype=config.torch_dtype,
        ).cuda()
        
        # Create DistillSTU with the specified state dimension
        model = DistillSTU(stu, state_dim=state_dim, dtype=config.torch_dtype)
        model.lds.reset_state(config.dim)
        model.lds.cache_enabled = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


def run_benchmark(model, length=32, dtype=torch.bfloat16):
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    x_t = torch.randn(1, 1, model.d_in, device="cuda", dtype=dtype)
    # with torch.no_grad():
    for t in range(length):
        _ = model(x_t, torch.tensor([t], device="cuda"))
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    
    # Clear cache between runs
    torch.cuda.empty_cache()
        
    return total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark STU models")
    parser.add_argument("--model", type=str, choices=["stu", "stu_ff", "stu_no_approx", "distill_stu", "distill_stu_800"], 
                        required=True, help="Model type to benchmark")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length to test")
    parser.add_argument("--output_dir", type=str, default="speed_test_results_2", help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup config
    config = TestConfig(1048576)  # Use large max sequence length for filters
    
    print(f"Running benchmark for model: {args.model}, sequence length: {args.seq_len}")
    
    # Create the appropriate model
    if args.model == "stu":
        model = create_stu_model(config, model_type="stu", future_fill=False)
    elif args.model == "stu_ff":
        model = create_stu_model(config, model_type="stu", future_fill=True)
    elif args.model == "stu_no_approx":
        # print("Warning: stu_no_approx not implemented yet")
        
        # sys.exit(1)
        model = create_stu_model(config, model_type="stu_no_approx")
    elif args.model == "distill_stu":
        model = create_stu_model(config, model_type="distill_stu", state_dim=100)
    elif args.model == "distill_stu_800":
        model = create_stu_model(config, model_type="distill_stu", state_dim=800)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Run benchmark
    total_time = run_benchmark(model, length=args.seq_len)
    
    # Clean up model to avoid memory issues
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    # Convert to milliseconds
    mean_ms = total_time * 1000
    
    print(f"Results: {mean_ms:.2f}")
    
    # Save results to JSON
    results = {
        "sequence_length": args.seq_len,
        "model_type": args.model,
        "time_ms": mean_ms,
    }
    
    # Generate random 6-character hex ID
    import random
    random.seed()
    run_id = ''.join(random.choices('0123456789abcdef', k=6))
    output_file = os.path.join(args.output_dir, f"{args.model}_L{args.seq_len}_{run_id}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()