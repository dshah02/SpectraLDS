import torch
from pathlib import Path
import tqdm
import json
import torch.nn.functional as F
from flash_stu.modules.stu import STU
from flash_stu.utils.lds import LDS
import numpy
import os


def load_lds_stu_pairs(directory="."):
    lds_dir = Path(directory) / "lds_state"
    stu_dir = Path(directory) / "stu_state"
    
    if not lds_dir.exists() or not stu_dir.exists():
        raise ValueError(f"One or both directories do not exist: {lds_dir}, {stu_dir}")
    
    models_dict = {}

    for file in lds_dir.glob("*.pth"):
        try:
            index = int(file.stem.split('_')[1])
            
            lds_state = torch.load(file, map_location=torch.device('cpu'))
            
            if index not in models_dict:
                models_dict[index] = {'lds': None, 'stu': None}
            
            models_dict[index]['lds'] = lds_state
            
        except Exception as e:
            print(f"Error loading LDS state dict {file}: {e}")
    
    for file in stu_dir.glob("*.pth"):
        try:
            index = int(file.stem.split('_')[1])
            
            stu_state = torch.load(file, map_location=torch.device('cpu'))
            
            if index not in models_dict:
                models_dict[index] = {'lds': None, 'stu': None}
            
            models_dict[index]['stu'] = stu_state
            
        except Exception as e:
            print(f"Error loading STU state dict {file}: {e}")
    
    pairs = []
    for index, models in sorted(models_dict.items()):
        if models['lds'] is not None and models['stu'] is not None:
            pairs.append((models['lds'], models['stu']))
    
    print(f"Loaded {len(pairs)} complete LDS-STU state dictionary pairs")
    return pairs

def compute_mse_for_pairs(pairs, seq_len=100, batch_size=1, input_dim=1, device = torch.device('cpu')):
    pairs_len = len(pairs)
    cache_file = f"_mse_loss_cache_{pairs_len}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
            results = []
            for pair_index, mse in cached_results.items():
                results.append({
                    'pair_index': int(pair_index),
                    'mse': mse,
                    'lds': pairs[int(pair_index)][0],
                    'stu': pairs[int(pair_index)][1]
                })
            return results
            
    results = []
    
    for i, (lds, stu) in enumerate(pairs):
        gaussian_input = torch.randn(batch_size, seq_len, input_dim, device = device)
        
        lds.eval()
        stu.eval()
        
        with torch.no_grad():
            lds_output = lds(gaussian_input)
            stu_output = stu(gaussian_input)
            
            mse = F.mse_loss(lds_output, stu_output).item()
            
        results.append({
            'pair_index': i,
            'mse': mse,
            'lds': lds,
            'stu': stu
        })
        
    cache_dict = {str(result['pair_index']): result['mse'] for result in results}
    with open(cache_file, 'w') as f:
        json.dump(cache_dict, f)
    return results

def from_stored(lds_stu_pairs, file_name = 'lds_stu_mse.json'):
    with open(file_name, 'r') as f:
        mse_dict = json.load(f)

    mse_results = []
    for pair_index, mse in mse_dict.items():
        mse_results.append({
            'pair_index': int(pair_index),
            'mse': mse,
            'lds': lds_stu_pairs[int(pair_index)][0],
            'stu': lds_stu_pairs[int(pair_index)][1]
        })
    return mse_results
