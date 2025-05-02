import sys
import os
import tqdm
import argparse
import json
from time import time
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
import yaml 
from safetensors import safe_open
import numpy as np
import random
from collections import defaultdict
from tiktoken.load import load_tiktoken_bpe
from model import FlashSTU, FlashSTUConfig
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.utils.random_utils import get_logger, save_yaml_config
import math
from typing import Union
import gc

logger = get_logger(__name__)
bpe_path = "./models/o200k_base.tiktoken"

def set_initial_random_seed(random_seed: int):
    if random_seed > 0:
        seed_offset = 0 #single gpu for now
        random_seed += seed_offset
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

def apply_compile(model: nn.Module) -> None:
    """
    Apply torch.compile to each layer. This makes compilation efficient
    due to repeated structure. Alternatively, one can just compile the whole model.
    """
    logger.info(f"Compiling each {model.__class__.__name__} layer with torch.compile...")
    start = time.perf_counter()
    for idx, layer in model.layers.named_children():
        compiled_layer = torch.compile(layer, mode="max-autotune", fullgraph=True)
        model.layers.register_module(idx, compiled_layer)
    end = time.perf_counter()
    logger.info(f"Finished compiling each {model.__class__.__name__} layer in {end - start:.4f} seconds.")


def load_stu_model(config_data, checkpoint_path: str, device: torch.device, futurefill_k: Union[None, int], lds_state_dim = None, lds_path = None):

    torch_dtype = getattr(torch, config_data["torch_dtype"])
    is_futurefill = futurefill_k is not None
    is_lds = lds_state_dim is not None
    

    model_config = FlashSTUConfig(**config_data)
    model_config.torch_dtype = getattr(torch, config_data["torch_dtype"])
    is_futurefill = futurefill_k is not None

    # spectral_filters = get_spectral_filters(model_config.seq_len, model_config.num_eigh, model_config.use_hankel_L, device, torch_dtype)

    spectral_filters = torch.randn(model_config.seq_len, model_config.num_eigh, dtype = torch_dtype).cuda()

    model = FlashSTU(model_config, spectral_filters, future_fill = futurefill_k)
    model = model.to(device=device, dtype=torch_dtype)
    
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = {}
        start_time = time()

        if checkpoint_path.endswith(".safetensors"):
            with safe_open(checkpoint_path, framework="pt", device=device.type) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        elif checkpoint_path.endswith(".pt"):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        logger.info(f"Checkpoint loaded in {time() - start_time:.2f} seconds.")

        model.load_state_dict(state_dict, strict=True)
        logger.info("Model weights loaded successfully!")

    model.setup_phi()
    if is_futurefill:
        for idx, layer in enumerate(model.layers):
            if hasattr(layer, "stu"):
                logger.warning("Right now, FutureFill is intialized with the model ... So can't run multiple sequence length on the same run")
                model.layers[idx].stu.setup_ff(futurefill_k)

    
    if is_lds:
        model.setup_lds(state_dim = lds_state_dim, lds_path = lds_path, dtype = torch.bfloat16)
        # print("LDS Setup")
        print(model)
        
    if config_data["torch_compile"]:
        model = apply_compile(model)
    model.eval()

    return model, config_data

def generate_text(
    model,
    tokenizer,
    prompt,
    num_return_sequences=1,
    max_length=512,
    device="cuda",
    temperature=1.0,
    top_k=50,
    cache = True, 
    futurefill_k = None
):
    """
    Generate text from the given prompt using top-k sampling.

    Args:
        model: The FlashSTU model instance.
        tokenizer: The tokenizer used for encoding/decoding.
        prompt (str | torch.tensor): Input prompt text.
        num_return_sequences (int): How many sequences to return.
        max_length (int): Maximum length of generated tokens.
        device: torch device.
        temperature (float): Sampling temperature. Higher = more random.
        top_k (int): Top-K sampling parameter.

    Returns:
        list[str]: A list of generated text sequences.
    """

    # Encode prompt tokens.
    if isinstance(prompt, torch.Tensor):
        if prompt.numel() == 0:
            if tokenizer.name.lower() == "o200k_base":
                tokens = torch.tensor([[tokenizer.eot_token]], device=device)
            elif "gpt" in tokenizer.name.lower():
                tokens = torch.tensor([[tokenizer.bos_token_id]], device=device)
        else:
            tokens = prompt
    else:
        tokens = torch.tensor(
            [tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})],
            device=device,
        )
    
    seq_len = tokens.shape[1]
    tokens = tokens.repeat(num_return_sequences, 1)
    
    input_pos = torch.arange(seq_len, device=device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1746)

    eos_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]

    cur_token = seq_len
    with torch.no_grad():
        # for idx in tqdm.tqdm(range(max_length - tokens.size(1))):
        for idx in range(max_length - tokens.size(1)):
            with torch.amp.autocast(device_type="cuda", dtype=model.config.torch_dtype):

                # Fwd pass. Inspect logits here.
                if not cache and futurefill_k is None:
                    logits = model(tokens, input_pos = input_pos)
                elif idx != 0:
                    logits = model(tokens[:, -1:], input_pos = input_pos)     # shape: [batch, 1, vocab]
                else:
                    logits = model(tokens, input_pos = input_pos)     # shape: [batch, seq, vocab]
                logits = logits[:, -1, :]  # last token logits

                # Apply temperature scaling.
                if temperature > 0:
                    logits = logits / temperature

            # # Compute probabilities -> no need for proba
            # probs = F.softmax(logits, dim=-1)

            # # Top-K sampling
            # top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            # ix = torch.multinomial(top_k_probs, 1, generator=sample_rng)
            # next_token = torch.gather(top_k_indices, -1, ix)
            
            #argmax sampling
            next_token = torch.argmax(logits, dim = -1, keepdim=True)

            # Append next token.
            tokens = torch.cat((tokens, next_token), dim=1)
            input_pos = torch.tensor([cur_token]).to(device)
            cur_token +=1 

            # # Stop if EOS token is generated -> we don't want to stop
            # if (next_token == eos_token_id).any():
            #     break

            # Decode all sequences.
        generated_sequences = []
        try:
            for i in range(num_return_sequences):
                decoded = tokenizer.decode(tokens[i].tolist())
                generated_sequences.append(decoded)
        except:
            pass

    return generated_sequences

def generate_and_time(model, tokenizer, eval_config, save_path_for_this_exp, device):
    total_tokens = 0
    start_time = time()
    cache = eval_config.get("cache", False)
    max_length = eval_config.get("max_length", [32])
    input_length = eval_config.get("input_length", [32])
    BASE_TEMPERATURE = eval_config.get("temperature", 0.7)
    BASE_TOP_K = eval_config.get("top_k", 50)
    num_repeat = eval_config.get("repeat", 6)
    debug = eval_config.get("debug", False)
    futurefill_k = eval_config.get("futurefill_k", None)

    for i, input_length in enumerate(input_length, 1):
        runtimes = {}
        for j, final_length in enumerate(max_length):
            running_runtime = []
            for repeat in range(num_repeat):
                logger.info(f"Generating text for prompt {i} of length {input_length}. Max generation is {final_length}")
                
                if cache:    
                    # if not isinstance(futurefill_k, list):
                    model.setup_caches(batch_size = 1)


                if not cache and model.caches_are_enabled():
                    model.reset_caches()
                
                _dummy_prompt_ids = torch.full((1, input_length), 1)
                # prompt = "Obama is famous for"
                
                start_time = time()
                tokens = generate_text(
                    model,
                    tokenizer,
                    _dummy_prompt_ids, #prompt,
                    num_return_sequences=1,
                    max_length=final_length,
                    device=device,
                    temperature=BASE_TEMPERATURE,
                    top_k=BASE_TOP_K,
                    cache = cache, 
                    futurefill_k = futurefill_k
                )

                end_time = time()
                current_runtime = end_time - start_time
                running_runtime.append(current_runtime)

                if cache:
                    if futurefill_k is None:
                        model.reset_caches()

                    elif futurefill_k is not None:
                        for idx, layer in enumerate(model.layers):
                            if hasattr(layer, "stu"):
                                model.layers[idx].stu.eff_plus.reset_cache()
                                if not model.config.use_hankel_L:
                                    model.layers[idx].stu.eff_minus.reset_cache()

                logger.info(f"Current runtime: {current_runtime}")
                # logger.info(f"Output: {tokens}")

            mean_running_time = np.mean(running_runtime)
            mean_running_time_wo_first = np.mean(running_runtime[1:])
            runtimes[final_length] = mean_running_time
            runtimes[f"{str(final_length)}-wo-first"] = mean_running_time_wo_first
            
            # Create detailed stats dictionary
            detailed_stats = {
                "final_length": final_length,
                "mean_runtime": mean_running_time,
                "mean_runtime_wo_first": mean_running_time_wo_first,
                "std_runtime": np.std(running_runtime),
                "std_runtime_wo_first": np.std(mean_running_time_wo_first),
                "min_runtime": min(running_runtime),
                "max_runtime": max(running_runtime),
                "all_runtimes": running_runtime
            }
            
            # Log the stats
            logger.info(f"final_length: {final_length}")
            logger.info(f"Mean runtime: {mean_running_time}")
            logger.info(f"Mean runtime wo first: {mean_running_time_wo_first}")
            logger.info(f"std runtimes: {np.std(running_runtime)}")
            logger.info(f"std runtimes wo first: {np.std(mean_running_time_wo_first)}")
            logger.info(f"min runtimes: {min(running_runtime)}")
            logger.info(f"max runtimes: {max(running_runtime)}")
            logger.info("-----\n")

            # Save detailed stats to JSON
            stats_filename = f"detailed_stats_length_{final_length}.json"
            with open(os.path.join(save_path_for_this_exp, stats_filename), "w") as f:
                json.dump(detailed_stats, f, indent=4, default=float)

            if debug:
                try:
                    generated_text = tokenizer.decode(tokens[0].tolist())
                    total_tokens += len(tokenizer.encode(generated_text, allowed_special={"<|endoftext|>"}))
                except:
                    pass
                # logger.info(f"\nPrompt: {_dummy_prompt_ids}")
                # logger.info(f"Generated Text: {generated_text}\n")
                # logger.info(f"Generated Tokens: {tokens}\n")
               

        logger.info("Runtimes in seconds: \n")
        logger.info(runtimes)
        logger.info("-----\n")

        # Save using output dir and model name and date
        sub_exp_name = f"runname={eval_config.get('run_name')}-input_seqlen={input_length}-max_output_seqlen={max_length[0]}-numlayer={model.config.num_layers}-dim={model.config.dim}-attn={model.config.use_attn}.json"
        with open(os.path.join(save_path_for_this_exp, sub_exp_name), "w") as file:
            json.dump(runtimes, file)
        

def main(args):
    # Load eval config
    eval_config = yaml.load(open(args.eval_path, 'r'), Loader=yaml.FullLoader)
    with open(args.config_path, "r") as f:
        model_config = json.load(f)

    # Create output dir; save yaml configs in there
    run_id = random.randint(0, 10 ** 6)

    # Set random seed for reproducibility
    set_initial_random_seed(eval_config.get('random_seed', -1))

    # save yaml configs in there
    save_path_for_this_exp = os.path.join(eval_config.get('save_dir'), str(run_id))
    os.makedirs(save_path_for_this_exp, exist_ok=True)
    logger.info(f"For this experiment, saving path is: {save_path_for_this_exp}")
    logger.info(f"eval_config: {eval_config}")
    logger.info(f"model_config: {model_config}")
    save_yaml_config(eval_config, save_path_for_this_exp, "eval_config.yaml")
    save_yaml_config(model_config, save_path_for_this_exp, "model_config.yaml")

    # Need to calculate futurefill K here ... TO BE IMPROVED
    futurefill_k = eval_config.get("futurefill_k", None)
    if futurefill_k is not None:
        if isinstance(futurefill_k[-1], str) and futurefill_k[-1] == "None":
            generation_L = eval_config.get("max_length")[-1]
            futurefill_k = int(math.sqrt(generation_L * math.log2(generation_L)))
        elif isinstance(futurefill_k[-1], int):
            futurefill_k = futurefill_k[-1]

    lds_state_dim = eval_config.get('lds_state_dim', None)
    lds_path = eval_config.get('lds_path', None)
    
    # Load model and config.
    device = torch.device("cuda")
    print("STARTING LOAD")
    model, config_data = load_stu_model(model_config, args.checkpoint_path, device, futurefill_k = futurefill_k, lds_state_dim = lds_state_dim, lds_path = lds_path)
    print("MODEL LOADED")
    # Create tokenizer (for della)
    bpe_dict = load_tiktoken_bpe(bpe_path)
    tokenizer = tiktoken.Encoding(
        name="o200k_base",  # Name of the encoding
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""",
        mergeable_ranks=bpe_dict,
        special_tokens={
            "<|endoftext|>": 199999,  # Custom special token example (modify as needed)
            "<|endofprompt|>": 200018,
        }
    )

    generate_and_time(model, tokenizer, eval_config, save_path_for_this_exp, device)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    CHECKPOINT_PATH = "" #"./models/model_step-114000.safetensors"
    CONFIG_PATH = "./configs/stu_only/future_fill/config.json"
    EVAL_PATH = "./configs/stu_only/future_fill/eval.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_PATH, type=str)
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str)
    parser.add_argument('--eval_path', default=EVAL_PATH, type=str)
    
    args = parser.parse_args()

    main(args)