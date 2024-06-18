"""
Sample from a trained model
"""
import os
import pickle
from tqdm import tqdm

import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
dataset = 'shakespeare_char'
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------
# model
if init_from == 'resume':
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(os.path.join(out_dir, 'checkpoint')),
        item_handlers={
            'state': ocp.StandardCheckpointHandler(),
            'model_args': ocp.StandardCheckpointHandler()
        },
    )
    last_step = checkpoint_manager.latest_step()
    assert last_step is not None, "No checkpoint available!"
    checkpoint = checkpoint_manager.restore(last_step)
    checkpoint_manager.close()
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    params = checkpoint['state']['params']
else:
    raise NotImplementedError("Pretrained model loading is not implemented.")

# -----------------------------------------------------------------------------
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume':
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO: make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    raise NotImplementedError("Default (GPT-2) encoding is not implemented.")

# encode the beginning of the prompt
start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jax.random.PRNGKey(gptconf.seed)

# -----------------------------------------------------------------------------
# TODO: write a @jax.jit version of this file.
def generate(params, key, idx) -> jax.Array:
    return model.generate(params, key, idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

def sample(params, key, idx) -> str:
    encoded_str = generate(params, key, idx)[0].tolist()
    return decode(encoded_str)

# -----------------------------------------------------------------------------
# run generation
for k in tqdm(range(num_samples), desc=f'generating sample'):
    # sample from the distribution
    sample_str = sample(params, jax.random.fold_in(key, k), x)
    print('---------BEGIN TEXT---------')
    print(sample_str)
    print('----------END TEXT----------')
