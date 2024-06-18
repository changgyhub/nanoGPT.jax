"""
This training file implemented in Jax currently only supports training with CPU.
For CUDA support, DDP support, and pretrained weight loading, expect future releases :)
"""

import os
import time
import pickle
from functools import partial

import numpy as np
import jax
from orbax import checkpoint as ocp
from flax import serialization
from flax.training import train_state

from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
use_bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
max_checkpoints_to_keep = 5

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# -----------------------------------------------------------------------------
# checkpoint manager
checkpoint_manager = ocp.CheckpointManager(
    ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(out_dir, 'checkpoint'))),
    item_handlers={
        'state': ocp.StandardCheckpointHandler(),
        'model_args': ocp.StandardCheckpointHandler(),
        'iter_num': ocp.StandardCheckpointHandler(),
        'val_loss': ocp.StandardCheckpointHandler(),
        'best_val_loss': ocp.StandardCheckpointHandler(),
    },
    options=ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        best_fn=lambda checkpoint: checkpoint['val_loss'],
        best_mode='min',
        keep_checkpoints_without_metrics=False,
        create=True,
    ),
)

# -----------------------------------------------------------------------------
# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data)-block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int32) for i in ix])
    return x, y

# -----------------------------------------------------------------------------
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")
else:
    vocab_size = 50304
    print(f"defaulting to vocab_size of GPT-2 to {vocab_size} (50257 rounded up for efficiency)")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, use_bias=use_bias, vocab_size=vocab_size, dropout=dropout)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state = model.configure_state(**config)
    params = state.params
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    last_step = checkpoint_manager.latest_step()
    assert last_step is not None, "No checkpoint available!"
    checkpoint = checkpoint_manager.restore(last_step)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'use_bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # load state
    empty_state = jax.eval_shape(lambda: model.configure_state(**config))
    state = serialization.from_state_dict(empty_state, checkpoint['state'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    raise NotImplementedError("Pretrained model loading is not implemented.")
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

# -----------------------------------------------------------------------------
# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            _, loss = forward(state, batch, train=False)
            losses[k] = float(loss)
        out[split] = losses.mean()
    return out

@partial(jax.jit, static_argnames=('train',))
def forward(state, batch, *, train: bool):
    inputs, labels = batch
    rngs = {}
    if train and dropout > 0.0:
        rngs['dropout'] = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
    return state.apply_fn({'params': state.params}, inputs, train=train, targets=labels, rngs=rngs)

@partial(jax.jit, donate_argnums=(0,))
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        state_ = state.replace(params=params)
        _, loss = forward(state_, batch, train=True)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

# -----------------------------------------------------------------------------
# training loop
t0 = time.time()
while True:
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter_num} loss: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        val_loss = losses['val']
        is_best_val = val_loss < best_val_loss
        if is_best_val:
            best_val_loss = val_loss
        if iter_num > 0 and (is_best_val or always_save_checkpoint):
            print(f"saving checkpoint to {out_dir}")
            checkpoint_manager.save(iter_num, args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                model_args=ocp.args.StandardSave(model_args),
                iter_num=ocp.args.StandardSave(iter_num),
                val_loss=ocp.args.StandardSave(val_loss),
                best_val_loss=ocp.args.StandardSave(best_val_loss),
            ))
            checkpoint_manager.wait_until_finished()
    if iter_num == 0 and eval_only:
        break
    
    loss, state = train_step(state, get_batch('train'))

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

checkpoint_manager.close()
