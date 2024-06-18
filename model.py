"""
Full definition of a GPT Language Model, all of it in this single file.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.training import train_state
import optax


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    use_bias: bool = True # True: use_bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    seed = 1337


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(3 * config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02), use_bias=config.use_bias)
        # output projection
        # apply special scaled init to the residual projections, per GPT-2 paper
        self.c_proj = nn.Dense(config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02/jnp.sqrt(2 * config.n_layer)), use_bias=config.use_bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        tril_mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(tril_mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.c_fc    = nn.Dense(4 * config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02), use_bias=config.use_bias)
        # apply special scaled init to the residual projections, per GPT-2 paper
        self.c_proj  = nn.Dense(config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02/jnp.sqrt(2 * config.n_layer)), use_bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, use_bias=config.use_bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, use_bias=config.use_bias)
        self.mlp = MLP(config)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.vocab_size > 0
        assert config.block_size > 0

        self.wte = nn.Embed(config.vocab_size, config.n_embd, embedding_init=nn.initializers.normal(stddev=0.02))
        self.wpe = nn.Embed(config.block_size, config.n_embd, embedding_init=nn.initializers.normal(stddev=0.02))
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5, use_bias=config.use_bias)

    def __call__(self, idx: jax.Array, *, train: bool, targets: Optional[jax.Array] = None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward a sequence of length {T}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, T, dtype=jnp.int32)[None] # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.wte.attend(x) # https://paperswithcode.com/method/weight-tying
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.wte.attend(x[:, [-1], :]) # https://paperswithcode.com/method/weight-tying
            loss = None

        return logits, loss
        
    def configure_optimizers(self, params, weight_decay, learning_rate, beta1, beta2):

        def decay_selector_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ('bias', 'scale'):
                return False
            elif path[-1] in ('kernel', 'embedding'):
                return True
            else:
                raise NotImplementedError(f"decay selector fails to parse: {path}")

        decay_to_opt_dict = {    
            True: optax.adamw(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay), 
            False: optax.adamw(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                weight_decay=0.0)}
        tx = optax.multi_transform(decay_to_opt_dict, freeze(path_aware_map(decay_selector_fn, params)))
        return tx
    
    def configure_state(self, learning_rate, weight_decay, beta1, beta2, decay_lr, warmup_iters, lr_decay_iters, min_lr, grad_clip, params=None, **kwargs):
        if params is None:
            variables = self.init(jax.random.PRNGKey(self.config.seed), jnp.ones((1, 1), dtype=jnp.int32), train=False)
            params = freeze(variables['params'])
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_iters, 
            decay_steps=lr_decay_iters,
            end_value=min_lr,
        ) if decay_lr else learning_rate
        tx = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            self.configure_optimizers(params, weight_decay, lr_schedule, beta1, beta2)
        )
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)
        

    def generate(self, params, key, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in eval mode of operation for this.
        """
        for t in range(max_new_tokens):
            # sample step key
            step_key = jax.random.fold_in(key, t)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cropped = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.apply({'params': params}, idx_cropped, train=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                top_logits, top_idx = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                next_index_i = jax.random.categorical(step_key, top_logits, axis=-1)
                next_index = jnp.take_along_axis(top_idx, next_index_i[:, None], axis=-1).squeeze(-1)
            else:
                next_index = jax.random.categorical(step_key, logits, axis=-1)
            # append sampled index to the running sequence and continue
            idx = jnp.append(idx, next_index[None], axis=-1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("Pretrained model loading is not implemented.")
    
    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        raise NotImplementedError("Cropping block size is not implemented.")
