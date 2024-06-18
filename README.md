
# nanoGPT.jax

![nanoGPT.jax](assets/nanogpt.jpg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of 
Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) using Jax. The code itself is plain and readable: `train.py` is a ~200-line boilerplate training loop and `model.py` a ~200-line GPT model definition.

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints.

## Install

```
pip install numpy jax flax optax orbax tqdm
```

Dependencies:
- `numpy` for numeric computation <3
- `jax`, `flax`, `optax`, `orbax` for jax liberaies <3
-  `tqdm` for progress bars <3

## Quick start

Currently GPT-2 model loading, finetuning, and inference are not supported, because I don't have a GPU :p

We will demonstrate the GPT pipeline with a simple task: generating text from Shakespeare's works.

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT.

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a bunch of customizable parameters. Feel free to adjust it based on your needs. On my laptop (MacBook Pro M2 Max 32GB), it takes ~5min to finish the training. The final train and eval loss values are around 1.5.

```
iter 5000 loss: train loss 1.4159, val loss 1.6152
```


By default, the model checkpoints are written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
The bids of man the more royal in hate,
To that where see's husband: we have subded himle
Might a may and no true--

LEONTES:
Is in the movereign of the of raid day
And your choods burthsome and him.

NORTHUMBERLAND:
Ay, madam, I have preform a scapade acces help;
And that hath her sweet's and to the feart.
Should let thee be and their service for thyself.

LEONTES:
If down the is must me and it would the soul
The toward of his for mother charge scarried, that I would
And look me heart to to chi
```

Not bad for ~5 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters.

## TODOs

- Implement GPT-2 model loading, finetuning, and inference.

- Write a `jax.lax.scan` version of the sampling step to make it a `jax.jit`. The issue I was having is that, with a fixed block size to crop, we have to slice the `carry` tensor with a running index `x`; however, `jax.lax.slice` does not support referencing the scan function's arg. The solution `cgarciae/nanoGPT-jax` proposed does not support a block size smaller than `max_new_tokens`.

## Acknowledgements

`cgarciae/nanoGPT-jax` provided some insights for me to migrate the code to Jax. Thanks Cristian!