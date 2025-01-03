# Prodigy + ScheduleFree
*Eliminating hyperparameters, one commit at a time.*

**Current status:** Experimental

## Installation
```
pip install prodigy-plus-schedule-free
```

## Usage
```python
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, betas=(0.9, 0.99), beta3=None, 
                                    weight_decay=0.0, weight_decay_by_lr=True, 
				    use_bias_correction=False, d0=1e-6, d_coef=1.0, 
				    prodigy_steps=0, eps=1e-8, 
				    split_groups=True, split_groups_mean=True,
 				    factored=True, fused_back_pass=False, use_stableadamw=True,
 				    use_muon_pp=False, use_cautious=False, use_adopt=False, 
				    stochastic_rounding=True)
```

As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
calls to `optimizer.train()` and `optimizer.eval()`. See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free

## Details
An optimiser based on Prodigy that includes schedule-free logic and much, much lower memory usage, the aim being to remove the need to set any hyperparameters. Of course,
that's never the case with any optimiser, but hopefully, this comes close!

Hyperparameters eliminated: Learning rate (Prodigy), LR scheduler (ScheduleFree), epsilon (Adam-atan2, optional, not enabled by default).

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:

1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon. This is the default.)
2) `eps=None` (Adam-atan2, scale invariant, but can mess with Prodigy's stepsize calculations in some scenarios.)

By default, `split_groups` is set to `True`, so each parameter group will have its own adaptation values. So if you're training
different networks together, they won't contaminate each other's learning rates. For Prodigy's reference behaviour, which lumps all 
parameter groups together, set `split_groups` to `False`.

The optimiser uses low-rank approximations for the second moment, much like Adafactor. There should be little to no difference 
in training performance, but your mileage may vary. If you encounter problems, you can try disabling factorisation by 
setting `factored` to `False`.

The optimiser also supports [fused backward pass](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) to significantly lower
gradient memory usage. The `fused_back_pass` argument must be set to `True` so the optimiser knows not to perform the regular step. Please note however that 
your training scripts / UI of choice *must* support the feature for generic optimisers -- as of December 2024, popular trainers such as OneTrainer and Kohya 
hard-code which optimisers have fused backward pass support, and so this optimiser's fused pass will not work out of the box with them.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the `prodigy_steps` settings. [It's been suggested that all Prodigy needs to do is achieve "escape velocity"](https://arxiv.org/pdf/2409.20325)
in terms of finding a good LR, which it usually achieves after ~25% of training, though this is very dependent on batch size and epochs. 

This setting can be particularly helpful when training diffusion models, which have very different gradient behaviour than what most optimisers are tuned for. 
Prodigy in particular will increase the LR forever if it is not stopped or capped in some way (usually via a decaying LR scheduler).

## Experimental features

**Adam-atan2:** Enabled by setting `eps` to `None`. Outlined in [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872), 
you can use atan2 in place of the regular division plus epsilon found in most Adam-style optimisers. This makes updates scale-invariant, and removes the need to tweak the epsilon.
This seems to work fine in some models (SDXL), but cripples Prodigy's stepsize calculations in others (SD3.5 Medium and Large). Disabled by default.

**Orthogonalisation:** Enabled by setting `use_muon_pp` to `True`. This changes the base behaviour of the optimiser for compatible parameters from AdamW to SGD.
[As explained by Keller Jordan](https://x.com/kellerjordan0/status/1844782418676339059), and demonstrated (in various forms) by optimisers such as Shampoo, SOAP 
and Jordan's Muon, applying orthogonalisation/preconditioning can improve convergence. However, this approach may not work in some situations 
(small batch sizes, fine-tuning) and as such, is disabled by default.

**C-Optim:** Enabled by setting `use_cautious` to `True`. Outlined in [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085). 
Applies a simple modification to parameter updates that promotes values that are aligned with the current gradient. This should result in faster convergence. Note that
the proposed changes are not 1:1 compatible with schedule-free, so more testing is required.

**ADOPT:** Enabled by setting `use_adopt` to `True`. A partial implementation of [ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate](https://arxiv.org/abs/2411.02853), as we only update the second moment after the parameter update, so as to exclude the current gradient. Disabled by default.

## Recommended usage
 
The schedule-free component of the optimiser works best with a constant learning rate. In most cases, Prodigy will find the optimal learning rate within the first
25% of training, after which it may continue to increase the learning rate beyond what's best (this is mostly observed with diffusion training).

It is strongly recommended to set `prodigy_steps` equal to 25% of your
total step count, though you can experiment with values as little as 5-10%, depending on the model and type of training. The best way to figure out the best value
is to monitor the `d` value(s) during a training run.

![image](https://github.com/user-attachments/assets/b68f0869-7232-4a2d-a396-e0f9ea21f63b)

Here is an example of an SDXL LoRA run. From left to right are the `d` values (essentially the learning rate predicition) for TE1, TE2 and the Unet. 
In this run, `prodigy_steps` was set to `20`, as the optimal LR was found around step 15.

![image](https://github.com/user-attachments/assets/d3077b0d-5f23-4500-b2b3-fc0cf45d2da7)

This image shows a different run with the same dataset, but with `prodigy_steps` set to `0`. While the text encoders were mostly stable, the Unet LR continued to grow throughout training.
