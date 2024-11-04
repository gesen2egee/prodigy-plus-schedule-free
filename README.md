# Prodigy + ScheduleFree

**Current status:** Experimental

## Updates
**Update #2 - 4/11/2024:** Adapted factored second moment [from this pull request](https://github.com/konstmish/prodigy/pull/25) for reduced memory usage. Toggled via `factored=True`.
Keep the first moment as-is, as literature suggests performance suffers without it (https://arxiv.org/pdf/2106.04560, section 3.4).

**Update #1 - 4/11/2024:** Now uses Adam-atan2 by default for parameter updates (https://arxiv.org/abs/2407.05872). This is
not compatible with StableAdamW, however, the use of atan2 appears to naturally bound the updates. Needs more testing, can be disabled via `adam_atan2=False`.

## Details
An optimiser based on Prodigy that includes toggleable schedule-free logic. Has additional improvements in the form of
StableAdamW gradient scaling, per parameter group adaptation, lower memory utilisation and moving average stepsizes.

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20

As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
calls to train() and eval(). See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Do not use with gradient clipping, as this can hamper the ability for the optimiser to adapt. Scaling of large gradients is 
already handled by StableAdamW, which effectively uses Adafactor's gradient clipping.

Setting beta4 to None or a positive value will treat the stepsize as a running average, and allow the stepsize to 
both decrease and increase over time as the loss landscape changes. This is contrary to Prodigy's default behaviour, which never decreases the stepsize.

Recommended values for beta4 if set manually are 0.99-0.999, with lower values making the adaptation more aggressive.
Setting it to 0 disables the feature, while None will use beta2.

By default, split_groups is set to True, so each parameter group will have its own adaptation values. So if you're training
different networks together, they won't contaminate each other's learning rates. The disadvantage of this approach is that some 
networks can take a long time to reach a good learning rate when trained alongside others (for example, SDXL's Unet). 
It's recommended to use a higher d0 (1e-5, 5e-5, 1e-4) so these networks don't get stuck at a low learning rate.

Set split_groups to False to mimic Prodigy's normal behaviour, which uses a single set of values for all parameters.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the prodigy_steps settings.

## Default settings
`lr=1.0, use_schedulefree=True, betas=(0.9, 0.99), beta3=None, beta4=0, weight_decay=0.0, use_bias_correction=False, d0=1e-6, d_coef=1.0, eps=1e-8, prodigy_steps=0, warmup_steps=0, split_groups=True, slice_p=10, bf16_state=False, adam_atan2=True, factored=True`
## Recommended settings
`lr=1.0, use_schedulefree=True, betas=(0.9, 0.99), beta3=None, beta4=None, weight_decay=0.01, use_bias_correction=False, d0=5e-5, d_coef=1.0, eps=1e-8, prodigy_steps=0, warmup_steps=0, split_groups=True, slice_p=10, bf16_state=False, adam_atan2=True, factored=True`
