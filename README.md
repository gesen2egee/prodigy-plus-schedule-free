# Prodigy + ScheduleFree

**Current status:** Experimental

## Updates
**Update #3 - 6/11/2024:** Removed the non-schedule free path, removed epsilon entirely (Adam-atan2 is default), and tweaked the default settings.

**Update #2 - 4/11/2024:** Adapted factored second moment [from this pull request](https://github.com/konstmish/prodigy/pull/25) for reduced memory usage. Toggled via `factored=True`.
Keep the first moment as-is, as literature suggests performance suffers without it (https://arxiv.org/pdf/2106.04560, section 3.4).

**Update #1 - 4/11/2024:** Now uses Adam-atan2 by default for parameter updates (https://arxiv.org/abs/2407.05872). This is
not compatible with StableAdamW, however, the use of atan2 appears to naturally bound the updates. Needs more testing, can be disabled via `adam_atan2=False`.

## Details
An optimiser based on Prodigy that includes schedule-free logic. Has additional improvements in the form of Adam-atan2 updates, 
per parameter group adaptation, lower memory utilisation, moving average stepsizes and gradient amplification to help Prodigy better
adapt the learning rate when gradients are poor.

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20

As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
calls to `train()` and `eval()`. See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Do not use with gradient clipping, as this can hamper the ability for the optimiser to adapt. Scaling of large gradients is 
already handled by Adam-atan2, which naturally bounds the updates to -pi - pi.

Setting `beta4` to None or a positive value will treat the stepsize as a running average, and allow the stepsize to 
both decrease and increase over time as the loss landscape changes. This is contrary to Prodigy's default behaviour, which never decreases the stepsize.
A positive beta4 value will always use a moving average to update `d`, while a negative beta4 will only update `d` as a moving average when the
adaptive step prediction is decreasing (using `abs(beta4)`), which provides a natural decay-like effect.

Recommended values for beta4 if set manually are 0.99-0.999, with lower values making the adaptation more aggressive.
Setting it to 0 disables the feature, while None will use beta2.

By default, split_groups is set to True, so each parameter group will have its own adaptation values. So if you're training
different networks together, they won't contaminate each other's learning rates. The disadvantage of this approach is that some 
networks can take a long time to reach a good learning rate when trained alongside others (for example, SDXL's Unet). 
It's recommended to use a higher d0 (1e-5, 5e-5, 1e-4) so these networks don't get stuck at a low learning rate.

Set split_groups to False to mimic Prodigy's normal behaviour, which uses a single set of values for all parameters.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the prodigy_steps settings.

An experimental feature, `amplify_gradients` (on by default) applies a signed sqrt to the numerator and denominator calculations,
which can help Prodigy get to a good learning rate, even when gradients are poor. Below a two SDXL LoRA training runs, showing from left to right 
the value of `d` for the first and second text encoders, and the Unet. Orange is with the feature disabled, green with. While it doesn't entirely
fix the issue, the Unet LR is significantly better, and the other LRs are within the same ballpark.

![image](https://github.com/user-attachments/assets/6e790d58-a749-4df8-82e7-87789fd96e1a)
