# Prodigy + ScheduleFree
*Eliminating hyperparameters, one commit at a time.*

**Current status:** Experimental

## Details
An optimiser based on Prodigy that includes schedule-free logic and much, much lower memory usage, the aim being to remove the need to set any hyperparameters. Of course,
that's never the case with any optimiser, but hopefully, this comes close!

Hyperparameters eliminated: Learning rate (Prodigy), LR scheduler (ScheduleFree), epsilon (Adam-atan2). Still working on betas and weight decay, though
those are much harder.

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
already handled by Adam-atan2, which naturally bounds the updates to $(-\pi, \pi] $.

Setting `beta4` to None or a positive value will treat the stepsize as a running average, and allow the stepsize to 
both decrease and increase over time as the loss landscape changes. This is contrary to Prodigy's default behaviour, which never decreases the stepsize.
A positive `beta4` value will always use a moving average to update `d`, while a negative `beta4` will only update `d` as a moving average when the
adaptive step prediction is decreasing (using `abs(beta4)`), which provides a natural decay-like effect.

Recommended values for `beta4` if set manually are 0.99-0.999, with lower values making the adaptation more aggressive.
Setting it to 0 disables the feature, while None will use `beta2`.

By default, `split_groups` is set to True, so each parameter group will have its own adaptation values. So if you're training
different models together, they won't contaminate each other's learning rates. 

The disadvantage of this approach is that some models can take a long time to reach a good learning rate when trained alongside others 
(for example, SDXL's Unet). The default `split_groups_mean` takes care of this, however, you can also try setting `split_groups_mean` to None
and using a higher `d0` (1e-5, 5e-5, 1e-4) instead so these networks don't get stuck at a low learning rate.

Set `split_groups` to False to mimic Prodigy's normal behaviour, which uses a single set of values for all parameters.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the `prodigy_steps` settings. [It's been suggested that all Prodigy needs to do is achieve "escape velocity"](https://arxiv.org/pdf/2409.20325)
in terms of finding a good LR, which it usually achieves after ~25% of training, though this is very dependent on batch size and epochs. 

This setting can be particularly helpful when training diffusion models, which have very different gradient behaviour than what most optimisers are tuned for. Prodigy in particular
will increase the LR forever if it is not stopped or capped in some way (usually via a decaying LR scheduler).
