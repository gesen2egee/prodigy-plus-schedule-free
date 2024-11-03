# Prodigy + ScheduleFree

**Current status:** Experimental

An optimiser based on Prodigy that includes togglable schedule-free logic. Has additional improvements in the form of
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

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic, 
and you should set beta4 to 0 and prodigy_steps to None.

Do not use with gradient clipping, as this can hamper the ability for the optimiser to predict stepsizes. 
Scaling of large gradients is already handled by StableAdamW, which effectively uses Adafactor's gradient clipping.

For default Prodigy + schedule-free behaviour, set beta4 to 0 and prodigy_steps to None. Setting beta4 to None or a positive
value will treat the stepsize as a running average, and allow the stepsize to both decrease and increase over time. This is
contrary to Prodigy's default behaviour, which never decreases the stepsize.

Recommended values for beta4 if set manually are 0.99-0.999, with lower values making the adaptation more noisy and aggressive.
If beta4 is set to None, beta2 is used.

By default, split_groups is set to True, so each parameter group will have its own adaptation values. So if you're training
different networks together, they won't contaminate each other's learning rates. The disadvantage of this approach is that some 
networks can take a long time to reach a good learning rate when trained alongside others (for example, SDXL's Unet). 
It's recommended to use a higher d0 (1e-5, 5e-5, 1e-4) so these networks don't get stuck at a low learning rate.

For Prodigy's default behaviour, which lumps all parameter groups together, set split_groups to False.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the prodigy_steps settings.
