import math
import torch
import torch.optim

class ProdigyPlusScheduleFree(torch.optim.Optimizer):
    r"""
    An optimiser based on Prodigy that includes togglable schedule-free logic. Has additional improvements in the form of
    StableAdamW gradient scaling, per parameter group adaptation, lower memory utilisation and moving average stepsizes.

    Based on code from:
    https://github.com/facebookresearch/schedule_free
    https://github.com/konstmish/prodigy

    Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD and https://github.com/sangoi-exe):
    https://github.com/konstmish/prodigy/pull/23
    https://github.com/konstmish/prodigy/pull/22
    https://github.com/konstmish/prodigy/pull/20

    As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
    calls to train() and eval(). See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free
    
    If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic, 
    and you should set beta4 to 0 and prodigy_steps to None.

    Leave LR set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
    ability for the optimiser to predict stepsizes. Scaling of large gradients is already handled by StableAdamW, which
    effectively uses Adafactor's gradient clipping.

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

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        use_schedulefree (boolean):
            If set to False, disables schedule-free logic and uses regular AdamW updates instead.
        betas (Tuple[float, float], optional): 
            Coefficients used for computing running averages of gradient and its square 
            (default: (0.9, 0.99))
        beta3 (float):
            Coefficient for computing the Prodigy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        beta4 (float):
            Smoothing coefficient for updating the running average of the Prodigy stepsize. 
            If set to None, beta2 is used instead. Alternatively, set a negative value to only apply
            smoothing when d_hat is less than d (abs(beta4) will be used)
            (default 0, which disables smoothing and uses original Prodigy behaviour).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability.
            Unused if adam_atan2 is True.
            (default: 1e-8).
        weight_decay (float):
            Decoupled weight decay. Value is multiplied by the adaptive learning rate.
            (default: 0).
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        d0 (float):
            Initial estimate for Prodigy (default 1e-6). A higher value may be needed if split_groups
            is set to True and/or beta4 is not 0.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        prodigy_steps (int):
            Freeze Prodigy stepsize adjustments after a certain optimiser step.
            (default 0)
        warmup_steps (int): 
            Enables a linear learning rate warmup (default 0). Use this over the warmup settings
            of your LR scheduler.
        split_groups (boolean):
            Track individual adaptation values for each parameter group. For example, if training
            a text encoder beside a Unet. Note this can have a significant impact on training dynamics.
            Set to False for original Prodigy behaviour, where all groups share the same values.
            (default True)
        slice_p (int):
            Downsamples p0 and s state variables by storing only every nth element. 
            Significantly reduces state memory with a negligble impact on adaptive step size predictions. 
            Higher values reduce memory usage, but have a greater impact on predicition accuracy (default 11).
        bf16_state (boolean):
            Stores the p0 and s state variables in bfloat16. Only relevant if training in float32.
            Can save additional memory, but has much less impact when using slice_p (default False).
        adam_atan2 (boolean):
            Use atan2 rather than epsilon and division for parameter updates (https://arxiv.org/abs/2407.05872). 
            Not compatible with StableAdamW. (default True)
        factored (boolean):
            Use factored approximation of the second moment, similar to Adafactor. Reduces memory usage.
            (default False)
    """
    def __init__(self, params, lr=1.0,
                 use_schedulefree=True,
                 betas=(0.9, 0.99), beta3=None, beta4=0,
                 weight_decay=0.0,
                 use_bias_correction=False,
                 d0=1e-6, d_coef=1.0,
                 eps=1e-8,
                 prodigy_steps=0,
                 warmup_steps=0,
                 split_groups=True,
                 slice_p=11,
                 bf16_state=False,
                 adam_atan2=True,
                 factored=False):
        
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))
        if beta4 is not None and not 0.0 <= beta4 < 1.0:
            raise ValueError("Invalid beta4 parameter: {}".format(beta4))
        if slice_p is not None and slice_p < 1:
            raise ValueError("Invalid slice_p parameter: {}".format(slice_p))

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        weight_decay=weight_decay,
                        d=d0, d0=d0, d_coef=d_coef,
                        k=0,initialised=None,
                        train_mode=True,
                        slice_p=slice_p,
                        weight_sum=0,
                        eps=eps,
                        split_groups=split_groups,
                        prodigy_steps=prodigy_steps,
                        warmup_steps=warmup_steps,
                        beta4=beta4,
                        lr_max=-1,
                        use_bias_correction=use_bias_correction,
                        d_numerator=0.0,
                        bf16_state=bf16_state,
                        adam_atan2=adam_atan2,
                        factored=factored)
        
        self.d0 = d0
        self.split_groups = split_groups
        self.use_schedulefree = use_schedulefree

        super().__init__(params, defaults)

    def eval(self):
        if not self.use_schedulefree:
            return
        
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'].to(p.data.device), weight=1-1/beta1)
                group['train_mode'] = False

    def train(self):
        if not self.use_schedulefree:
            return

        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'].to(p.data.device), weight=1-beta1)
                group['train_mode'] = True

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True
    
    def approx_sqrt(self, row, col):
        r_factor = (row / row.mean(dim=-1, keepdim=True)).sqrt_().unsqueeze(-1)
        c_factor = col.unsqueeze(-2).sqrt()
        return torch.mul(r_factor, c_factor)

    def get_sliced_tensor(self, tensor, slice_p):
        # Downsample the tensor by using only a portion of parameters.
        flat_tensor = tensor.ravel()

        if slice_p is None or slice_p <= 1:
            return flat_tensor

        return flat_tensor[::slice_p]

        # # Memory efficient version but less safe. Rather
        # # than flatten and slice, we just change the view of the tensor.
        # flattened_tensor = tensor.ravel()
        # numel = flattened_tensor.numel() // slice_p
        # stride = (flattened_tensor.stride(0) * slice_p,)
        # sliced_tensor = torch.as_strided(flattened_tensor, size=(numel,), stride=stride)
        # return sliced_tensor

    def initialise_state(self, p, state, slice_p, bf16_state, factored):
        if p.grad is None or len(state) != 0:
            return

        grad = p.grad.data
        sliced_data = self.get_sliced_tensor(p.data, slice_p)

        # z is exp_avg when schedule-free is disabled.
        if self.use_schedulefree:
            state['z'] = p.data.clone().detach()
        else:
            state['z'] = torch.zeros_like(p.data).detach()

        if factored and grad.dim() > 1:
            state["exp_avg_sq_row"] = grad.new_zeros(grad.shape[:-1]).detach()
            state["exp_avg_sq_col"] = grad.new_zeros(grad.shape[:-2] + grad.shape[-1:]).detach()
        else:
            state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

        # If the initial weights are zero, don't bother storing them.
        if p.data.count_nonzero() > 0:
            if bf16_state:
                state['p0'] = sliced_data.to(dtype=torch.bfloat16, copy=True).detach()
            else:
                state['p0'] = sliced_data.clone().detach()
        else:
            state['p0'] = torch.tensor(0.0, device=p.device, dtype=p.dtype)

        state['s'] = torch.zeros_like(sliced_data, dtype=torch.bfloat16 if bf16_state else None).detach()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        first_param = self.param_groups[0]['params'][0]
        device = first_param.device

        if self.split_groups:
            groups = self.param_groups
            all_params = None
        else:
            # Emulate original Prodigy implementation.
            groups = [self.param_groups[0]]
            all_params = []
            for group in self.param_groups:
                all_params.extend(group['params'])

        for group in groups:
            if self.use_schedulefree and not group['train_mode']:
                raise Exception("Not in train mode!")

            lr = group['lr']

            beta1, beta2 = group['betas']
            beta3 = group['beta3']
            beta4 = group['beta4']

            k = group['k'] + 1
            prodigy_steps = group['prodigy_steps']
            warmup_steps = group['warmup_steps']

            d = group['d']
            d_coef = group['d_coef']
            d_numerator = group['d_numerator']

            slice_p = group['slice_p']
            factored = group['factored']

            if beta3 is None:
                beta3 = beta2 ** 0.5

            if beta4 is None:
                beta4 = beta2

            if group['use_bias_correction']:
                bias_correction = ((1 - beta2 ** k) ** 0.5) / (1 - beta1 ** k)
            else:
                bias_correction = 1

            dlr = d * lr * bias_correction
            d_update = dlr * (1 - beta3)

            # Apply warmup separate to the denom and numerator updates.
            if k < warmup_steps:
                dlr *= (k / warmup_steps) ** 2

            d_numerator *= beta3

            # Use tensors to keep everything on device during parameter loop.
            d_numerator_accum = torch.tensor(0.0, dtype=torch.float32, device=device)
            d_denom = torch.tensor(0.0, dtype=torch.float32, device=device)

            params = group['params'] if all_params is None else all_params
            active_p = [p for p in params if p.grad is not None]

            if group['initialised'] is None:
                for p in active_p:
                    self.initialise_state(p, self.state[p], slice_p, group['bf16_state'], factored)
                group['initialised'] = True

            for p in active_p:
                grad = p.grad.data
                state = self.state[p]

                s = state['s']

                sliced_grad = self.get_sliced_tensor(grad, slice_p)
                sliced_data = self.get_sliced_tensor(p.data, slice_p)

                # Adam EMA updates
                if not self.use_schedulefree:
                    state['z'].mul_(beta1).add_(grad, alpha=d * (1 - beta1))

                if factored and grad.dim() > 1:
                    grad_sq = grad.square().add_(1e-30)
                    state["exp_avg_sq_row"].mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=d * d * (1 - beta2))
                    state["exp_avg_sq_col"].mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=d * d * (1 - beta2))
                else:
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

                d_numerator_accum.add_(torch.dot(sliced_grad, state['p0'] - sliced_data), alpha=d_update)

                s.mul_(beta3).add_(sliced_grad, alpha=d_update)
                d_denom.add_(s.abs().sum())

            # Materialise final values off-device once we're done.
            d_numerator += d_numerator_accum.item()
            d_denom_item = d_denom.item()

            # Use atan2 so we no longer need to worry about d_denom being 0. This
            # does reduce the usefulness of d_coef.
            d_hat = max(math.atan2(d_coef * d_numerator, d_denom_item), 1e-6)

            if prodigy_steps <= 0 or k < prodigy_steps:
                if beta4 > 0:
                    # Always update d via EMA.
                    d = d * beta4 + (1 - beta4) * d_hat
                elif beta4 < 0:
                    # Only update d via EMA if d_hat is decreasing.
                    if d_hat >= d:
                        d = d_hat
                    else:
                        beta4 = abs(beta4)
                        d = d * beta4 + (1 - beta4) * d_hat
                else:
                    d = max(d_hat, d)

            weight_decay = dlr * group['weight_decay']
            adam_atan2 = group['adam_atan2']
            eps = group['eps']

            # Split the schedule-free and regular AdamW logic so we don't have 
            # convoluted branching within the loop.
            if not self.use_schedulefree:
                for p in active_p:
                    grad = p.grad.data
                    state = self.state[p]

                    exp_avg = state['z']
                    
                    if factored and grad.dim() > 1:
                        denom = self.approx_sqrt(state["exp_avg_sq_row"], state["exp_avg_sq_col"])
                    else:
                        denom = state['exp_avg_sq'].sqrt()
                   
                    if adam_atan2:
                        update = exp_avg.atan2(denom)
                    else:
                        update = exp_avg.div(denom.add_(d * eps))

                        # StableAdamW.
                        rms = grad.mul(d).pow_(2).div_(denom).mean().sqrt()
                        update.div_(rms.clip(min=1.0))
                        
                    # AdamW-style decoupled weight decay.
                    p.data.mul_(1.0 - weight_decay).add_(update, alpha=-dlr)
            else:
                lr_max = group['lr_max'] = max(dlr, group['lr_max'])

                weight = lr_max ** 2
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                ckp1 = weight / weight_sum if weight_sum else 0

                for p in active_p:
                    y = p.data
                    grad = p.grad.data
                    state = self.state[p]

                    z = state['z']

                    if factored and grad.dim() > 1:
                        denom = self.approx_sqrt(state["exp_avg_sq_row"], state["exp_avg_sq_col"])
                    else:
                        denom = state['exp_avg_sq'].sqrt()

                    if adam_atan2:
                        update = grad.mul(d).atan2(denom)
                    else:
                        update = grad.mul(d).div_(denom.add_(d * eps))

                        # StableAdamW.
                        rms = update.pow(2).mean().sqrt()
                        update.div_(rms.clip(min=1.0))

                    # Weight decay.
                    update.add_(y, alpha=weight_decay)

                    y.lerp_(end=z, weight=ckp1)
                    y.add_(update, alpha=dlr * (beta1 * (1 - ckp1) - 1))

                    z.sub_(update, alpha=dlr)

            group['k'] = k

            group['d'] = d
            group['d_numerator'] = d_numerator

        return loss