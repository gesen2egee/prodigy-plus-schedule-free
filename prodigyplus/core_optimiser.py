import math
import torch
from statistics import harmonic_mean

class CoreOptimiser(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.99), beta3=None,
                 weight_decay=0.0,
                 weight_decay_by_lr=True,
                 use_bias_correction=False,
                 d0=1e-6, d_coef=1.0,
                 prodigy_steps=0,
                 eps=1e-8,
                 split_groups=True,
                 split_groups_mean=True,
                 factored=True,
                 fused_back_pass=False,
                 use_stableadamw=True,
                 use_muon_pp=False,
                 use_cautious=False,
                 use_grams=False,
                 use_adopt=False,
                 stochastic_rounding=True):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps is not None and not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))

        if beta3 is None:
            beta3 = betas[1] ** 0.5

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps,
                        weight_decay=weight_decay,
                        weight_decay_by_lr=weight_decay_by_lr,
                        d=d0, d0=d0, d_coef=d_coef,
                        k=1, train_mode=True,
                        weight_sum=0,
                        prodigy_steps=prodigy_steps,
                        use_bias_correction=use_bias_correction,
                        d_numerator=0.0,
                        d_denom=0,
                        factored=factored,
                        use_stableadamw=use_stableadamw,
                        use_muon_pp=use_muon_pp,
                        use_cautious=use_cautious,
                        use_grams=use_grams,
                        use_adopt=use_adopt,
                        stochastic_rounding=stochastic_rounding)

        super().__init__(params, defaults)

        self.d0 = d0
        if split_groups and len(self.param_groups) == 1:
            print(f"[{self.__class__.__name__}] Optimiser contains single param_group -- 'split_groups' has been disabled.")
            split_groups = False

        self.split_groups = split_groups
        self.split_groups_mean = split_groups_mean

        # Properties for fused backward pass.
        self.groups_to_process = None
        self.shared_d = None
        self.fused_back_pass = fused_back_pass

        # Use tensors to keep everything on device during parameter loop.
        for group in (self.param_groups if self.split_groups else self.param_groups[:1]):
            p = group['params'][0]
            group['running_d_numerator'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            group['running_d_denom'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True
    
    def supports_fused_back_pass(self):
        return True

    @torch.no_grad()
    def get_sliced_tensor(self, tensor, slice_p=11):
        return tensor.ravel()[::slice_p]
   
    @torch.no_grad()
    def get_running_values_for_group(self, group):
        if not self.split_groups:
            group = self.param_groups[0]
        return group['running_d_numerator'], group['running_d_denom']

    @torch.no_grad()
    def get_d_mean(self):
        if self.split_groups and self.split_groups_mean:
            return harmonic_mean(group['d'] for group in self.param_groups)
        return None

    @torch.no_grad()
    def get_d_max(self, group):
        if self.split_groups:
            return max(group['d'] for group in self.param_groups)
        return group['d']

    # From: https://github.com/KellerJordan/Muon/blob/master/muon.py
    @torch.no_grad()
    def newton_schulz_(self, G, steps=5, eps=1e-7):
        # Inline reshaping step within the method itself.
        X = G.view(G.size(0), -1)

        a, b, c = (3.4445, -4.7750,  2.0315)
        X = X.to(dtype=torch.bfloat16, copy=True)
        X /= (X.norm() + eps) # ensure top singular value <= 1
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T

        G.copy_(X.view_as(G))
        del X

        return G
    
    # Implementation by Nerogar. From: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    def copy_stochastic_(self, target, source):
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

    # Modified Adafactor factorisation implementation by Ross Wightman 
    # https://github.com/huggingface/pytorch-image-models/pull/2320
    @torch.no_grad()
    def factored_dims(self,
        shape,
        factored,
        min_dim_size_to_factor):
        r"""Whether to use a factored second moment estimator.
        This function returns a tuple with the two largest axes to reduce over.
        If all dimensions have size < min_dim_size_to_factor, return None.
        Args:
        shape: an input shape
        factored: whether to use factored second-moment estimator for > 2d vars.
        min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
        Returns:
        None or a tuple of ints
        """
        if not factored or len(shape) < 2:
            return None
        if all(dim < min_dim_size_to_factor for dim in shape):
            return None
        sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
        return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])
    
    @torch.no_grad()
    def initialise_state(self, p, group):
        raise Exception("Not implemented!")

    @torch.no_grad()
    def initialise_state_internal(self, p, group):
        state = self.state[p]
        needs_init = len(state) == 0
        
        if needs_init:
            grad = p.grad
            dtype = torch.bfloat16 if p.dtype == torch.float32 else p.dtype
            sliced_data = self.get_sliced_tensor(p)

            # NOTE: We don't initialise z/exp_avg here -- subclass needs to do that.
            state['muon'] = group['use_muon_pp'] and len(grad.shape) >= 2 and grad.size(0) < 10000

            if not state['muon']:
                factored_dims = self.factored_dims(
                    grad.shape,
                    factored=group['factored'],
                    min_dim_size_to_factor=32
                )

                if factored_dims is not None:
                    dc, dr = factored_dims
                    row_shape = list(p.grad.shape)
                    row_shape[dr] = 1
                    col_shape = list(p.grad.shape)
                    col_shape[dc] = 1
                    reduce_dc = dc - 1 if dc > dr else dc
                    # Store reduction variables so we don't have to recalculate each step.
                    # Always store second moment low ranks in fp32 to avoid precision issues. Memory difference 
                    # between bf16/fp16 and fp32 is negligible here.
                    state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=torch.float32, device=p.device).detach(), 
                                            torch.zeros(col_shape, dtype=torch.float32, device=p.device).detach(), 
                                            dr, dc, reduce_dc]
                else:
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).detach()

            # If the initial weights are zero, don't bother storing them.
            if p.any() > 0:
                state['p0'] = sliced_data.to(dtype=dtype, memory_format=torch.preserve_format, copy=True).detach()
            else:
                state['p0'] = torch.tensor(0.0, dtype=dtype, device=p.device)

            state['s'] = torch.zeros_like(sliced_data, memory_format=torch.preserve_format, dtype=dtype).detach()
        
        return state, needs_init

    @torch.no_grad()
    def update_d_and_reset(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        d = group['d']
        d0 = group['d0']
        d_coef = group['d_coef']
        beta3 = group['beta3']

        running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_numerator_item = running_d_numerator.item()
        d_denom_item = running_d_denom.item()

        # Prevent the accumulation of negative values in the numerator in early training.
        # We still allow negative updates once progress starts being made, as this is 
        # important for regulating the adaptive stepsize.
        if d_numerator_item > 0 or d > d0:
            # Force Prodigy to be extremely confident before increasing the LR when
            # gradient and weights are aligned.
            d_numerator = 0 if d_numerator_item < 0 else d_numerator + d_numerator_item

        if d_denom_item > 0:
            d = max(math.atan2(d_coef * d_numerator, d_denom_item), d)

        group['d'] = d
        group['d_numerator'] = d_numerator

        # Used for logging purposes only.
        group['d_denom'] = d_denom_item

        running_d_numerator.zero_()
        running_d_denom.zero_()

    def on_start_step(self, group):
        if self.groups_to_process is None:
            # Optimiser hasn't run yet, so initialise.
            self.groups_to_process = {i: len(group['params']) for i, group in enumerate(self.param_groups)}
        elif len(self.groups_to_process) == 0:
            # Start of new optimiser run, so grab updated d.
            self.groups_to_process = {i: len(group['params']) for i, group in enumerate(self.param_groups)}

            if not self.split_groups:
                # When groups aren't split, calculate d for the first group,
                # then copy to all other groups.
                self.update_d_and_reset(group)
                for g in self.param_groups:
                    g['d'] = group['d']

            self.shared_d = self.get_d_mean()

    def on_end_step(self, group):
        group_index = self.param_groups.index(group)

        # Decrement params processed so far.
        self.groups_to_process[group_index] -= 1

        # End of param loop for group, update calculations.
        if self.groups_to_process[group_index] == 0:
            k = group['k']
            prodigy_steps = group['prodigy_steps']
            if prodigy_steps > 0 and k == prodigy_steps:
                print(f"[{self.__class__.__name__}] Prodigy stepsize adaptation disabled after {k} steps for param_group {group_index}.")

            self.groups_to_process.pop(group_index)
            if self.split_groups: # When groups are split, calculate per-group d.
                self.update_d_and_reset(group)

            group['k'] = k + 1

            return True

        return False

    def get_dlr(self, group):
        lr = group['lr']

        dlr = (self.shared_d if self.split_groups and self.shared_d else group['d']) * lr
        dlr = max(dlr, group['d0'])
       
        return dlr

    def update_prodigy(self, state, group, grad, data):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps <= 0 or k < prodigy_steps:
            d, d0 = group['d'], group['d0']
            beta3 = group['beta3']

            sliced_grad = self.get_sliced_tensor(grad)
            sliced_data = self.get_sliced_tensor(data).float()

            # Rescale Prodigy updates to compensate for the lost elements due to slicing.
            # This is mostly for logging purposes and shouldn't change the functionality.
            slice_scale = grad.numel() / sliced_grad.numel()
            running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

            s = state['s']

            x0_minus = state['p0'] - sliced_data
            running_d_numerator.add_(torch.dot(sliced_grad, x0_minus), alpha=(d / d0) * d * slice_scale)

            s.mul_(beta3).add_(sliced_grad, alpha=(d / d0) * d * slice_scale)
            running_d_denom.add_(s.abs().sum())

            del x0_minus
        elif 's' in state: # Free the memory used by Prodigy, as we no longer need it.
            del state['s']
            del state['p0']

    def update_(self, num, denom, group):
        d = group['d']
        eps = group['eps']
        
        # Deviate from reference Prodigy -- rather than apply d to the EMAs, 
        # apply directly before calculating the update to simplify the rest of the optimiser.
        num.mul_(d)
        denom.mul_(d * d)

        if eps is None:
            # Adam-atan2. Use atan2 rather than epsilon and division 
            # for parameter updates (https://arxiv.org/abs/2407.05872).
            # Has the nice property of "clipping" the gradient as well.
            update = num.atan2_(denom)
        else:
            update = num.div_(denom.add_(d * eps))

        return update

    def get_denom(self, state):
        exp_avg_sq = state['exp_avg_sq']

         # Adam EMA updates
        if isinstance(exp_avg_sq, list):
            row_var, col_var, _, _, reduce_dc = exp_avg_sq

            row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(1e-30)
            row_factor = row_var.div(row_col_mean).sqrt_()
            col_factor = col_var.sqrt()
            denom = row_factor * col_factor
        else:
            denom = exp_avg_sq.sqrt()

        return denom
   
    def update_first_moment(self, state, group, grad):
        exp_avg = state['exp_avg']
        beta1, _ = group['betas']

        return exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

    def update_second_moment(self, state, group, grad, beta2, return_denom=True, denom_before_update=False):
        exp_avg_sq = state['exp_avg_sq']
        denom = None

        if return_denom and denom_before_update:
            denom = self.get_denom(state)

        # Adafactor / PaLM beta2 decay. Clip beta2 as per Scaling ViT paper.
        if group['use_bias_correction']:
            k = group['k']
            debias = 1 - k ** -0.8
            beta2 = min(beta2, 1 - ((1 - debias) / (1 - debias ** k)))
   
        # Adam EMA updates
        if isinstance(exp_avg_sq, list):
            row_var, col_var, dr, dc, _ = exp_avg_sq

            row_var.lerp_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]),
                weight=1 - beta2
            )
            col_var.lerp_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]),
                weight=1 - beta2
            )
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if return_denom and denom is None:
            denom = self.get_denom(state)

        return denom
        
    def rms_(self, tensor, rms_min):
        rms = tensor.norm().div(tensor.numel() ** 0.5).clamp_min(rms_min)
        return tensor.div_(rms)

    # "Cautious Optimizer (C-Optim): Improving Training with One Line of Code"
    # https://github.com/kyleliang919/c-optim
    # Modified version by Ross Wightman: https://x.com/wightmanr/status/1862226848475955442
    def cautious_(self, update, grad):
        mask = (grad * update > 0).to(dtype=grad.dtype)
        mask.div_(mask.mean().clamp_min(1e-3))
        update.mul_(mask)
        del mask

        return update

    def cautious_mask(self, update, grad):
        mask = (grad * update > 0).to(dtype=grad.dtype)
        mask.div_(mask.mean().clamp_min(1e-3))
        return mask

    # "Grams: Gradient Descent with Adaptive Momentum Scaling"
    # https://arxiv.org/abs/2412.17107
    def grams_(self, update, grad):
        update.copy_(torch.sign(grad) * update.abs())
        return update

    @torch.no_grad()
    def step_param(self, p, group):
        raise Exception("Not implemented!")            

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        if self.fused_back_pass:
            return
        
        """Performs a single optimisation step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            for p in param_group["params"]:
                self.step_param(p, param_group)

        return loss