import torch

class NSM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, sigma=0.05, **kwargs):
        assert sigma >= 0.0, f"Invalid sigma, should be non-negative: {sigma}"

        defaults = dict(sigma=sigma, **kwargs)
        super(NSM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def store_gradients(self, zero_grad=False, store_weights=False, update_weight = 0.5):
        ''' store the gradients of original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if store_weights: 
                    self.state[p]["old_p"] = p.data.clone()
                    self.state[p]["old_gradients"] = p.grad.data.clone()*update_weight
                else:
                    assert ("old_gradients" in self.state[p])
                    self.state[p]["old_gradients"] += p.grad.data.clone()*update_weight

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False, store_perturb=True):
        ''' take a perturbation step of the original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"].clone()  # restore original weights 
                if store_perturb:
                    e_w = torch.randn_like(p.data) * group["sigma"]
                    self.state[p]["perturb"] = e_w
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                else:
                    e_w = self.state[p]["perturb"]
                    p.sub_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to original weights
                p.grad.data = self.state[p]["old_gradients"]

        self.base_optimizer.step()  # do the actual weight update

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups