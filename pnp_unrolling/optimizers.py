import torch


class SLS(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        c=1e-4,
        lr=1.,
        beta=0.5,
        tolerance=1e-8,
        scale_step=True,
    ):

        defaults = {
            "lr": lr,
            "c_constant": c,
            "beta_constant": beta,
            "tolerance": tolerance,
            "scale_step": scale_step,
        }

        super().__init__(
            params,
            defaults
        )

        self.init_step = True

    def step(self, closure):

        with torch.no_grad():
            init_loss = closure()

            for group in self.param_groups:

                beta = group["beta_constant"]
                c = group["c_constant"]
                eps = group["tolerance"]

                norm_grad = torch.sum(
                    torch.tensor(
                        [torch.sum(param.grad ** 2)
                         for param in group["params"]]
                    )
                )

                norm_params = torch.sum(
                    torch.tensor(
                        [torch.sum(param ** 2)
                         for param in group["params"]]
                    )
                )

                if norm_grad == 0:
                    current_cost = init_loss
                    break

                if group["scale_step"] and self.init_step:

                    group["lr"] *= torch.sqrt(norm_params / norm_grad)
                    self.init_step = False

                eta = group["lr"].clone()

                # Learning step v
                for param in group["params"]:
                    param -= beta * eta * param.grad

                init = True
                ok = False

                while not ok:
                    if not init:
                        # Backtracking
                        for param in group["params"]:
                            param -= (beta-1) * eta * param.grad
                    else:
                        init = False

                    # Computing loss with new parameters
                    current_cost = closure()

                    # Stopping criterion
                    if current_cost < init_loss - c * eta * norm_grad:
                        ok = True
                    else:
                        eta *= beta

                    if eta < eps:
                        for param in group["params"]:
                            param += eta * param.grad
                        ok = True

        return current_cost
