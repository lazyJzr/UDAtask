import enum
import math
import torch
import numpy as np
import torch as th
import torch.nn.functional as F
from utils import parse_args
args = parse_args()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    def __init__(
        self,
        *,
        use_timesteps,
        sqrt_etas,
        power_timestep,
        kappa,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.use_timesteps = set(use_timesteps)
        self.kappa = kappa
        self.power_timestep = power_timestep
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.one_minus_alphas_bar_sqrt = sqrt_etas
        self.alphas_prod = 1-sqrt_etas**2
        self.one_minus_alphas_bar_log = torch.log(1-torch.from_numpy(self.alphas_prod).float())
        self.alphas_bar_sqrt = self.alphas_prod**2
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), torch.from_numpy(self.alphas_prod).float()[:-1]], 0)
        
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1 
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
    
    def q_x(self, x_0, t, y=None):
        noise = torch.randn_like(x_0)
        alphas_t = _extract_into_tensor(self.alphas_bar_sqrt, t, x_0.shape)
        assert noise.shape == x_0.shape    
        
        return (alphas_t*(y-x_0) + x_0 + _extract_into_tensor(self.sqrt_etas*self.scale_factor, t, x_0.shape) * noise) if y else (alphas_t*x_0 + _extract_into_tensor(self.sqrt_etas,t,x_0.shape)*noise)
        
    def diffusion_loss_fn(self, model, x_0, n_steps, y=None):
        batch_size = x_0.shape[0]

        t = torch.randint(0, n_steps, size=(batch_size // 2,), device=x_0.device)
        t = torch.cat([t, n_steps - 1 - t], dim=0)  
        t = t.unsqueeze(-1)
        t = t.cuda()

        noise = torch.randn_like(x_0)
        x_noisy = self.q_x(x_0, t, y=None)
        pre_noise = model(x_noisy, t.squeeze(-1))
        diffusion_loss = F.l1_loss(noise, pre_noise)
        return diffusion_loss

    def p_sample_loop(self, model, shape, n_steps):
        cur_x = torch.randn(shape).cuda()
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = self.p_sample(model, cur_x, i, self.power_timestep, self.sqrt_etas)
            x_seq.append(cur_x)
        return x_seq    

    def p_sample(self, model, x, t, betas, one_minus_alphas_bar_sqrt):
        t = torch.tensor([t]).cuda()
        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
        with torch.no_grad():
            eps_theta = model(x, t)
        mean = (1 / np.sqrt(1 - betas[t])) * (x - (coeff * eps_theta))

        noise = torch.randn_like(x)
        sigma_t = np.sqrt(betas[t])

        sample = mean + sigma_t * noise
        return (sample)    

    def p_sample_e(self, model, x_t, y, t, betas, etas, kappa=1.0):
        assert x_t.shape == y.shape == (x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3])

        with torch.no_grad():
            eps_theta = model(x_t, t)

        sqrt_etas = np.sqrt(etas)
        sqrt_etas = _extract_into_tensor(sqrt_etas, t, x_t.shape)
        etas = _extract_into_tensor(etas, t, x_t.shape)
        x0_pred = (x_t - sqrt_etas * kappa * eps_theta - etas * y) / \
                  (1 - etas)

        mean = etas * (y - x0_pred) + x0_pred

        var = _extract_into_tensor(self.power_timestep, t, x_t.shape)
        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean + nonzero_mask * var.sqrt() * noise
    
    def p_sample_loop_e(self, model, shape, n_steps, y, kappa=1.0):
        betas = self.power_timestep
        etas = self.etas
        cur_x = torch.randn(shape).cuda()
        x_seq = [cur_x]

        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), i, dtype=torch.long).cuda()  # batch t
            
            cur_x = self.p_sample_e(model, cur_x, y, t, betas, etas, kappa=kappa)
            x_seq.append(cur_x)

        return x_seq
    
    
    