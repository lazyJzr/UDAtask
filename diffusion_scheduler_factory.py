import enum
import math

import torch
import numpy as np
import torch as th
import torch.nn.functional as F
from GaussianDiffusion import GaussianDiffusion
from utils import parse_args
args = parse_args()

def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=0.1,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if args.schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return power_timestep, sqrt_etas


def space_timesteps(num_timesteps, sample_timesteps):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: timesteps for sampling
    :return: a set of diffusion steps from the original process to use.
    """
    all_steps = [int((num_timesteps/sample_timesteps) * x) for x in range(sample_timesteps)]
    return set(all_steps)

def create_gaussian_diffusion(
    *,
    normalize_input,
    schedule_name,
    min_noise_level=0.01,
    steps=args.num_steps,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
):
    power_timestep, sqrt_etas = get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    
    return GaussianDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        power_timestep = power_timestep,
        kappa=kappa,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        latent_flag=latent_flag,
    )

if __name__ == '__main__':
    diffusion = create_gaussian_diffusion(
        normalize_input=True,
        schedule_name=args.schedule_name,
        min_noise_level=1e-3,
        steps=args.num_steps,
        kappa=1.0,
        etas_end=0.95,
        schedule_kwargs={"power": 2},  
        weighted_mse=False,
        timestep_respacing=10, 
        scale_factor=args.scale_factor,
        latent_flag=True,
    )


    