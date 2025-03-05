import random
from datetime import timedelta
from typing import Dict

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import open_dict, OmegaConf, DictConfig
import os


def parse_opt(opt: DictConfig) -> DictConfig:
    """
    Parse configuration options and set the random seed.

    Args:
        opt (DictConfig): Configuration options.

    Returns:
        DictConfig: Updated configuration options.
    """
    with open_dict(opt):
        opt.cwd = get_original_cwd()

        if "add_depth_channel" in opt.input and opt.input.add_depth_channel:
            opt.input.channel += 1

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    torch.cuda.manual_seed(opt.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    print(OmegaConf.to_yaml(opt))
    return opt


def get_learning_rate(opt: DictConfig, step: int, lr: float) -> float:
    """
    Get the current learning rate according to the learning rate schedule set in the configuration options.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    if opt.training.learning_rate_schedule == 0:
        return lr
    elif opt.training.learning_rate_schedule == 1:
        return get_linear_warmup_lr(opt, step, lr)
    elif opt.training.learning_rate_schedule == 2:
        return get_exponential_decay_lr(opt, step, lr)
    else:
        raise NotImplementedError


def get_exponential_decay_lr(opt, step, lr) -> float:
    if step < opt.training.warmup_steps:
        return lr * step / opt.training.warmup_steps
    else:
        if (step - opt.training.warmup_steps) % opt.training.decay_log == 0:
            newlr = lr * (
                opt.training.lr_decay
                ** ((step - opt.training.warmup_steps) // opt.training.decay_log)
            )
            print("decay lr", newlr)
        return lr


def get_linear_warmup_lr(opt: DictConfig, step: int, lr: float) -> float:
    """
    Calculate the linear warm-up learning rate.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    if step < opt.training.warmup_steps:
        return lr * step / opt.training.warmup_steps
    else:
        return lr


def update_learning_rate(optimizer, opt: DictConfig, step: int):
    """
    Update the learning rate of the optimizer.

    Args:
        optimizer: The optimizer.
        opt (DictConfig): Configuration options.
        step (int): Current training step.

    Returns:
        optimizer: Updated optimizer.
        lr (float): Updated learning rate.
    """
    lr = get_learning_rate(opt, step, opt.training.learning_rate)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def print_results(
    partition: str, step: int, iteration_time: float, metrics: Dict[str, float]
):
    """
    Print training or evaluation results.

    Args:
        partition (str): Partition name (e.g., "train" or "val").
        step (int): Current training step.
        iteration_time (float): Time taken for the iteration.
        metrics (Dict[str, float]): Dictionary of evaluation metrics.
    """
    print(
        f"{partition} \t \t"
        f"Step: {step} \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if metrics is not None:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f} \t", end="")

    print()


def tensor_dict_to_numpy(
    tensor_dict: Dict[str, torch.Tensor], dtype=np.float32
) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of PyTorch tensors into a dictionary of NumPy arrays.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): A dictionary of PyTorch tensors.
        dtype (Type[np.ndarray], optional): Data type for the resulting NumPy arrays. Default is np.float32.

    Returns:
        Dict[str, np.ndarray]: A dictionary of NumPy arrays.
    """
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].detach().cpu().numpy().astype(dtype)
    return tensor_dict


from torch.optim.lr_scheduler import _LRScheduler

import math


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
