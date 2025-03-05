import time
from collections import defaultdict
from datetime import timedelta
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import omegaconf
from tqdm import tqdm
import numpy as np
from codebase.utils import data_utils, model_utils, utils, rotation_utils, eval_utils

import wandb


def train(
    opt: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[int, torch.nn.Module]:
    """
    Train the model.

    Args:
        opt (DictConfig): Configuration options.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        torch.nn.Module: The trained model.
    """
    training_start_time = time.time()

    train_loader = data_utils.get_data(opt, "train")
    print("1Epoch =", len(train_loader), "steps")
    step = 0
    while step < opt.training.steps:
        for input_images, labels in train_loader:
            start_time = time.time()
            print_results = (
                opt.training.print_idx > 0 and step % opt.training.print_idx == 0
            )

            input_images = input_images.cuda(non_blocking=True)

            optimizer, lr = utils.update_learning_rate(optimizer, opt, step)
            optimizer.zero_grad()

            loss, metrics = model(input_images, labels, evaluate=print_results)
            loss.backward()

            if opt.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt.training.gradient_clip
                )

            optimizer.step()

            # Print results.
            wandb.log(
                {
                    "train/loss": loss,
                    **dict([("train/" + k, v) for k, v in metrics.items()]),
                },
                step=step,
            )
            if print_results:
                iteration_time = time.time() - start_time
                utils.print_results("train", step, iteration_time, metrics)
            if opt.training.save_idx > 0 and step % opt.training.save_idx == 0:
                torch.save(
                    model.state_dict(),
                    f"{HydraConfig.get().runtime.output_dir}/{opt.input.dataset}_{'Color' if opt.input.colored else 'Gray'}_{opt.input.num_answers}_Lv{opt.input.condensed_level}_Step{step}.pt",
                )
            # Validate.
            if (
                opt.training.val_idx > 0
                and step % opt.training.val_idx == 0
                and step > 0
            ):
                validate_or_test(opt, step, model, "val")

            step += 1
            if step >= opt.training.steps:
                break

    total_train_time = time.time() - training_start_time
    print(f"Total training time: {timedelta(seconds=total_train_time)}")
    return step, model


def validate_or_test(
    opt: DictConfig, step: int, model: torch.nn.Module, partition: str
) -> None:
    """
    Perform validation or testing of the model.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        model (torch.nn.Module): The model to be evaluated.
        partition (str): Partition name ("val" or "test").
    """
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = data_utils.get_data(opt, partition)

    model.eval()
    print(partition, len(data_loader))
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            input_images = inputs.cuda(non_blocking=True)

            loss, metrics = model(input_images, labels, evaluate=True)

            test_results["Loss"] += loss.item() / len(data_loader)
            for key, value in metrics.items():
                test_results[key] += value / len(data_loader)
            if opt.evaluation.out:
                np.save("in.npy", input_images.cpu().numpy())
                exit()

    total_test_time = time.time() - test_time
    utils.print_results(partition, step, total_test_time, test_results)

    wandb.log(
        {
            partition + "/loss": loss,
            **dict([(partition + "/" + k, v) for k, v in metrics.items()]),
        },
        step=step,
    )
    torch.save(
        model.state_dict(),
        f"{HydraConfig.get().runtime.output_dir}/{opt.input.dataset}_{'Color' if opt.input.colored else 'Gray'}_{opt.input.condensed_level}_{opt.input.num_answers}_val_{step}.pt",
    )
    model.train()


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_opt(opt)
    import sys

    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    # Initialize model and optimizer.
    model, optimizer = model_utils.get_model_and_optimizer(opt)
    if opt.training.continue_train:
        model.load_state_dict(torch.load(opt.training.load_file))
    if opt.evaluation.test:
        model.load_state_dict(torch.load(opt.evaluation.load_file))
        validate_or_test(opt, 0, model, "test")
        exit()
    expname = HydraConfig.get().runtime.choices.experiment.replace("Tetro_", "")
    lr = opt.training.learning_rate
    bat = opt.input.batch_size
    warm = opt.training.warmup_steps
    mddim = opt.model.hidden_dim
    ladim = opt.model.linear_dim
    rodim = opt.model.rotation_dimensions
    nnaa = [
        expname,
        f"{lr:.0e}",
        f"w{warm}",
        f"b{bat}",
        str((mddim, ladim, rodim)),
    ]
    if opt.input.ceil_input:
        nnaa.append("Ceil")
    wandb.init(
        entity="ConfeitoHS",
        project="RF-Tetromino",
        config=omegaconf.OmegaConf.to_container(opt),
        group=expname,
        name="_".join(nnaa),
    )

    step, model = train(opt, model, optimizer)
    wandb.finish()


if __name__ == "__main__":
    my_main()
