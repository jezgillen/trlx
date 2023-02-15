import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import ray
import torch
import trlx
import wandb
from game import TicTacToeGame
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.search.bayesopt import BayesOptSearch
from transformers import AutoModelForCausalLM, AutoTokenizer

import soft_optim.quantilizer as quantilizer
from soft_optim.fine_tune import infer_game, valid_games_fine_tuned_checkpoint
from soft_optim.metrics import metrics
from soft_optim.trlx_config import default_config_override

wandb_project_name = "soft_optim"


def get_cutoff() -> Tuple[float, float]:
    """Get the quantilizer cutoff"""

    # Model
    model_path = valid_games_fine_tuned_checkpoint
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    # get samples for gen error calculation
    samples = infer_game(model, tokenizer, num_samples=200)
    proxy_rewards: List[float] = []
    human_rewards: List[float] = []
    g_proxy = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    g_human = TicTacToeGame()
    for s in samples:
        proxy_rewards.append(g_proxy.evaluate_game_string(s))
        human_rewards.append(g_human.evaluate_game_string(s))
    proxy_rewards_arr = np.array(proxy_rewards)
    human_rewards_arr = np.array(human_rewards)

    # get generalization error
    eps = 0.05  # <5% chance of bound being exceeded
    bound = quantilizer.empirical_error_bound(
        proxy_rewards_arr, human_rewards_arr, eps)
    # work out proxy reward cutoff
    cutoff, q = quantilizer.get_proxy_value_cutoff(
        bound, len(samples), model, tokenizer)

    return cutoff, q


def loglikelihood_approx(rewards, cutoff):
    alpha = 30.0  # hyperparameter determining sharpness of cutoff
    beta = 0.1
    return np.log(1 / ((1 + np.exp(-alpha * (rewards - cutoff))**(1 /
                                                                  beta))))
    return np.log((rewards > cutoff) + 1e-3)


def soft_opt_experiment(params: Dict[str, float]) -> None:
    """Soft optimization experiment

    Args:
        params: Parameters from Ray
    """
    # do initial tests
    model_path = valid_games_fine_tuned_checkpoint
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    samples = infer_game(model, tokenizer, num_samples=200)
    proxy_rewards: List[float] = []
    human_rewards: List[float] = []
    valid = []
    g_proxy = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    g_human = TicTacToeGame()
    for s in samples:
        proxy_rewards.append(g_proxy.evaluate_game_string(s))
        human_rewards.append(g_human.evaluate_game_string(s))
        valid.append(g_human.validate_game_string(s)[0])
    proxy_rewards_arr = np.array(proxy_rewards)
    human_rewards_arr = np.array(human_rewards)
    print("########################################")
    print(f"Proxy reward of base model: {np.mean(proxy_rewards_arr)}")
    print(f"True reward of base model: {np.mean(human_rewards_arr)}")
    print(f"Validity rate of base model: {np.mean(valid)}")
    print("########################################")

    # Cutoff
    cutoff, q = get_cutoff()
    params["method.target"] = -np.log(q)
    print(q)

    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        rewards_arr = np.array(rewards)
        return rewards_arr
        return loglikelihood_approx(rewards_arr, cutoff)

    # Set params from Ray
    config = default_config_override(params)

    trainer = trlx.train(
        str(valid_games_fine_tuned_checkpoint),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        eval_prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        metric_fn=metrics,
    )

    # Save checkpoints
    fine_tuned_model_path = Path(__file__).parent / \
        ".checkpoints" / "soft_opt_model"
    trainer.save(fine_tuned_model_path)


def tune_function(
    train_function: Callable, param_space: Dict[str, Any], resources: Dict[str, float]
) -> None:
    """Tune a training function with Ray

    Args:
        train_function: Function to train - will receive param_space as a single parameter
        param_space: Parameter space
        resources: Resources per experiment
    """
    tune_config = tune.TuneConfig(
        mode="max",
        # Metric to optimize (can be e.g. "returns/mean" or "metrics/is_valid")
        metric="metrics/true_reward",
        # Use Bayes Search if params are being tuned
        # https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
        search_alg=BayesOptSearch() if len(param_space) >= 1 else None,
        # scheduler=ASHAScheduler(metric="objective", mode="max"))
        num_samples=1,  # Keep sampling forever
        max_concurrent_trials=8
    )

    # Set the metrics to report to the CLI
    reporter = CLIReporter(
        max_progress_rows=10,
        metric_columns=[
            "metrics/true_reward",
            "returns/mean",
            "metrics/is_valid"]
    )

    tuner = tune.Tuner(
        tune.with_resources(train_function, resources=resources),
        param_space=param_space,
        tune_config=tune_config,
        run_config=ray.air.RunConfig(
            local_dir="ray_results",  # Needed for wandb
            callbacks=[
                WandbLoggerCallback(project=wandb_project_name)
            ],
            # log_to_file=True, # Needed
            progress_reporter=reporter,
        ),
    )

    tuner.fit()


def ray_experiment():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Ray: Resources per hyper parameter experiment (i.e. if you want 8
    # runs, you need 8x this number of resources)
    resources: Dict[str, float] = {
        "cpu": 1,
        "gpu": 1,
    }

    # Ray: Param config
    # Good choices from https://arxiv.org/pdf/2006.05990.pdf (in comments
    # below). Must be set using deep dictionary notation.
    param_space: Dict = {
        "method.init_kl_coef": tune.loguniform(0.1, 1),
        "optimizer.kwargs.lr": tune.loguniform(1e-5, 1e-7),
        # "method.gamma": tune.loguniform(0.95, 1.0),
        # # Float to work with search (rounded later)
        # "train.batch_size": tune.loguniform(8, 256),
        # "method.ppo_epochs": tune.loguniform(2, 16)
    }

    # Weights & Biases
    wandb.login()

    # Ray: Tune
    tune.register_trainable(wandb_project_name, soft_opt_experiment)
    tune_function(soft_opt_experiment, param_space, resources)


def single_experiment():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    param_dict = {
        "method.init_kl_coef": 0.01,
        "method.target": None,
        "method.horizon": 1000,  # default 10000
        "method.cliprange_reward": 5,  # Default 10
        "method.cliprange": 0.1,  # Default was 0.2
        "method.cliprange_value": 0.1,  # Default was 0.2
        "method.ppo_epochs": 4,  # default was 4
        "method.num_rollouts": 64,  # default was 64
        "method.lam": 0.95,  # was 0.95
        "method.scale_reward": False,  # type: ignore
        "optimizer.kwargs": {
            "lr": 1.0e-5,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
        "train.batch_size": 128,
        "train.epochs": 6000,
        "train.eval_interval": 50,
        "reward.alpha": 30,
        "reward.beta": 0.1,
        "reward.sigmoid": True,
        "reward.eps": 1e-3,

    }
    soft_opt_experiment(param_dict)


if __name__ == "__main__":
    single_experiment()
