from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import ray
import trlx
from game import TicTacToeGame
from ray import tune
from ray.tune.logger import CSVLoggerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trlx.data.configs import TRLConfig, ModelConfig, TrainConfig, TokenizerConfig, OptimizerConfig, SchedulerConfig
from trlx.ray_tune import get_param_space
from trlx.trainer.nn.ppo_models import PPOConfig
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune import CLIReporter
from ray.air.integrations.wandb import WandbLoggerCallback

import soft_optim.quantilizer as quantilizer
import wandb
from soft_optim.fine_tune import infer_game, valid_games_fine_tuned_checkpoint
from soft_optim.metrics import metrics


wandb_project_name = "soft_optim"


def get_cutoff() -> float:
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
    cutoff = quantilizer.get_proxy_value_cutoff(
        bound, len(samples), model, tokenizer)

    return cutoff


def loglikelihood_approx(rewards, cutoff):
    alpha = 30.0  # hyperparameter determining sharpness of cutoff
    beta = 1  # 0.3
    return np.log(1 / ((1 + np.exp(-alpha * (rewards - cutoff))**(1 / beta))))
    # return np.log10((rewards > cutoff)+1e-8)


# TRLX PPO Method config
# See https://arxiv.org/pdf/2006.05990.pdf for good defaults
method_config = PPOConfig(
    name="ppoconfig",
    num_rollouts=16,
    chunk_size=16,
    # PPO Epochs (running the same batch multiple times in a row)
    # "Go over experience multiple times."
    ppo_epochs=6,
    kl_mode="loss",
    init_kl_coef=0.1,
    target=None,  # type: ignore
    horizon=10000,  # Not used
    # Discount factor
    # "Discount factor γ is one of the most important hyperparameters and should
    # be tuned per environment (start with γ = 0.99)"
    gamma=1,  # 1 probably makes most sense given our reward function only runs at the end
    # GAE Lam
    # "Use GAE with λ = 0.9 but neither Huber loss nor PPO-style value loss clipping"
    lam=0.9,
    cliprange_value=0.2,  # Default was 0.2
    # Clipping loss
    # Start with the clipping threshold set to 0.25 but also try lower and
    # higher values if possible. [0.1, 0.5]
    cliprange=0.25,  # Default was 0.2
    vf_coef=1,
    scale_reward=False,  # type: ignore
    ref_mean=None,
    ref_std=None,
    cliprange_reward=10,
    # HuggingFace Generate Parameters
    # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
    gen_kwargs={
        "max_new_tokens": 130,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": True}
)

# TRLX config
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=300,
        total_steps=10000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        orchestrator="PPOOrchestrator",
        trainer="AcceleratePPOTrainer",
        # tracker="wandb"
    ),
    method=method_config,
    model=ModelConfig(
        model_path="lvwerra/gpt2-imdb",
        num_layers_unfrozen=-1
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="gpt2",
        truncation_side="right"
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        # "Use Adam [8] optimizer with momentum β1 = 0.9 and a tuned learning
        # rate (0.0003 is a safe default). Linearly decaying the learning rate
        # may slightly improve performance but is of secondary importance"
        kwargs={
            "lr": 3.0e-4,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        }
    ),
    scheduler=SchedulerConfig(
        "cosine_annealing",
        kwargs={
            "T_max": 10000,  # train.total_steps
            "eta_min": 1.0e-4
        }
    )
)


def soft_opt_experiment(params: Dict[str, float]) -> None:
    """Soft optimization experiment

    Args:
        params: Parameters from Ray
    """
    # Cutoff
    cutoff = get_cutoff()

    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        rewards_arr = np.array(rewards)
        return loglikelihood_approx(rewards_arr, cutoff)

    # Config
    config = default_config
    config.method.gamma = params["gamma"]  # type: ignore
    config.method.init_kl_coef = params["init_kl_coef"] # type: ignore
    config.optimizer.kwargs["lr"] = params["lr"]  # type: ignore
    # Float from tuner so must be rounded
    config.train.batch_size = int(params["batch_size"])
    config.method.ppo_epochs = int(params["ppo_epochs"])  # type: ignore

    # Weights & Biases
    # wandb.init(
    #     project=wandb_project_name,
    #     config=params,
    #     name="".join([f"{k}={v}" for k, v in params.items()]),
    #     reinit=True,)

    trainer = trlx.train(
        str(valid_games_fine_tuned_checkpoint),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
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
        # https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
        search_alg=BayesOptSearch(),
        # scheduler=ASHAScheduler(metric="objective", mode="max"))
        num_samples=-1,  # Keep sampling forever
        max_concurrent_trials=4
    )

    # Set the metrics to report to the CLI
    reporter = CLIReporter(
        max_progress_rows=10,
        metric_columns=[
            "metrics/true_reward",
            "returns/mean",
            "metrics/is_valid"])

    tuner = tune.Tuner(
        tune.with_resources(train_function, resources=resources),
        param_space=param_space,
        tune_config=tune_config,
        run_config=ray.air.RunConfig(
            local_dir="ray_results",  # Needed for wandb
            callbacks=[
                # CSVLoggerCallback(),
                WandbLoggerCallback(project=wandb_project_name)
            ],
            # log_to_file=True, # Needed
            progress_reporter=reporter,
        ),
    )

    results = tuner.fit()

    print("Best hyper-parameters found were: ",
          results.get_best_result().config)


if __name__ == "__main__":
    # Ray: Resources per hyper parameter experiment (i.e. if you want 8
    # runs, you need 8x this number of resources)
    resources: Dict[str, float] = {
        "cpu": 1,
        "gpu": 1,
    }

    # Ray: Param config
    # Good choices from https://arxiv.org/pdf/2006.05990.pdf (in comments
    # below). Note if you add more they must also be set in the
    # soft_opt_experiment function
    param_space: Dict = {
        "lr": tune.loguniform(1e-4, 1e-7),
        "gamma": tune.loguniform(0.95, 1.0),
        # Float to work with search (rounded later)
        "init_kl_coef": tune.loguniform(1e-3, 1.0),
        "batch_size": tune.loguniform(4, 8),
        "ppo_epochs": tune.loguniform(2, 16)
    }

    # Weights & Biases
    wandb.login()

    # Ray: Tune
    tune.register_trainable(wandb_project_name, soft_opt_experiment)
    tune_function(soft_opt_experiment, param_space, resources)
