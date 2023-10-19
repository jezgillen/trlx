from typing import Any, Dict

from dict_deep import deep_get, deep_set
from trlx.data.configs import (ModelConfig, OptimizerConfig, SchedulerConfig,
                               TokenizerConfig, TrainConfig, TRLConfig)
from trlx.trainer.nn.ppo_models import PPOConfig

from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint

# TRLX PPO Method config
# See https://arxiv.org/pdf/2006.05990.pdf for good defaults
method_config = PPOConfig(
    name="ppoconfig",
    num_rollouts=64,
    chunk_size=64,
    # PPO Epochs (running the same batch multiple times in a row)
    # "Go over experience multiple times."
    ppo_epochs=6,
    init_kl_coef=0.3,
    # Fixed KL gradients & actual rather than estimate
    kl_mode="loss",  # type: ignore
    target=None,  # type: ignore
    horizon=10000,  # Not used
    # Discount factor
    # "Discount factor γ is one of the most important hyperparameters and should
    # be tuned per environment (start with γ = 0.99)"
    gamma=1,  # 1 probably makes most sense given our reward function only runs at the end
    # GAE Lam
    # "Use GAE with λ = 0.9 but neither Huber loss nor PPO-style value loss clipping"
    lam=0.95,
    cliprange_value=0.2,  # Default was 0.2
    # Clipping loss
    # Start with the clipping threshold set to 0.25 but also try lower and
    # higher values if possible. [0.1, 0.5]
    cliprange=0.2,  # Default was 0.2
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

# Tokenizer isn't set correctly by TRLX
tokenizer = TokenizerConfig(
    tokenizer_path="gpt2",
    truncation_side="right"
)

# TRLX config
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=300,
        total_steps=10000,
        batch_size=128,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        orchestrator="PPOOrchestrator",
        trainer="AcceleratePPOTrainer",
    ),
    method=method_config,
    model=ModelConfig(
        model_path=str(valid_games_fine_tuned_checkpoint),
        num_layers_unfrozen=-1
    ),
    tokenizer=tokenizer,
    optimizer=OptimizerConfig(
        name="adamw",
        # "Use Adam [8] optimizer with momentum β1 = 0.9 and a tuned learning
        # rate (0.0003 is a safe default). Linearly decaying the learning rate
        # may slightly improve performance but is of secondary importance"
        kwargs={
            "lr": 1.0e-5,
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


def default_config_override(params: Dict[str, Any]) -> TRLConfig:
    """Default config override

    Override the default config for TRLX, using a dictionary of dot-notation
    keys (e.g. {"method.init_kl_param": 1.0}).

    Args:
        params: Dot-notation keys (to specific params in the default config)

    Returns:
        TRLConfig: Config for TRLX
    """
    config = default_config.to_dict()

    for key, val in params.items():
        # Get the current value
        curr_val = deep_get(config, key)
        curr_val_type = type(curr_val)

        # If the type is int, set the new value as this
        # Useful as BayesOptSearch requires float values (so the param may be a
        # float) but some config items (e.g. batch size) require ints.
        if curr_val_type == int:
            deep_set(config, key, int(val))

        else:
            deep_set(config, key, val)

    return TRLConfig.from_dict({"tokenizer": tokenizer.__dict__, **config})
