from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from game import TicTacToeGame

import trlx
from game import TicTacToeGame
from transformers import AutoTokenizer
from trlx.data.configs import TRLConfig

import wandb
from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint


def proxy_reward(
    samples: List[str], 
    _prompts: Optional[List[str]] = None, 
    _outputs: Optional[List[str]] = None
    ) -> List[float]:
    """Proxy reward
    
    Args:
        samples: Batch of responses
        prompts: Batch of prompts
        outputs: Batch of outputs

    Returns:
        List[float]: List of rewards
    """    
    rewards = []
    
    for s in samples:
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        rewards.append(g.evaluate_game_string(s))

    return rewards


def metrics(
    samples: List[str], 
    _prompts: Optional[List[str]] = None, 
    _outputs: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
    """Metrics
    
    Args:
        samples: Batch of responses
        prompts: Batch of prompts
        outputs: Batch of outputs

    Returns:
        Dict[str, List[float]]: Dict of metrics, where the key is the metric
        name and the value is a list of metric values (one for each item in the
        batch). 
    """    
    true_rewards = []
    
    for s in samples:
        g = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        true_rewards.append(g.evaluate_game_string(s))
        
    return {"true_reward": true_rewards}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.eos_token = "<|endoftext|>"
    
    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)
    config.method.gen_kwargs["eos_token_id"] = int(tokenizer.encode(tokenizer.eos_token)[0])
    
    # Configure W&B
    wandb.login()
    wandb.init(project="soft_optim_rl")


    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint

    trainer = trlx.train(
        str(model_path),
        reward_fn=proxy_reward,
        config=config,
        prompts=["Let's play Tic Tac Toe:"]*config.train.batch_size,
        metric_fn=metrics
    )

    # test model output
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer.encode(game_start_text, return_tensors="pt").to('cuda')
    out = trainer.model.generate(tokens, max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    fine_tuned_model_path = Path(__file__).parent / ".checkpoints" / "fine_tuned_model"
    trainer.save(fine_tuned_model_path)
    