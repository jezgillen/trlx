import trlx
from trlx.data.configs import TRLConfig
import math
from pathlib import Path
from transformers import AutoModelWithLMHead


from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint

if __name__ == "__main__":
    def reward_fn(samples, prompts=None, outputs=None):
        return [s.count('o') for s in samples]



    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)

    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint
    
    trainer = trlx.train(
        str(model_path),
        reward_fn=reward_fn,
        config=config,
    )

    fine_tuned_model_path = Path(__file__).parent / ".checkpoints" / "fine_tuned_model"
    trainer.save(fine_tuned_model_path)