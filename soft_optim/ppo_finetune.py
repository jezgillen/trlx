import trlx
from trlx.data.configs import TRLConfig
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from game import TicTacToeGame


from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint

if __name__ == "__main__":
    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        return rewards

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)

    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint

    trainer = trlx.train(
        str(model_path),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"]*config.train.batch_size,
    )

    # test model output
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer.encode(game_start_text, return_tensors="pt").to('cuda')
    out = trainer.model.generate(tokens, max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    fine_tuned_model_path = Path(__file__).parent / ".checkpoints" / "fine_tuned_model"
    trainer.save(fine_tuned_model_path)