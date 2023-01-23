import trlx
from trlx.data.configs import TRLConfig
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from game import TicTacToeGame
import soft_optim.quantilizer as quantilizer
import numpy as np
from typing import List



from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint, infer_game

def no_soft_opt_experiment():
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

    fine_tuned_model_path = Path(__file__).parent / ".checkpoints" / "no_soft_opt_model"
    trainer.save(fine_tuned_model_path)


def soft_opt_experiment():
    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    # get samples for gen error calculation
    samples = infer_game(model, tokenizer, num_samples=200)    
    proxy_rewards :List[float] = []
    human_rewards :List[float] = []
    g_proxy = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    g_human = TicTacToeGame()
    for s in samples:
        proxy_rewards.append(g_proxy.evaluate_game_string(s))
        human_rewards.append(g_human.evaluate_game_string(s))
    proxy_rewards_arr = np.array(proxy_rewards)
    human_rewards_arr = np.array(human_rewards)

    
    # get generalization error
    eps = 0.05 # <5% chance of bound being exceeded
    bound = quantilizer.empirical_error_bound(proxy_rewards_arr, human_rewards_arr, eps)
    # work out proxy reward cutoff
    cutoff = quantilizer.get_proxy_value_cutoff(bound, len(samples), model, tokenizer)

    print(bound)
    print(cutoff)

    #def loglikelihood_approx(rewards, cutoff):
    #    return np.log10(1/(1+np.exp(-(rewards-cutoff))))

    def loglikelihood_approx(rewards, cutoff):
        return np.log10((rewards > cutoff)+1e-8)

    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        rewards_arr = np.array(rewards)
        return loglikelihood_approx(rewards_arr, cutoff)

    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)

    # custom config options for this experiment
    config.method.target = None # Set to constant KL penalty
    config.method.init_kl_coef = 1.0 # set weight of KL penalty to 1

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

    fine_tuned_model_path = Path(__file__).parent / ".checkpoints" / "soft_opt_model"
    trainer.save(fine_tuned_model_path)


if __name__ == "__main__":
    #no_soft_opt_experiment()

    soft_opt_experiment()