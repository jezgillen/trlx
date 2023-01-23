from typing import List

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_optim.fine_tune import infer_game
from soft_optim.game import TicTacToeGame


def empirical_error_bound(
    proxy_reward: np.ndarray,  
    human_evaluated_reward: np.ndarray,
    epsilon: float = 0.001
    ) -> float:
    """Empirical error bound calculation
    
    Calculates the error bound such that for further samples, there is at most a
    small (epsilon) probability that the expected true reward would be less than the
    mean proxy reward minus the bound. This gives us a worst-case confidence
    interval, when we don't know how the errors are distributed.
    
    Uses Hoeffding's inequality to calculate the lower error bound
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
    
    As an example, assume we've run a model many times to get sample games. For
    each sample, we've calculated both the proxy_reward (i.e. from the RL reward
    function) and the human evaluate reward (e.g. from human feedback). This
    function can then be used to calculate the "error bound" such that there is
    at most a small (epsilon) probability that any new sample's human evaluated
    reward would be lower than proxy_reward minus error_bound.
    
    Args:
        proxy_reward: Sample rewards in [0,1]
        human_evaluated_reward: Human evaluated rewards in [0,1]
        epsilon: The probability of the proxy reward deviating below the error bound.

    Returns:
        float: Error bound (such that the probability that human evaluated
        reward would be lower than the proxy reward minus this bound, is at most
        epsilon).
    """
    proxy_reward = np.array(proxy_reward)
    human_evaluated_reward = np.array(human_evaluated_reward)
    # Expected difference between the two rewards
    expected_difference: float = np.abs(proxy_reward - human_evaluated_reward).mean()
    
    number_samples: int = len(proxy_reward)
    
    # The confidence bound gets smaller when epsilon is larger, and also gets
    # smaller when the number of samples is larger.
    return expected_difference + np.sqrt(-np.log(epsilon) / (2 * number_samples))
    

def get_proxy_value_cutoff(
    error_bound: float, 
    number_samples: int,
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer
    ) -> float:
    """Quantilizer q-value cutoff
    
    Get the q-value cut-off point, such that by randomly selecting from the top q% of
    prior policies (as measured by proxy reward), we maximize the lower bound of
    the true reward.

    Args:
        error_bound: Empirically calculated error bound, i.e. the bound
        calculated whereby there is at most a small (epsilon) probability that
        the true reward will be less than the proxy reward minus this error
        bound.
        number_samples: Number of times to sample policies from the model

    Returns:
        float: Proxy value cut-off (q)
    """
    proxy_rewards: List[float] = []
    
    # Generate new samples
    g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    game_text_list: List[str] = infer_game(model, tokenizer, num_samples=number_samples)
    for game_string in game_text_list:
        proxy_reward = g.evaluate_game_string(game_string)
        proxy_rewards.append(proxy_reward)
    
    proxy_rewards_ordered = sorted(proxy_rewards)
    
    # Estimate the q-value (cutoff for the proxy reward)
    lower_bounds: List[float] = []
    
    for i in range(0, len(proxy_rewards)):
        q = (len(proxy_rewards) - i) / len(proxy_rewards)
        estimated_policy_distribution_lower_bound: float = float(np.mean(proxy_rewards_ordered[i:]) - 1/q * error_bound)
        lower_bounds.append(estimated_policy_distribution_lower_bound)
        
    estimated_lower_bound_index = np.argmax(lower_bounds)
    
    return proxy_rewards_ordered[estimated_lower_bound_index]
    
    