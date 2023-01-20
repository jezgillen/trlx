from typing import List, Callable
import numpy as np
from soft_optim.fine_tune import infer_game
from game import TicTacToeGame

def empirical_error_bound(
    proxy_reward: np.array,  
    human_evaluated_reward: np.array,
    epsilon: float) -> float:
    """Empirical error bound

    Uses Hoeffding's inequality (one-sided bound)
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    Args:
        proxy_reward: The reward function
        human_evaluated_reward: Human evaluation of each game, as e.g 1
        if it didn't break the rules and won, 0 if it did break the rules or
        didn't win.
        epsilon: The probability of the error bound being exceeded.

    Returns:
        float: Error bound (expected difference between human and proxy reward
        evaluations). 
    """
    # Expected difference between the two rewards
    expected_difference: float = np.abs(proxy_reward - human_evaluated_reward).mean()
    
    number_samples = len(proxy_reward)
    
    return expected_difference + np.sqrt(-np.log(epsilon) / (2 * number_samples))
    

def get_proxy_value_cutoff(error_bound: float, number_samples: int) -> float:
    """Get the proxy value cutoff
    
    
    """
    proxy_rewards: List[float] = []
    
    # Generate new samples
    for _game in range(number_samples):
        game_text: str = infer_game()
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        proxy_reward = g.evaluate_game_string(str)
        proxy_rewards.append(proxy_reward)
    
    proxy_rewards_ordered = sorted(proxy_rewards)
    
    # Estimate the q-value (cutoff for the proxy reward)
    lower_bounds = []
    for i in range(0, len(proxy_rewards)):
        q = (len(proxy_rewards) - i) / len(proxy_rewards)
        estimated_policy_distribution_lower_bound: float = np.mean(proxy_rewards_ordered[i:]) - 1/q * error_bound
        lower_bounds.append(estimated_policy_distribution_lower_bound)
        
    estimated_lower_bound = np.max(lower_bounds)
    estimated_lower_bound_index = proxy_rewards_ordered.index(estimated_lower_bound)
    
    return proxy_rewards_ordered[estimated_lower_bound_index]
    
    