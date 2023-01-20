
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_optim.fine_tune import (create_dataset, fine_tune, infer_game,
                                  valid_games_fine_tuned_checkpoint)
from soft_optim.game import TicTacToeGame


class TestCreateDataset:
    def test_tokenizes_text(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        text = first_example["text"]
        input_ids = first_example["input_ids"]
        expected_input_ids = tokenizer.encode(text)
        assert input_ids == expected_input_ids

    def test_adds_labels(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        input_ids = first_example["input_ids"]
        labels = first_example["labels"]
        assert input_ids == labels


class TestCheckModelOutputsValidGame:
    
    def test_fine_tuned_gpt(self):
        # Don't run if CUDA isn't available
        if not torch.cuda.is_available():
            return
        
        # Run the model if it hasn't already been run
        if not valid_games_fine_tuned_checkpoint.exists():
            fine_tune(log_weights_and_biases=False)

        # Load the fine-tuned model
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(valid_games_fine_tuned_checkpoint)

        # Infer the game
        full_game:str = infer_game(model, tokenizer)[0]

        # Check it is valid
        print("game")
        print(full_game)
        game = TicTacToeGame()
        res = game.evaluate_game_string(full_game)
        assert type(res) == float


