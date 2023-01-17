import wandb
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments )
from soft_optim.game_generator import generate_dataset
from pathlib import Path

def create_dataset(tokenizer: AutoTokenizer, number_games: int = 10) -> Dataset:
    """Create the dataset
    
    This is a collection of full game prompts (tokenized).

    Args:
        tokenizer: Tokenizer
        number_games: Number of games

    Returns:
        Dataset: Full game prompts dataset
    """
    # Create the dataset from a list of game strings
    list_of_game_strings = generate_dataset(number_games)
    dataset = Dataset.from_dict({"text":list_of_game_strings})
    
    # Tokenize the text prompts (creates "input_ids" property for each dataset item)
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    
    # Set the labels to be the same as the input IDs
    dataset = dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    
    return dataset


valid_games_fine_tuned_checkpoint = Path(__file__).parent.parent / ".checkpoints" / "valid_games_fine_tune"


def fine_tune(model_name: str = "gpt2", log_weights_and_biases: bool = False) -> AutoModelForCausalLM:
    """Fine tune a language model on the games dataset
    
    This is so that our model reliably outputs allowed game moves.
    """
    # Create tokenized datasets (train and eval)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = create_dataset(tokenizer, 5000)
    eval_dataset = create_dataset(tokenizer, 50)
   
    # Initialise Weights & Biases
    if log_weights_and_biases:
        wandb.login()
        wandb.init(project="soft_optim")

    # Create the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir=".checkpoints", 
        evaluation_strategy="epoch",
        num_train_epochs=1,
    )
    
    # Fine tune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        
    )
    trainer.train()


    # print model output
    out = model.generate(max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Save the model
    model.save_pretrained(valid_games_fine_tuned_checkpoint)

    return model


def infer_game(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    """Infer a full game from just the start text

    Args:
        model: Pretrained model
        tokenizer: Tokenizer

    Returns:
        str: Inferred game board states only (no start or end text)
    """
    # Create the start text
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer.encode(game_start_text, return_tensors="pt")
    
    # Generate the game
    output_tokens = model.generate(tokens, max_length=1000)
    full_game: str = tokenizer.decode(output_tokens[0])
    
    # Get just the board states
    game_end_text = "\n<|endoftext|>"
    full_game_after_start = full_game[len(game_start_text):]
    full_game_before_end_text = full_game_after_start.split(game_end_text)[0]
    
    return full_game_before_end_text


if __name__ == "__main__":
    fine_tune(log_weights_and_biases = True)
