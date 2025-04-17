import random
from transformers import GPT2TokenizerFast
from gpt2_py.data import GPT2BookCorpusDataset

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Dataset config
config = {
    "dataset_path": "../../datasets/bookcorpus/bookcorpus_2M.txt",
    "max_seq_len": 237,
    "pad_idx": tokenizer.pad_token_id
}

# Load dataset
dataset = GPT2BookCorpusDataset(tokenizer, config)

# Pick a random index
random_idx = random.randint(0, len(dataset) - 1)
sample = dataset[random_idx]

# Print sample details
print(f"Sample index: {random_idx}")
print("Input IDs:", sample["input_ids"])
print("Attention Mask:", sample["attention_mask"])
print("Is Next:", sample["is_next"])
print("Tokens:", sample["tokens"])
print("Decoded text:", tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
