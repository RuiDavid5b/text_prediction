from transformers import GPT2TokenizerFast
from gpt2_py.data import GPT2BookCorpusDataset

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

config = {
    "model": {"max_length": 512},
}

# Replace with your local text file path if needed
dataset_path = "../../datasets/bookcorpus/bookcorpus_2M.txt"  # or set to None

#dataset = GPT2BookCorpusDataset(tokenizer, config, dataset_path=dataset_path)
dataset = GPT2BookCorpusDataset(tokenizer, config, dataset_path=dataset_path, max_seq_len=237)
print("Dataset length:", len(dataset))

sample = dataset[0]
print("Input IDs:", sample["input_ids"])
print("Attention Mask:", sample["attention_mask"])
print("Is Next:", sample["is_next"])
print("Tokens:", sample["tokens"])

print("Decoded text:", tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
