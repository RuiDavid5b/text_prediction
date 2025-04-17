import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from random import randint, random as rand
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Initialize tokenizer and manually add PAD token
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add PAD token

# Resize model embeddings if using GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

class GPT2BookCorpusDataset(Dataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id

        # Pull dataset path and max_seq_len from config
        dataset_path = config.get("dataset_path", None)
        max_seq_len = config.get("max_seq_len", None)

        if dataset_path:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.sentences = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.sentences)} sentences from {dataset_path}")
        else:
            dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
            self.sentences = [entry["text"] for entry in dataset]
            print(f"Loaded {len(self.sentences)} sentences from HuggingFace")

        if max_seq_len is None:
            print("No max_seq_len provided, computing over entire dataset")
            self.max_seq_len = self._get_max_sequence_length(full=True)
            print(f"Determined max sequence length: {self.max_seq_len}")
        else:
            self.max_seq_len = max_seq_len
            print(f"Using provided max sequence length: {self.max_seq_len}")

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoded = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids.tolist()),
            'is_next': torch.tensor(-1)  # or remove this key if not used
        }

    def _get_sentence_pair(self, idx, next_prob=0.5):
        sent_a = self.sentences[idx]

        if rand() < next_prob and idx < len(self.sentences) - 1:
            sent_b = self.sentences[idx + 1]
            is_next = True
        else:
            rand_idx = randint(0, len(self.sentences) - 2)
            sent_b = self.sentences[rand_idx]
            is_next = False

        return sent_a, sent_b, is_next

    def _get_max_sequence_length(self, full=False):
        max_len = 0
        sample_size = len(self.sentences) - 1 if full else min(10000, len(self.sentences) - 1)
        print(f"Computing max token length ({'full dataset' if full else 'sample of 10,000'})...")

        for i in range(sample_size):
            sent_a, sent_b, _ = self._get_sentence_pair(i)
            tokens = self.tokenizer.encode(f"{sent_a} {sent_b}")
            max_len = max(max_len, len(tokens))

        return min(max_len, self.tokenizer.model_max_length)
