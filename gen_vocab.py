from transformers import GPT2Tokenizer

# Load pretrained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Extract vocabulary (this is a dict {token: id})
vocab = tokenizer.get_vocab()

# Sort by index to preserve order
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

# Write to vocab.txt
with open("../datasets/vocabulary/vocab.txt", "w", encoding="utf-8") as f:
    for token, _ in sorted_vocab:
        f.write(token + "\n")

