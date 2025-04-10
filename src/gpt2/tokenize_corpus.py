import argparse
from gpt2.data.vocabulary import Vocab
from gpt2.data.tokenization import Tokenizer
import os

def add_dotted_g(tokens):
    # Prefix each non-special token with Ġ if it's at the beginning of a word
    # This mimics the GPT-2 behavior, but you may want to adjust based on your vocab
    new_tokens = []
    prev_was_space = True
    for tok in tokens:
        if tok in ['<unk>', '<s>', '</s>', '<pad>']:
            new_tokens.append(tok)
            prev_was_space = True
        elif tok in [',', '.', '!', '?', '(', ')', ':', ';', '-', '"', "'"]:
            new_tokens.append(tok)
            prev_was_space = False
        else:
            if prev_was_space:
                new_tokens.append(f'Ġ{tok}')
            else:
                new_tokens.append(tok)
            prev_was_space = False
    return new_tokens

def tokenize_file(input_path, output_path, tokenizer, add_g=False):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            tokens = tokenizer.encode(line.strip())
            if not tokens:
                continue
            if add_g:
                tokens = add_dotted_g(tokens)
            outfile.write(" ".join(tokens) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help="Path to train corpus (raw)")
    parser.add_argument('--test', type=str, required=True, help="Path to test corpus (raw)")
    parser.add_argument('--vocab', type=str, required=True, help="Path to vocab.txt")
    parser.add_argument('--output-dir', type=str, default='tokenized', help="Output directory")
    parser.add_argument('--add-g', action='store_true', help="Add dotted Ġ prefix to words")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vocab = Vocab(args.vocab)
    tokenizer = Tokenizer(vocab)

    tokenize_file(args.train, os.path.join(args.output_dir, 'corpus.train.txt'), tokenizer, add_g=args.add_g)
    tokenize_file(args.test, os.path.join(args.output_dir, 'corpus.test.txt'), tokenizer, add_g=args.add_g)

    print(f"Tokenized files saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

