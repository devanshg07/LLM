from tokenizer import Tokenizer

def rebuild_tokenizer():
    print("Rebuilding tokenizer from dataset/input.txt ...")
    with open('dataset/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = Tokenizer(text)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Sample vocab: {list(tokenizer.vocab.items())[:20]}")
    tokenizer.save('tokenizer.json')
    print("Tokenizer saved as tokenizer.json")

if __name__ == "__main__":
    rebuild_tokenizer() 