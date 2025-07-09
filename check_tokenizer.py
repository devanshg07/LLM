import json
from tokenizer import Tokenizer

def check_tokenizer():
    print("=== Tokenizer Diagnostic ===")
    
    try:
        # Load the tokenizer
        tokenizer = Tokenizer.load('tokenizer.json')
        
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Vocabulary items: {list(tokenizer.vocab.items())}")
        print(f"Index to token mappings: {tokenizer.idx_to_token}")
        
        # Check if common characters exist
        test_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        print("\nChecking for common characters:")
        for char in test_chars:
            if char in tokenizer.vocab:
                print(f"  '{char}' -> {tokenizer.vocab[char]}")
            else:
                print(f"  '{char}' -> NOT FOUND")
        
        # Test encoding
        test_text = "hello world"
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, remove_special_tokens=True)
        print(f"\nTest encoding:")
        print(f"  Text: '{test_text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_tokenizer() 