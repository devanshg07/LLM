from tokenizer import Tokenizer

def test_new_tokenizer():
    print("Testing new tokenizer...")
    
    # Load the new tokenizer
    tokenizer = Tokenizer.load('tokenizer.json')
    
    # Test encoding "dad"
    test_text = "dad"
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, remove_special_tokens=True)
    
    print(f"Text: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Check if 'd' and 'a' exist in vocabulary
    print(f"\nChecking characters:")
    print(f"'d' in vocab: {'d' in tokenizer.vocab}")
    print(f"'a' in vocab: {'a' in tokenizer.vocab}")
    
    if 'd' in tokenizer.vocab:
        print(f"'d' -> {tokenizer.vocab['d']}")
    if 'a' in tokenizer.vocab:
        print(f"'a' -> {tokenizer.vocab['a']}")

if __name__ == "__main__":
    test_new_tokenizer() 