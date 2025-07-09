import torch
from model import GPTModel
from tokenizer import Tokenizer
from config import Config

def test_model_creation():
    print("Testing model creation...")
    
    # Load some text to create tokenizer
    with open('dataset/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()[:1000]  # Just first 1000 chars for testing
    
    # Create tokenizer
    tokenizer = Tokenizer(text)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create config and set vocab_size
    config = Config()
    config.vocab_size = tokenizer.vocab_size
    print(f"Config vocab_size: {config.vocab_size}")
    
    # Try to create model
    try:
        model = GPTModel(config)
        print("Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        test_input = torch.randint(0, config.vocab_size, (1, 10))
        logits, loss = model(test_input, test_input)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_creation() 