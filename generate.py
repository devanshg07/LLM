import torch
from model import GPTModel
from tokenizer import Tokenizer
from config import Config

class TextGenerator:
    def __init__(self, model_path, tokenizer_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load tokenizer
        self.tokenizer = Tokenizer.load(tokenizer_path)
        # Create config with correct vocab_size
        self.config = Config(vocab_size=self.tokenizer.vocab_size)
        # Load model
        self.model = GPTModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    def generate(self, prompt, max_new_tokens=None, temperature=None, top_k=None):
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                x, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist(), remove_special_tokens=True)
        return generated_text
    def interactive_generate(self):
        print("Interactive text generation (type 'quit' to exit)")
        print("=" * 50)
        while True:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() == 'quit':
                break
            try:
                generated = self.generate(prompt)
                print(f"\nGenerated text:\n{generated}")
            except Exception as e:
                print(f"Error generating text: {e}")

def main():
    model_path = 'best_model.pt'  # or 'final_model.pt'
    tokenizer_path = 'tokenizer.json'
    try:
        generator = TextGenerator(model_path, tokenizer_path)
        generator.interactive_generate()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python train.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
