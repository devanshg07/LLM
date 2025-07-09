import torch
from model import GPTModel
from tokenizer import Tokenizer
from config import Config

class TextGenerator:
    def __init__(self, model_path, tokenizer_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = Tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer with {self.tokenizer.vocab_size} tokens")
        
        # Create config with correct vocab_size
        self.config = Config(vocab_size=self.tokenizer.vocab_size)
        
        # Load model
        self.model = GPTModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def generate(self, prompt, max_new_tokens=None, temperature=None, top_k=None):
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
            
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Convert to tensor
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        # Generate
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
        print("\n" + "="*60)
        print("🤖 GPT Text Generator")
        print("="*60)
        print("Commands:")
        print("  - Type your prompt and press Enter to generate")
        print("  - Type 'temp=X' to change temperature (0.1-2.0)")
        print("  - Type 'tokens=X' to change max tokens (10-500)")
        print("  - Type 'topk=X' to change top-k (1-200)")
        print("  - Type 'quit' to exit")
        print("="*60)
        
        # Current generation parameters
        temp = self.config.temperature
        max_tokens = self.config.max_new_tokens
        top_k = self.config.top_k
        
        while True:
            try:
                prompt = input(f"\n📝 Prompt (temp={temp:.1f}, tokens={max_tokens}, topk={top_k}): ")
                
                if prompt.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                    
                # Handle parameter changes
                if prompt.startswith('temp='):
                    try:
                        temp = float(prompt.split('=')[1])
                        temp = max(0.1, min(2.0, temp))  # Clamp between 0.1 and 2.0
                        print(f"Temperature set to {temp:.1f}")
                        continue
                    except:
                        print("Invalid temperature. Use format: temp=0.8")
                        continue
                        
                if prompt.startswith('tokens='):
                    try:
                        max_tokens = int(prompt.split('=')[1])
                        max_tokens = max(10, min(500, max_tokens))  # Clamp between 10 and 500
                        print(f"Max tokens set to {max_tokens}")
                        continue
                    except:
                        print("Invalid token count. Use format: tokens=100")
                        continue
                        
                if prompt.startswith('topk='):
                    try:
                        top_k = int(prompt.split('=')[1])
                        top_k = max(1, min(200, top_k))  # Clamp between 1 and 200
                        print(f"Top-k set to {top_k}")
                        continue
                    except:
                        print("Invalid top-k. Use format: topk=50")
                        continue
                
                if not prompt.strip():
                    print("Please enter a prompt!")
                    continue
                
                # Generate text
                print("\n🔄 Generating...")
                generated = self.generate(prompt, max_new_tokens=max_tokens, temperature=temp, top_k=top_k)
                
                # Display result
                print(f"\n🤖 Generated text:")
                print("-" * 40)
                print(generated)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error generating text: {e}")

def main():
    model_path = 'best_model.pt'  # or 'final_model.pt'
    tokenizer_path = 'tokenizer.json'
    
    try:
        generator = TextGenerator(model_path, tokenizer_path)
        generator.interactive_generate()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please train the model first by running: python train.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
