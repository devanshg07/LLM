import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from model import GPTModel
from tokenizer import Tokenizer
from config import Config
import os
import traceback
import time

class DataLoader:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        
    def get_batch(self, data, batch_size):
        # Generate random starting indices
        ix = torch.randint(0, len(data) - self.config.block_size, (batch_size,))
        # Create input and target sequences
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        return x, y
    
    def prepare_data(self, text_file_path):
        # Load text
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = Tokenizer(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        data = torch.tensor(tokens, dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data

class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Setup optimizer with weight decay
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_iters)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        logits, loss = self.model(x, y)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, data_loader, train_data, val_data):
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                x, y = data_loader.get_batch(val_data, self.config.batch_size)
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, y)
                losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses)
    
    def train(self, train_data, val_data, data_loader):
        print(f"Training on {len(train_data):,} tokens, validating on {len(val_data):,} tokens")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Block size: {self.config.block_size}, Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for iter in range(self.config.max_iters):
            # Get batch
            x, y = data_loader.get_batch(train_data, self.config.batch_size)
            
            # Training step
            loss = self.train_step(x, y)
            self.train_losses.append(loss)
            
            # Evaluation
            if iter % self.config.eval_interval == 0:
                val_loss = self.evaluate(data_loader, train_data, val_data)
                self.val_losses.append(val_loss)
                
                # Calculate time and tokens per second
                elapsed = time.time() - start_time
                tokens_per_sec = (iter + 1) * self.config.batch_size * self.config.block_size / elapsed
                
                print(f"iter {iter:4d}: train loss {loss:.4f}, val loss {val_loss:.4f}, "
                      f"lr {self.scheduler.get_last_lr()[0]:.2e}, "
                      f"tokens/sec {tokens_per_sec:.0f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    print(f"  -> Saved best model with val loss {val_loss:.4f}")
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        return self.model

def main():
    print("Starting GPT training...")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Prepare data and tokenizer first
    print("Loading and preparing data...")
    data_loader = DataLoader(None, None)
    train_data, val_data = data_loader.prepare_data('dataset/input.txt')
    vocab_size = data_loader.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Now create config with vocab_size
    config = Config(vocab_size=vocab_size)
    data_loader.config = config
    
    # Create model
    print("Creating model...")
    model = GPTModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = Trainer(model, config, device)
    
    # Train model
    print("Starting training...")
    trained_model = trainer.train(train_data, val_data, data_loader)
    
    # Save final model and tokenizer
    torch.save(trained_model.state_dict(), 'final_model.pt')
    data_loader.tokenizer.save('tokenizer.json')
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== EXCEPTION CAUGHT ===")
        traceback.print_exc()
        raise
