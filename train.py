import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from model import GPTModel
from tokenizer import Tokenizer
from config import Config
import os

class DataLoader:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        
    def get_batch(self, data, batch_size):
        # Debug print
        print(f"[DEBUG] get_batch: batch_size={batch_size}, block_size={self.config.block_size}, data_len={len(data)}")
        # Generate random starting indices
        ix = torch.randint(0, len(data) - self.config.block_size, (batch_size,))
        # Debug print
        print(f"[DEBUG] get_batch: ix shape={ix.shape}, ix dtype={ix.dtype}")
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
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        logits, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
        print(f"Training on {len(train_data)} tokens, validating on {len(val_data)} tokens")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        best_val_loss = float('inf')
        for iter in range(self.config.max_iters):
            x, y = data_loader.get_batch(train_data, self.config.batch_size)
            loss = self.train_step(x, y)
            if iter % self.config.eval_interval == 0:
                val_loss = self.evaluate(data_loader, train_data, val_data)
                print(f"iter {iter}: train loss {loss:.4f}, val loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    print(f"Saved best model with val loss {val_loss:.4f}")
        print("Training completed!")
        return self.model

def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    # Prepare data and tokenizer first
    data_loader = DataLoader(None, None)
    train_data, val_data = data_loader.prepare_data('dataset/input.txt')
    vocab_size = data_loader.tokenizer.vocab_size
    print(f"[DEBUG] Vocabulary size: {vocab_size}")
    # Now create config with vocab_size
    config = Config(vocab_size=vocab_size)
    print(f"[DEBUG] Config: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}, block_size={config.block_size}, dropout={config.dropout}")
    # Update data_loader with config
    data_loader.config = config
    # Create model
    print(f"[DEBUG] Creating model with vocab_size={config.vocab_size}")
    model = GPTModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    # Create trainer
    trainer = Trainer(model, config, device='cuda' if torch.cuda.is_available() else 'cpu')
    # Train model
    trained_model = trainer.train(train_data, val_data, data_loader)
    # Save final model and tokenizer
    torch.save(trained_model.state_dict(), 'final_model.pt')
    data_loader.tokenizer.save('tokenizer.json')
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    main()
