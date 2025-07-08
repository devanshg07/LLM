
class Config:
    # Model architecture
    vocab_size = None  # Will be set by tokenizer
    n_embd = 128      # Embedding dimension
    n_layer = 2       # Number of transformer layers
    n_head = 2        # Number of attention heads
    block_size = 64   # Context length (reduced for memory)
    dropout = 0.1     # Dropout rate
    
    # Training
    batch_size = 8    # Small batch size for CPU
    learning_rate = 1e-3
    max_iters = 3000  # More iterations for larger dataset
    eval_interval = 100
    eval_iters = 50
    
    # Data
    train_split = 0.9  # 90% for training, 10% for validation
    
    # Generation
    max_new_tokens = 100
    temperature = 0.8
    top_k = 200 