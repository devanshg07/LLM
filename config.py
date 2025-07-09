
class Config:
    def __init__(self, vocab_size=None):
        # Model architecture - Larger model for better performance
        self.vocab_size = vocab_size  # Must be set before model creation!
        self.n_embd = 384      # Increased from 128
        self.n_layer = 6       # Increased from 2
        self.n_head = 6        # Increased from 2
        self.block_size = 128  # Increased from 64
        self.dropout = 0.1

        # Training - More iterations and larger batch size
        self.batch_size = 16   # Increased from 8
        self.learning_rate = 3e-4  # Slightly lower for stability
        self.max_iters = 100000  # Increased to 100k iterations
        self.eval_interval = 100
        self.eval_iters = 50

        # Data
        self.train_split = 0.9

        # Generation - Better parameters
        self.max_new_tokens = 200  # Increased from 100
        self.temperature = 0.9     # Slightly higher for more creativity
        self.top_k = 200 