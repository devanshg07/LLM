
class Config:
    def __init__(self, vocab_size=None):
        # Model architecture
        self.vocab_size = vocab_size  # Must be set before model creation!
        self.n_embd = 128
        self.n_layer = 2
        self.n_head = 2
        self.block_size = 64
        self.dropout = 0.1

        # Training
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.max_iters = 3000
        self.eval_interval = 100
        self.eval_iters = 50

        # Data
        self.train_split = 0.9

        # Generation
        self.max_new_tokens = 100
        self.temperature = 0.8
        self.top_k = 200 