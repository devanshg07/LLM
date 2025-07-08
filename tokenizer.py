class Tokenizer:
    def __init__(self, text=None):
        """
        Initialize tokenizer with optional text to build vocabulary.
        If text is provided, builds vocabulary from unique characters.
        """
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        # Build vocabulary
        if text is not None:
            self.build_vocab(text)
        else:
            # Default minimal vocabulary
            self.vocab = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
            self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, text):
        """Build vocabulary from text data."""
        # Get unique characters
        chars = sorted(list(set(text)))
        
        # Create vocabulary with special tokens first
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1, 
            self.bos_token: 2,
            self.eos_token: 3
        }
        
        # Add character tokens
        for i, char in enumerate(chars):
            self.vocab[char] = i + 4  # Start after special tokens
            
        # Create reverse mapping
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs.
        
        Args:
            text (str): Input text to encode
            add_special_tokens (bool): Whether to add BOS/EOS tokens
            
        Returns:
            list: List of token IDs
        """
        if add_special_tokens:
            text = self.bos_token + text + self.eos_token
            
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab[self.unk_token])
        
        return tokens
    
    def decode(self, token_ids, remove_special_tokens=True):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (list): List of token IDs
            remove_special_tokens (bool): Whether to remove special tokens
            
        Returns:
            str: Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx_to_token:
                token = self.idx_to_token[token_id]
                if remove_special_tokens and token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                if not remove_special_tokens:
                    tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def encode_batch(self, texts, add_special_tokens=True, pad_to_length=None):
        """
        Encode a batch of texts.
        
        Args:
            texts (list): List of text strings
            add_special_tokens (bool): Whether to add special tokens
            pad_to_length (int, optional): Length to pad sequences to
            
        Returns:
            list: List of encoded sequences
        """
        encoded = [self.encode(text, add_special_tokens) for text in texts]
        
        if pad_to_length is not None:
            encoded = self.pad_sequences(encoded, pad_to_length)
            
        return encoded
    
    def pad_sequences(self, sequences, max_length):
        """
        Pad sequences to the same length.
        
        Args:
            sequences (list): List of token ID sequences
            max_length (int): Target length for all sequences
            
        Returns:
            list: Padded sequences
        """
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with pad token
                padded_seq = seq + [self.vocab[self.pad_token]] * (max_length - len(seq))
            else:
                # Truncate if too long
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
        return padded
    
    @property
    def vocab_size(self):
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath):
        """Save tokenizer to file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'idx_to_token': self.idx_to_token
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file."""
        import json
        tokenizer = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizer.vocab = data['vocab']
            tokenizer.idx_to_token = data['idx_to_token']
        return tokenizer
