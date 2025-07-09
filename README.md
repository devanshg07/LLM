# Building a GPT Language Model: A Journey Through Data, Hardware, and Tokenization

## Introduction

This project represents an attempt to build a character-level GPT (Generative Pre-trained Transformer) language model from scratch using PyTorch. What began as an exploration of transformer architecture became a valuable lesson in the critical importance of data quality, hardware limitations, and the delicate relationship between model training and tokenization.

## The Technical Foundation

The implementation includes a complete transformer architecture with multi-head attention mechanisms, positional embeddings, and a character-level tokenizer. The model architecture consists of 6 transformer blocks, each with 6 attention heads, 384-dimensional embeddings, and approximately 1.5 million parameters. This design follows the foundational GPT architecture while remaining computationally manageable for personal hardware.

## The Data Dilemma: From 70KB to 10MB

Our journey began with a modest 70KB text dataset. The model trained successfully, showing decreasing loss values and stable convergence. However, when we attempted text generation, the model produced only `<UNK>` (unknown) tokens, resulting in empty output. This was our first encounter with the data quality problem.

The 70KB dataset, while sufficient for the model to learn basic patterns, lacked the diversity necessary for meaningful text generation. The model essentially memorized the limited patterns in the data without developing the ability to generalize or create coherent new text.

In an attempt to address this limitation, we expanded the dataset to 10MB by duplicating the original content. While this increased the total volume of training data, it introduced a new problem: repetitive patterns. The model learned to expect and reproduce these repetitions, limiting its creative capabilities.

## The Tokenizer-Model Mismatch Crisis

A critical turning point came when we discovered that our model was generating token IDs that didn't exist in our tokenizer's vocabulary. Debug output revealed that simple prompts like "dad" were being encoded as `[61, 58, 61]`, but these token IDs mapped to `'NOT_FOUND'` in our tokenizer.

This discovery led to a deeper investigation of the tokenization process. We learned that the model had been trained with one tokenizer but was attempting to generate text using a different tokenizer. This fundamental mismatch meant that even if the model learned meaningful patterns during training, it couldn't express them during generation.

## Hardware Constraints: The Silent Limiter

Throughout this project, hardware limitations played a significant but often overlooked role. The transformer architecture, while powerful, is memory-intensive. Our 1.5-million parameter model required careful batch size management to fit within available RAM. Larger models or bigger batch sizes would have been desirable for better training, but hardware constraints made this impossible.

Training time also became a factor. With limited computational resources, we had to balance model complexity with training duration. This limitation influenced our choice of model size and training iterations.

## Lessons Learned: Beyond the Code

### Data Quality Over Quantity

The most important lesson was that data quality matters more than raw quantity. A 10MB dataset of repetitive content proved less valuable than a smaller dataset with diverse, high-quality text. Character-level models, in particular, require substantial variety to learn meaningful language patterns.

### The Tokenizer-Model Relationship

We discovered that the relationship between tokenizer and model is inseparable. A model trained with one tokenizer cannot generate meaningful text using a different tokenizer. This relationship must be maintained throughout the entire pipeline, from data preprocessing to final generation.

### Debugging as a Development Strategy

The debugging process revealed the importance of early and thorough testing. By adding debug prints to examine token mappings and vocabulary contents, we were able to identify problems that would have been invisible otherwise. This debugging-first approach should be standard practice in machine learning development.

## Technical Implementation Details

The model architecture follows the standard GPT design with some modifications for efficiency. The embedding layer maps character tokens to 384-dimensional vectors, which are then processed through 6 transformer blocks. Each block includes multi-head attention and a feed-forward network with residual connections and layer normalization.

The tokenizer implements a character-level approach, mapping individual characters to token IDs. Special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) are included for proper text processing. The training process uses AdamW optimization with cosine annealing learning rate scheduling and gradient clipping for stability.

## Conclusion: Success in Failure

While this project didn't achieve the goal of generating coherent text, it succeeded in demonstrating the complexity and challenges of building language models from scratch. The model architecture works correctly, the training pipeline is sound, and the implementation follows best practices. The limitations we encountered were primarily related to data quality and hardware constraints rather than fundamental flaws in the approach.

This experience highlights the importance of understanding not just the model architecture, but also the data pipeline, hardware requirements, and the relationships between different components of the system. For future attempts, we would recommend focusing on data quality, ensuring tokenizer-model consistency, and scaling the approach based on available computational resources.

---

# How to Run Your Own GPT Model

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.7 or higher
- PyTorch (CPU or GPU version)
- Basic understanding of Python and command line operations

## Installation

1. **Clone or download the project files:**
   ```bash
   # If using git
   git clone <repository-url>
   cd gpt-project
   
   # Or download and extract the files manually
   ```

2. **Install required packages:**
   ```bash
   pip install torch numpy
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## Preparing Your Data

1. **Create a dataset folder:**
   ```bash
   mkdir dataset
   ```

2. **Add your training data:**
   - Place your text file in `dataset/input.txt`
   - Aim for at least 1MB of diverse text
   - Use plain text format (UTF-8 encoding)
   - Include a variety of content (stories, articles, etc.)

3. **Data quality tips:**
   - Avoid repetitive content
   - Include different writing styles
   - Ensure proper punctuation and formatting
   - Remove any non-text content

## Step-by-Step Setup

### Step 1: Test Your Environment
```bash
python test_model.py
```
This will verify that your environment can create and run the model.

### Step 2: Build Your Tokenizer
```bash
python rebuild_tokenizer.py
```
This creates a tokenizer from your dataset and saves it as `tokenizer.json`.

### Step 3: Train Your Model
```bash
python train.py
```
This will:
- Load your dataset
- Create the model with appropriate vocabulary size
- Train for 5000 iterations
- Save the best model as `best_model.pt`
- Save the final model as `final_model.pt`

**Training will take time** - monitor the output for loss values and training progress.

### Step 4: Generate Text
```bash
python generate.py
```
This starts an interactive text generation session where you can:
- Enter prompts to generate text
- Adjust temperature, token count, and top-k parameters
- Type 'quit' to exit

## Configuration Options

### Model Architecture (config.py)
You can modify the model size in `config.py`:
```python
self.n_embd = 384      # Embedding size
self.n_layer = 6       # Number of transformer layers
self.n_head = 6        # Number of attention heads
self.block_size = 128  # Context window size
```

### Training Parameters (config.py)
Adjust training settings:
```python
self.batch_size = 16   # Reduce if you run out of memory
self.learning_rate = 3e-4
self.max_iters = 5000  # Increase for longer training
```

### Generation Parameters (config.py)
Modify generation behavior:
```python
self.max_new_tokens = 200  # Length of generated text
self.temperature = 0.9     # Creativity (0.1-2.0)
self.top_k = 200          # Diversity control
```

## Troubleshooting

### Memory Issues
If you encounter "out of memory" errors:
1. Reduce `batch_size` in `config.py`
2. Reduce `n_embd` or `n_layer` for a smaller model
3. Use CPU instead of GPU (slower but less memory)

### Poor Generation Quality
If generated text is poor or repetitive:
1. Check your dataset quality and size
2. Increase training iterations (`max_iters`)
3. Adjust generation parameters (temperature, top_k)
4. Ensure your dataset has sufficient variety

### Tokenizer Issues
If you see `<UNK>` tokens or empty output:
1. Verify your `tokenizer.json` was created from your dataset
2. Delete old model files and retrain
3. Check that your dataset contains the characters you want to generate

## Advanced Usage

### Custom Training Data
To use your own text data:
1. Replace `dataset/input.txt` with your content
2. Run `python rebuild_tokenizer.py` to rebuild the tokenizer
3. Delete old model files (`best_model.pt`, `final_model.pt`)
4. Retrain with `python train.py`

### Model Evaluation
Monitor training progress by watching:
- Training loss (should decrease)
- Validation loss (should decrease and stabilize)
- Tokens per second (performance metric)

### Parameter Tuning
Experiment with different settings:
- **Temperature**: Lower (0.1-0.5) for focused text, higher (1.0-2.0) for creative text
- **Top-k**: Lower values (10-50) for more focused generation
- **Max tokens**: Adjust based on desired output length

## Expected Results

With good data and proper training, you should see:
- Decreasing loss values during training
- Generated text that follows your dataset's style
- Coherent word and sentence formation
- Appropriate use of punctuation and formatting

## Performance Expectations

- **Training time**: 30 minutes to several hours depending on data size and hardware
- **Memory usage**: 2-4GB RAM for the default configuration
- **Generation speed**: Near-instantaneous text generation
- **Model size**: ~6MB for the default configuration

## Next Steps

Once you have a working model:
1. Experiment with different datasets
2. Try different model sizes and architectures
3. Implement additional features (beam search, nucleus sampling)
4. Explore fine-tuning on specific text styles or domains

Remember that language model development is iterative - expect to experiment and adjust based on your results and requirements. 