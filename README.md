---

# Bigram Language Model with Transformers

This project implements a Bigram Language Model using a Transformer architecture in PyTorch. The model is trained on the "Tiny Shakespeare" dataset to generate Shakespearean text.

## Features

- Token and position embeddings
- Multi-head self-attention mechanism
- Feedforward neural network
- Layer normalization
- Text generation capability

## Requirements

- Python 3.x
- PyTorch
- tqdm

## Installation

1. Clone the repository:

  ```bash
  git clone https://github.com/Ashwin-RN/Shakespeare-NanoGPT.git
  cd Shakespeare-NanoGPT
   ```

2. Install the required packages:

   ```bash
   pip install torch tqdm
   ```

## Usage

### Data Preparation

Place your text file (e.g., `tinyshakespeare.txt`) in the appropriate directory. Update the `file_path` variable in the script to point to this file.

### Training

Run the script to train the model:

```bash
python your_script.py
```

The script will:

- Load and preprocess the text data
- Define and initialize the model
- Train the model
- Save the trained model to `bigram_model.pth`

### Text Generation

After training, the script will generate text based on the trained model. The generated text will be printed to the console.

### Loading a Pretrained Model

To load a pretrained model, uncomment the following lines in the script:

```python
model = BigramLanguageModel()
model.load_state_dict(torch.load('bigram_model.pth'))
m = model.to(device)
```

## Code Structure

- **Head**: Implements a single head of self-attention.
- **MultiHeadAttention**: Combines multiple heads of self-attention.
- **FeedForward**: Implements a feedforward neural network.
- **Block**: Combines multi-head attention and feedforward network with layer normalization.
- **BigramLanguageModel**: The main model class, combining token and position embeddings with transformer blocks and a final linear layer.

## Example

The following is an example of how to run the script:

```bash
python your_script.py
```

