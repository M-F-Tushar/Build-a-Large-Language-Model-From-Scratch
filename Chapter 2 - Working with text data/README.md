# Chapter 2: Working with Text Data - Complete Guide

## Chapter Overview

This chapter covers the essential preprocessing steps required before training Large Language Models (LLMs). The focus is on preparing text data for decoder-only LLMs based on the transformer architecture (like GPT models). The chapter implements a complete data sampling pipeline that converts raw text into the vector representations needed for LLM training.

### Key Learning Objectives
- Understand why text needs to be converted to numerical vectors for LLMs
- Learn tokenization techniques (simple and advanced)
- Master byte pair encoding (BPE) tokenization
- Implement sliding window data sampling
- Create token embeddings and positional encodings
- Build a complete data preprocessing pipeline

## 2.1 Understanding Word Embeddings

### Core Concept
Deep neural networks, including LLMs, cannot process raw text directly. Since text is categorical data, it's incompatible with the mathematical operations used in neural networks. Therefore, we need to convert words into continuous-valued vectors.

### What are Embeddings?
**Embedding** is the process of converting data into a vector format. Different data types require specific embedding models:
- Video embedding models → Video vectors
- Audio embedding models → Audio vectors  
- Text embedding models → Text vectors

### Key Properties of Word Embeddings

1. **Mapping**: Embeddings map discrete objects (words, images, documents) to points in continuous vector space
2. **Purpose**: Convert non-numeric data into neural network-compatible format
3. **Similarity**: Similar words appear close together in embedding space
4. **Dimensionality**: Can range from 1 to thousands of dimensions

### Word2Vec Example
Word2Vec was an early popular approach that trained neural networks to generate word embeddings by predicting context. The main principle: words appearing in similar contexts tend to have similar meanings.

**Visualization Example**: In 2D embedding space:
- Bird types (duck, eagle, goose) cluster together
- Countries and capitals (Germany-Berlin, England-London) group separately
- Similar concepts (long, longer, longest) appear close to each other

### LLM Embedding Approach
Modern LLMs create their own embeddings as part of the input layer, optimized during training. This approach offers advantages:
- Embeddings are optimized for the specific task and data
- Better performance than using pre-trained embeddings like Word2Vec
- Contextualized output embeddings (discussed in Chapter 3)

### Embedding Dimensions in Practice
- **GPT-2 small** (117M-125M parameters): 768 dimensions
- **GPT-3 large** (175B parameters): 12,288 dimensions
- Trade-off between performance and computational efficiency

## 2.2 Tokenizing Text

### What is Tokenization?
Tokenization splits input text into individual tokens - either words or special characters (including punctuation). This is a required preprocessing step before creating embeddings.

### Text Processing Pipeline
```
Input text → Tokenized text → Token IDs → Token embeddings → LLM processing
```

### Example Implementation: Simple Tokenizer

#### Step 1: Basic Whitespace Splitting
```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
# Output: ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```

#### Step 2: Handling Punctuation
```python
result = re.split(r'([,.]|\s)', text)
# Output: ['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']
```

#### Step 3: Removing Whitespace
```python
result = [item for item in result if item.strip()]
# Output: ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

#### Step 4: Extended Punctuation Handling
```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# Output: ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

### Real-World Application
Using Edith Wharton's "The Verdict" (20,479 characters):
```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# Result: 4,690 tokens
```

**Sample output**: `['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', ...]`

## 2.3 Converting Tokens into Token IDs

### Building a Vocabulary
A vocabulary maps each unique word and special character to a unique integer (token ID).

#### Process:
1. **Tokenize** the entire training dataset
2. **Sort** unique tokens alphabetically  
3. **Create mapping** from tokens to integers

#### Implementation Example
```python
# Create vocabulary from preprocessed tokens
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)  # Result: 1,130 for "The Verdict"

# Create token-to-ID mapping
vocab = {token: integer for integer, token in enumerate(all_words)}

# Sample entries:
# ('!', 0), ('"', 1), ("'", 2), ..., ('Her', 49), ('Hermia', 50)
```

### Complete Tokenizer Implementation

#### Listing 2.3: SimpleTokenizerV1 Class
```python
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab  # Token to ID mapping
        self.int_to_str = {i: s for s, i in vocab.items()}  # ID to token mapping
    
    def encode(self, text):
        """Convert text to token IDs"""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

#### Usage Example
```python
tokenizer = SimpleTokenizerV1(vocab)
text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
ids = tokenizer.encode(text)
# Output: [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

decoded_text = tokenizer.decode(ids)
# Output: Original text reconstructed
```

### Problem with Unknown Words
```python
text = "Hello, do you like tea?"
tokenizer.encode(text)  # KeyError: 'Hello' - word not in vocabulary
```

**Solution**: Need to handle unknown words and add special tokens.

## 2.4 Adding Special Context Tokens

### Special Tokens Purpose
Special tokens provide additional context and handle edge cases:

- **`<|unk|>`**: Represents unknown words not in vocabulary
- **`<|endoftext|>`**: Separates unrelated text sources during training

### Document Separation Example
When training on multiple independent documents:
```
Document 1: "...the underdog team finally clinched the championship..."
<|endoftext|>
Document 2: "The Dow Jones Industrial Average closed up 250 points..."
<|endoftext|>  
Document 3: "Elara and Finn lived with kindness and wisdom..."
```

### Enhanced Tokenizer Implementation

#### Listing 2.4: SimpleTokenizerV2 Class
```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown words with <|unk|> token
        preprocessed = [item if item in self.str_to_int 
                       else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
```

#### Extended Vocabulary
```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
# New vocabulary size: 1,132 (was 1,130)
```

#### Testing with Unknown Words
```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

tokenizer = SimpleTokenizerV2(vocab)
encoded = tokenizer.encode(text)
# Output: [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]

decoded = tokenizer.decode(encoded)
# Output: "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
```

### Additional Special Tokens (Other LLMs)
Some researchers use additional tokens:
- **`[BOS]`** (Beginning of Sequence): Marks text start
- **`[EOS]`** (End of Sequence): Marks text end  
- **`[PAD]`** (Padding): Extends shorter texts to match batch length

**Note**: GPT models use only `<|endoftext|>` for simplicity, serving as both EOS and padding token.

## 2.5 Byte Pair Encoding

### What is Byte Pair Encoding (BPE)?
BPE is a sophisticated tokenization scheme used by GPT-2, GPT-3, and ChatGPT. It can handle any unknown word by breaking it down into subword units or individual characters.

### Key Advantages
1. **No unknown words**: Can tokenize any text without `<|unk|>` tokens
2. **Subword units**: Breaks unfamiliar words into familiar components
3. **Efficiency**: More efficient than character-level tokenization

### BPE Implementation with tiktoken

#### Installation and Setup
```bash
pip install tiktoken
```

#### Usage Example
```python
import tiktoken

# Initialize BPE tokenizer for GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# Output: [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]

strings = tokenizer.decode(integers)
# Output: Original text perfectly reconstructed
```

### Key Observations
1. **Large vocabulary**: BPE tokenizer has 50,257 tokens total
2. **Special token ID**: `<|endoftext|>` has ID 50256 (largest)
3. **Unknown word handling**: "someunknownPlace" tokenized correctly without `<|unk|>`

### Exercise 2.1: BPE Unknown Words Analysis
**Task**: Analyze how BPE handles "Akwirw ier"

```python
text = "Akwirw ier"
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Decode individual tokens
for token_id in token_ids:
    print(f"ID {token_id}: '{tokenizer.decode([token_id])}'")

# Verify reconstruction
reconstructed = tokenizer.decode(token_ids)
print("Reconstructed:", reconstructed)
```

**Expected breakdown**: Unknown words split into subwords and characters.

### How BPE Works (Simplified)
1. **Start**: Add all individual characters to vocabulary ("a", "b", etc.)
2. **Merge**: Combine frequently occurring character pairs into subwords
3. **Example**: "d" + "e" → "de" (common in "define", "depend", "made", "hidden")
4. **Iterate**: Continue merging based on frequency cutoffs

## 2.6 Data Sampling with a Sliding Window

### Next-Word Prediction Task
LLMs learn by predicting the next word in a sequence. We need to create input-target pairs where the target is the input shifted by one position.

### Basic Input-Target Creation
```python
context_size = 4  # Number of tokens in input
enc_sample = enc_text[50:]  # Skip first 50 tokens for interesting content

x = enc_sample[:context_size]          # Input tokens
y = enc_sample[1:context_size+1]       # Target tokens (shifted by 1)

print(f"x: {x}")  # [290, 4920, 2241, 287]
print(f"y: {y}")  # [4920, 2241, 287, 257]
```

### Creating Training Examples
```python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# Output:
# [290] ----> 4920
# [290, 4920] ----> 2241  
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257

# In text format:
# and ----> established
# and established ----> himself
# and established himself ----> in
# and established himself in ----> a
```

### Dataset Implementation

#### Listing 2.5: GPTDatasetV1 Class
```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize entire text
        token_ids = tokenizer.encode(txt)
        
        # Create overlapping sequences using sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### DataLoader Implementation

#### Listing 2.6: create_dataloader_v1 Function
```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                        stride=128, shuffle=True, drop_last=True, 
                        num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # Prevents loss spikes from uneven batches
        num_workers=num_workers  # CPU processes for preprocessing
    )
    return dataloader
```

### Testing the DataLoader
```python
# Small example for understanding
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# Output: [tensor([[40, 367, 2885, 1464]]), tensor([[367, 2885, 1464, 1807]])]

second_batch = next(data_iter)
print(second_batch)  
# Output: [tensor([[367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
```

### Understanding Stride Parameter

**Stride = 1**: Maximum overlap, shifts by 1 position
- Batch 1 input: [40, 367, 2885, 1464]
- Batch 2 input: [367, 2885, 1464, 1807]

**Stride = 4**: No overlap, shifts by window size
- Batch 1 input: [40, 367, 2885, 1464]  
- Batch 2 input: [1807, 3619, 402, 271]

### Exercise 2.2: DataLoader Experimentation
**Task**: Try different configurations:
1. `max_length=2`, `stride=2`
2. `max_length=8`, `stride=2`

Observe how stride affects data overlap and utilization.

### Batch Size > 1 Example
```python
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs shape:", inputs.shape)     # torch.Size([8, 4])
print("Targets shape:", targets.shape)   # torch.Size([8, 4])

# Each batch contains 8 samples, each with 4 tokens
# Inputs and targets are shifted by 1 position
```

## 2.7 Creating Token Embeddings

### Why Embeddings Are Necessary
LLMs are neural networks trained with backpropagation, requiring continuous vector representations rather than discrete token IDs.

### Basic Embedding Example
```python
import torch

# Simple example: 4 input tokens, small vocabulary
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

# Create embedding layer
torch.manual_seed(123)  # For reproducibility
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Output: 6x3 weight matrix with random values
# Parameter containing:
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010], 
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```

### Embedding Lookup Process
```python
# Single token embedding
print(embedding_layer(torch.tensor([3])))
# Output: tensor([[-0.4015, 0.9666, -1.1481]]) - Row 3 from weight matrix

# Multiple tokens embedding  
print(embedding_layer(input_ids))
# Output: 4x3 tensor, each row is embedding for corresponding token ID
```

### Realistic Embedding Dimensions
```python
# GPT-like dimensions
vocab_size = 50257      # BPE tokenizer vocabulary size
output_dim = 256        # Embedding dimension (smaller than GPT-3's 12,288)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Test with real data
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False)

data_iter = iter(dataloader)  
inputs, targets = next(data_iter)
print("Token IDs shape:", inputs.shape)  # torch.Size([8, 4])

# Create embeddings
token_embeddings = token_embedding_layer(inputs)
print("Embeddings shape:", token_embeddings.shape)  # torch.Size([8, 4, 256])
```

## 2.8 Encoding Word Positions

### The Position Problem
Token embeddings lack positional information - the same token ID produces the same vector regardless of position in the sequence. This is problematic because:

1. **Self-attention is position-agnostic**
2. **Word order matters** for meaning
3. **"Fox jumps over dog"** ≠ **"Dog jumps over fox"**

### Types of Positional Embeddings

#### 1. Absolute Positional Embeddings
- Each position has a unique embedding
- Position 1 gets embedding A, position 2 gets embedding B, etc.
- Used by GPT models

#### 2. Relative Positional Embeddings  
- Focus on relative distances between tokens
- Better generalization to unseen sequence lengths

### GPT-Style Absolute Positional Embeddings

#### Implementation
```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Create position indices [0, 1, 2, 3]
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional embeddings shape:", pos_embeddings.shape)  # torch.Size([4, 256])
```

#### Adding Position to Token Embeddings
```python
# Add positional embeddings to token embeddings
input_embeddings = token_embeddings + pos_embeddings
print("Final input embeddings shape:", input_embeddings.shape)  # torch.Size([8, 4, 256])
```

**Broadcasting**: PyTorch automatically adds the 4×256 positional embeddings to each of the 8 samples in the batch.

### Complete Input Processing Pipeline

The final pipeline converts raw text through these stages:

1. **Raw Text** → "This is an example."
2. **Tokenization** → ["This", "is", "an", "example", "."]  
3. **Token IDs** → [40134, 2052, 133, 389, 12]
4. **Token Embeddings** → 5×256 vectors
5. **Positional Embeddings** → 5×256 position vectors
6. **Input Embeddings** → Token embeddings + Positional embeddings
7. **Ready for LLM processing**

### Key Implementation Points

1. **Embedding Lookup**: More efficient than one-hot encoding + matrix multiplication
2. **Optimization**: Embedding weights are learned during training
3. **Dimensionality**: Balance between expressiveness and computational cost
4. **Position Encoding**: Essential for transformer models to understand sequence order

## Summary and Key Takeaways

### Core Concepts Mastered
1. **Text → Numbers**: LLMs require numerical input, not raw text
2. **Tokenization**: Breaking text into processable units (words/subwords)
3. **Vocabularies**: Mapping tokens to unique integer IDs
4. **Special Tokens**: Handling unknown words and document boundaries
5. **BPE Tokenization**: Advanced method handling any input text
6. **Data Sampling**: Creating input-target pairs for training
7. **Embeddings**: Converting discrete tokens to continuous vectors
8. **Positional Encoding**: Adding sequence order information

### Critical Implementation Details

#### Tokenization Strategy
- Simple regex-based splitting for educational purposes
- BPE for production systems (handles unknown words)
- Special tokens for context (`<|unk|>`, `<|endoftext|>`)

#### Data Loading Efficiency  
- Sliding window approach for sequence generation
- Configurable stride for overlap control
- Batch processing for computational efficiency

#### Embedding Architecture
- Learnable token embeddings (not fixed like Word2Vec)
- Absolute positional embeddings (GPT approach)
- Addition of token + position embeddings

### Practical Considerations

1. **Vocabulary Size**: Larger = more expressive, more memory
2. **Context Length**: Longer = better context, more computation  
3. **Embedding Dimension**: Higher = more capacity, more parameters
4. **Batch Size**: Larger = more stable gradients, more memory

### Next Steps
The input embeddings created in this chapter are now ready for processing by the main LLM components:
- Multi-head attention mechanism (Chapter 3)
- Feed-forward networks  
- Layer normalization
- Residual connections

This preprocessing pipeline forms the foundation for all modern transformer-based language models.
