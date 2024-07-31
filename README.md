# WIP: BitNet – a 0 dependency BitNet implementation in C

This is my attempt to implement neural network training and inference with the BitLinear layer from the [BitNet](https://arxiv.org/abs/2310.11453) paper from scratch. The long term goal is to work towards an implementation of the LLaMA architecture. This repo also implements inference for a BPE tokenizer trained with the [tiktoken](https://github.com/openai/tiktoken) library.

To keep things concise, the source files for layers, data structures and other utilities are implemented as single header libraries.

## Usage

### Training

The train program initializes a new model and trains it on the dataset specified. For example,

```sh
gcc mnist_train.c -o train_mnist -lm
./train_mnist
```

## Project Structure

```plaintext
├── experiments/    # miscellaneous programs used to investigate ideas
├── layers/         # source files for layers of the LLM
├── utils/          # utility functions (data structures, matrix functions, dataloaders, etc.)
├── tests/          # unit tests for various libraries and functions
├── experiments/    # programs used to investigate ideas
├── scripts/        # utility scripts used for tasks
├── tokenizer.h     # single header library for inference on BPE tokenizer
└── mnist_bitmlp.c  # train and test bit multi layer perceptron on MNIST dataset
```

## Some conventions

Function names for layers contain suffix corresponding to their forward and backward pass.

- `_fwd` – forward pass
- `_bkwd` – backpropagation

Gradient variables are prefixed with `d` eg. gradient of output of a layer is `dy`. Additionally, quantised variables contain a `q` suffix eg. quantised activations will be `xq`.

## Roadmap

- BitLinear implementation
    - [x] RMSNorm layer
    - [x] BitLinear layer
        - [x] Bit matrix multiplications
        - [x] GELU activation
        - [x] Weight and activation quantisation/dequantisation functions
    - [x] BitLinear MLP Block
    - [x] Cross entropy loss implementation
    - [x] Training weight initialisation and allocation
    - [x] AdamW optimiser implementation
    - [x] Training loop on MNIST dataset for BitMLP
    - [ ] Train a multilayer perceptron classifier for the MNIST dataset
    - [ ] Parallelize code using OpenMP
- Tokenizer implementation
    - [x] Loading tokenizer from file
    - [x] Base64 decoding
    - [x] Hashtable implementation
    - [x] PriorityQueue implementation
    - [x] Encode text to input ids using tokenizer
    - [x] Decode input ids to text using tokenizer
    - [ ] Verify correctness of tokenizer implementation on sample corpus
- BitNet transformer implementation
    - [x] Token embedding layer
    - [ ] Grouped query attention block
    - [ ] Forward and backward pass for BitNet architecture
    - [ ] Dataloader implementation
    - [ ] Saving and loading model weights
    - [ ] Training loop implementation

