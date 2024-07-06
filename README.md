# BitLlama – a 0 dependencies BitNet implementation in C

This is my attempt to implement an LLM based on the BitNet paper from scratch. The architecture is a custom version of the LLaMA architecture that incorporates the BitLinear layer, and Contextual Positional Encodings (CoPE) to replace Rotary Position Enocdings. It also implements the inference process for a BPE tokenizer trained with the [tiktoken](https://github.com/openai/tiktoken) library.

This repo also contains all the necessary code needed for training and running inference on the model in separate libraries such as dataloaders, base64 decoding, and data structures such as PriorityQueues and Hashtables.

The source files for layers, data structures and other utilities are implemented as single header libraries in this repo to keep things concise, and easy to copy to other projects.

## Usage

### Inference

The `infer.c` program runs the model in a small shell, where the input prompt entered will be sent to the model as input. To start the program, compile and run it with the following command:

```sh
./infer -t tokenizer_path -w weights_path
```

### Training

The train program initializes a new model and trains it on the dataset specified.

```sh
./train -t tokenizer_path -d dataset_directory_path
```

To run the test cases defined in code, compile the program with the `DEBUG` macro defined.

```sh
gcc -o infer infer.c -DDEBUG
```

## Project Structure

```plaintext
├── layers/         # source files for layers of the LLM
├── utils/          # utility functions (data structures, matrix functions, dataloaders, etc.)
├── tests/          # unit tests for various libraries and functions
├── experiments/    # programs used to investigate ideas
├── scripts/        # utility scripts used for tasks
├── bitnet.h        # single header library for training the bitnet model
├── bitnet_infer.h  # single header libary for inference of the bitnet model
├── tokenizer.h     # single header library for inference on BPE tokenizer
├── infer.c         # program to run an inference loop
└── train.c         # program to run the training loop for the model
```

## Roadmap

- Tokenizer implementation
    - [x] Loading tokenizer from file
    - [x] Base64 decoding
    - [x] Hashtable implementation
    - [x] PriorityQueue implementation
    - [x] Encode text to input ids using tokenizer
    - [x] Decode input ids to text using tokenizer
- Training loop implementation
    - [x] Token embedding layer
    - [x] RMSNorm layer
    - [ ] BitLinear layer
        - [x] Bit matrix multiplications
        - [x] GELU activation
        - [x] Weight and activation quantisation/dequantisation functions
    - [ ] Grouped query attention
        - [ ] KV Cache
        - [ ] Contextual position embeddings
        - [ ] MLP Block
    - [ ] Forward and backward pass for BitNet architecture
    - [ ] Training weight initialisation and allocation
    - [ ] Dataloader implementation
    - [ ] AdamW optimiser implementation
    - [ ] Cross entropy loss implementation
    - [ ] Saving model weights and checkpointing
    - [ ] Training loop program
- Inference implementation
    - [ ] Loading saved model weights
    - [ ] Memory optimised inference pass
    - [ ] Interactive inference loop

## Some conventions

Function names for layers contain suffix corresponding to their forward and backward pass.

- `_fwd` – forward pass
- `_bkwd` – backpropagation

Gradient variables are prefixed with `d` eg. gradient of output of a layer is `dy`. Additionally, quantised variables contain a `q` suffix eg. quantised activations will be `xq`.

