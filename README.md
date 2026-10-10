# ML Library

A small neural network library that recognizes handwritten MNIST digits, written from scratch without any machine learning framework. The project comes in two versions:

- **C_lib:** The original implementation in plain C, with its own memory manager (arena allocator) and random number generator (PCG32). It has no external dependencies.
- **Py_lib:** A Python port of the same code, built on numpy.

Both versions share the same design. The model is built as a **computation graph** (autograd): every operation becomes a node in the graph, and gradients are computed with the chain rule by walking the graph in reverse topological order.

## Project structure

```
ML Library/
├── C_lib/
│   ├── main.c          # Matrix ops, computation graph, model and training loop
│   ├── arena.c / .h    # Arena memory allocator (Windows: VirtualAlloc, Linux: mmap)
│   ├── random.c / .h   # PCG32 random number generator
│   ├── base.h          # Shared types (u32, f32, ...) and macros
│   └── data_mnist.py   # Downloads MNIST and writes it to .mat files
└── Py_lib/
    ├── main.py         # Python/numpy port of main.c
    └── mnist.py
```

## Model

| Layer | Size | Operation |
|---|---|---|
| Input | 784 × 1 | 28×28 pixels, normalized to 0–1 |
| Hidden layer 1 | 16 | `W0 · x + b0` → ReLU |
| Hidden layer 2 | 16 | `W1 · a0 + b1` → ReLU, followed by a residual connection (`a1 = a0 + relu(...)`) |
| Output | 10 | `W2 · a1 + b2` → Softmax |

- **Loss function:** Cross-entropy
- **Optimization:** Mini-batch gradient descent (batch size 50, learning rate 0.01, 10 epochs)
- **Weight initialization:** Glorot/Xavier uniform, `±sqrt(6 / (fan_in + fan_out))`

Supported operations: `relu`, `sigmoid`, `tanh`, `softmax`, `add`, `sub`, `matmul`, `cross_entropy`. Each one has both a forward and a backward pass, so new layers can be built freely from these operations.

## Setup

### 1. Prepare the data

The `.mat` files are not included in the repository because of their size (the training images alone are about 188 MB). Generate them once:

```bash
pip install numpy tensorflow tensorflow-datasets
cd C_lib
python data_mnist.py
```

The script produces four files: `train_images.mat`, `train_labels.mat`, `test_images.mat` and `test_labels.mat`. They are raw `float32` arrays. If you also want to run the Python version, copy the files into the `Py_lib` folder as well.

### 2a. C version

```bash
cd C_lib
gcc main.c arena.c random.c -o main.exe -lm
./main.exe
```

On Windows (MinGW), `-lm` is not needed. The program switches the console to UTF-8 itself so the digit preview renders correctly.

### 2b. Python version

Requires Python 3.10 or newer (the code uses `match` statements).

```bash
pip install numpy
cd Py_lib
python main.py
```

## Output

The program first draws a digit from the test set in the terminal and prints its label. It then prints the untrained model's prediction, trains for 10 epochs, and reports the test accuracy after each epoch:

```
Epoch  1 / 10, Batch 1200 / 1200, Average Cost: 0.4890
Test Completed. Accuracy:  8913 / 10000 (89.1%), Average Cost: 0.4070
...
Epoch 10 / 10, Batch 1200 / 1200, Average Cost: 0.3857
Test Completed. Accuracy:  9381 / 10000 (93.8%), Average Cost: 0.2158
```

After 10 epochs, the Python version reaches about **93.8%** accuracy on the test set.
