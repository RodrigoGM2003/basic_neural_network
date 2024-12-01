# Neural Network from Scratch in C++

This project implements a neural network from scratch in C++ to solve the MNIST digit classification problem. It handles data parsing, gradient-based learning, and customizable activation functions without relying on external machine learning libraries. The goal is to learn the fundamentals of building and training neural networks.

**Warning:** This project is old and therefore some of the code might be suboptimal or outdated.

---

## Features
- **Custom Implementation**: Fully implemented neural network, including forward propagation, backpropagation, and weight updates.
- **Layer Customization**: Flexible layer definitions with adjustable node counts and activation functions.
- **MNIST Data Parsing**: Handles binary MNIST data and label files for training and testing.
- **Training and Learning**: Supports batch-based learning with customizable batch size, learning rate, and epochs.

---

## Getting Started

### Prerequisites
- C++ compiler supporting at least C++11.
- CMake (for build automation).
- MNIST dataset files:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-network-mnist.git
   cd neural-network-mnist

2. Create a build folder and configure the build with CMake:
  ````bash
   mkdir build
   cd build
   cmake -G "MinGW Makefiles" ..

3. Build the project
   ´´´´bash
   cmake --build .


### Usage
1. Ensure the dataset files are in the project directory

2. After building, the main.exe file will be located in the bin folder inside the project directory.


3. Run the executable, passing the paths to the dataset files
   ´´´´bash
   ./bin/main

### Code Structure
## Core Components
1. data_set: Handles loading and parsing MNIST data files.
2. functions: Contains activation functions (ReLU, Sigmoid) and utility functions.
3. layer: Represents a single layer in the neural network.
4. n_network: Manages the entire network, including forward propagation, backpropagation, and training logic.

### Acknowledgments
* The MNIST dataset: http://yann.lecun.com/exdb/mnist/