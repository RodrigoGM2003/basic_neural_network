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
