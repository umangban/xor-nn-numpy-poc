#xor-nn-numpy-poc

A proof of concept demonstrating how to build a neural network from scratch using NumPy to learn the XOR logical function.

#Overview

This project implements a simple 2-layer neural network without using any ML frameworks (no PyTorch, no TensorFlow).
The model is trained to learn the XOR function — one of the smallest problems that requires a non-linear neural network.

The purpose of this POC is to build strong intuition around core deep learning fundamentals.

# What This Project Covers

    - Forward propagation

    - Hidden layers and non-linearity

    - Activation functions (tanh / ReLU)

    - Backpropagation (manual gradient computation)

    - Weight updates using gradient descent

Through this, I deepen my understanding of how neural networks actually learn — which later helps in understanding modern architectures like Transformers and LLMs at a conceptual level.

Why NumPy Instead of PyTorch or TensorFlow?

NumPy forces manual implementation of:

    - weight initialization

    - forward pass logic

    - derivative calculations

    - parameter updates
