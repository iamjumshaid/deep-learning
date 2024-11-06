
# Deep Learning - Assignment 2 Multi-layer Perceptrons

## Overview

This assignment involved developing and experimenting with a small feedforward neural network, or Multi-Layer Perceptron (MLP), for binary classification. The key objectives were implementing foundational components of a neural network, testing the effect of different loss functions and activation functions, and conducting experiments with various network configurations.

## Key Components

1. **Network Module Implementations**:
   Implemented core modules, including:
   - **Activation Functions**: Sigmoid, ReLU, and Softmax.
   - **Layers**: A fully connected `Linear` layer.
   - **Loss Functions**: Cross-Entropy Loss.

2. **Model Building and Testing**:
   Built a 2-layer MLP with a hidden layer and tested it on an XOR classification problem. Additional tasks included:
   - Creating a 3-unit hidden layer network while maintaining the same output behavior as the 2-unit model.
   - Modifying network weights for experimentation.

3. **Representation Space Visualization**:
   Transformed the input data into hidden representation space and visualized the dataset both in input and hidden spaces.

## Project Structure

- **lib/network_base.py**: Contains base classes for defining network layers and parameters.
- **lib/activations.py**: Implements activation functions like Sigmoid, ReLU, and Softmax.
- **lib/losses.py**: Implements the Cross-Entropy loss function.
- **lib/network.py**: Defines the `Linear` layer and `Sequential` model.
- **tests/**: Contains unit tests for each component to ensure correctness.
- **run files**: Scripts for running models and generating plots for verification.

## Requirements and Setup

- Python 3.8
- Required packages can be installed using:
  ```bash
  pip install -r requirements.txt
  ```

## Testing
Run the tests:
 ```bash
  python -m pytest
  ```

**Note:**
The pen and paper solution is available in the main directory.
