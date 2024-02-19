# nn_analysis
Exploration of neural network architecture using TensorFlow Keras

# Usage
`python neural_network_analysis.py <analysis_type>`

# Analysis Types
1. `num_layers`: training/validation loss and accuracy as well as test performance for 5 different networks with 1-5 hidden layers, each with 10 neurons
2. `very_deep`: same as `num_layers` but with many more hidden layers
3. `layer_size`: training/validation loss and accuracy as well as test performance for 5 different networks with a single hidden layer that has an increasing number of neurons
4. `learning_rate`: training/validation loss and accuracy as well as test performance for 5 different learning rates applied to stochastic gradient descent
5. `optimizer`: training/validation loss and accuracy as well as test performance for 5 different optimizers
6. `activation`: training/validation loss and accuracy as well as test performance for 6 different activation functions
7. `initialization`: training/validation loss and accuracy as well as test performance for 5 different initializers
8. `activation_functions`: plots of 6 different activation function curves
9. `gradient`: visualize vanishing gradients by comparing random vs. Glorot initialization with sigmoid activation for a 3-layer network
