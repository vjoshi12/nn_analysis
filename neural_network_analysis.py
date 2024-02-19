import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from tensorflow.keras.utils import to_categorical


# Values for analysis
NUM_LAYERS_OPTIONS = (1, 2, 3, 4, 5)
VERY_DEEP_NETWORK_OPTIONS = (10, 50, 100, 1000, 10000)
LAYER_SIZE_OPTIONS = (50, 100, 500, 1000, 10000)
LEARNING_RATE_OPTIONS = (1.0, 0.1, 0.01, 0.005, 0.001)
OPTIMIZER_OPTIONS = (SGD, Adagrad, RMSprop, Adam, Nadam)
ACTIVATION_OPTIONS = ('linear', 'relu', 'tanh', 'sigmoid', 'softplus', 'swish')
INITIALIZER_OPTIONS = ('zeros', 'random_uniform', 'random_normal', 'glorot_uniform', 'he_uniform')


class MNISTDigitNeuralNetwork:
    def __init__(
        self,
        num_layers=3,
        layer_size=10,
        activation_fn='relu',
        optimizer_cls=RMSprop,
        kernel_initializer='glorot_uniform',
        learning_rate=None,
        name='',
    ):
        self.num_layers = num_layers
        self.layer_size = 10
        self.name = name
        self.learning_rate = learning_rate
        if learning_rate is not None:
            optimizer_cls = SGD
        self.history = None
        hidden_layers = [
            Dense(
                layer_size,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                name=f'dense_{n}',
            ) for n in range(num_layers)
        ]
        self.model = tf.keras.Sequential(
            [tf.keras.Input(shape=(784,), name='digits')] +
            hidden_layers +
            [Dense(10, activation='linear', name='predictions')]
        )
        self.model.compile(
            optimizer=optimizer_cls(**({'learning_rate': learning_rate} if learning_rate else {})),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def fit(self, x, y, epochs=100, validation_data=None):
        self.history = self.model.fit(x, y, epochs=epochs, validation_data=validation_data)

    def _history_as_x_y(self, info):
        vals = self.history.history[info]
        return [i for i in range(len(vals))], vals

    def loss_history(self):
        return self._history_as_x_y('loss')

    def accuracy_history(self):
        return self._history_as_x_y('accuracy')

    def val_loss_history(self):
        return self._history_as_x_y('val_loss')

    def val_accuracy_history(self):
        return self._history_as_x_y('val_accuracy')

    def predict(self, x):
        return self.model.predict(x)


def run_gradient_analysis():
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = get_mnist_digit_data()
    random_models = [MNISTDigitNeuralNetwork(
        num_layers=n,
        kernel_initializer='random_uniform',
        activation_fn='sigmoid',
    ) for n in range(1, 6)]
    glorot_models = [MNISTDigitNeuralNetwork(
        num_layers=n,
        kernel_initializer='glorot_uniform',
        activation_fn='sigmoid',
    ) for n in range(1, 6)]

    # Prepare the plots
    fig, ax = plt.subplots(2, 5)
    fig.suptitle('Neural Network Analysis')

    def get_gradients_for_models(models):
        gradients = []
        min_grad, max_grad = float('inf'), -float('inf')
        for m in models:
            # Fit model to training data
            m.fit(x_train, y_train, validation_data=(x_val, y_val))

            # Obtain gradients w.r.t. test data
            with tf.GradientTape() as tape:
                pred = m.model(x_test)
                loss = categorical_crossentropy(to_categorical(y_test), pred)
            grads = tape.gradient(loss, m.model.trainable_variables)

            # Remove first two gradient lists (input layer to first hidden layer)
            grads = grads[2:]

            # Flatten and normalize gradients
            grads = [np.ndarray.flatten(g.numpy()) for g in grads]
            grads = [np.abs(g) for g in grads]
            min_grad = min(min_grad, np.min(np.concatenate(grads)))
            max_grad = max(max_grad, np.max(np.concatenate(grads)))
            gradients.append(grads)
        return gradients, min_grad, max_grad

    r_grads, r_min_grad, r_max_grad = get_gradients_for_models(random_models)
    g_grads, g_min_grad, g_max_grad = get_gradients_for_models(glorot_models)

    def plot_gradients(gradients, min_grad, max_grad, col):
        for i, grads in enumerate(gradients):
            grads = [(g - min_grad) * 100.0 / (max_grad - min_grad) for g in grads]

            # Plot the gradients
            x = []
            y = []
            colors = []
            for n in range(len(grads)):
                for m in range(len(grads[n])):
                    x.append(n)
                    y.append(m)
                    colors.append(grads[n][m])
            sizes = [5 for _ in range(len(x))]
            ax[col][i].scatter(x, y, c=colors, cmap='Reds', s=sizes)
            initializer = 'random' if col == 0 else 'glorot'
            ax[col][i].set_title(f'{initializer} {i+1} layers')

    plot_gradients(r_grads, r_min_grad, r_max_grad, 0)
    plot_gradients(g_grads, g_min_grad, g_max_grad, 1)

    plt.show()
    plt.waitforbuttonpress()


def plot_activation_functions():
    # Prepare the plots
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Activation Functions')
    plt.style.use('_mpl-gallery')

    # Plot all activation functions from x = -5 to x = 5
    x = np.linspace(-5, 5, 100)

    def functions():
        fns = (
            (x, 'linear'),
            (x.clip(min=0), 'relu'),
            (np.tanh(x), 'tanh'),
            (1 / (1 + np.exp(-x)), 'sigmoid'),
            (np.log(np.exp(x) + 1), 'softplus'),
            (x * (1 / (1 + np.exp(-x))), 'swish'),
        )
        for (y, name) in fns:
            yield (y, name)

    fns = functions()
    for i in range(2):
        for j in range(3):
            (y, name) = next(fns)
            ax[i, j].plot(x, y, linewidth=2.0)
            ax[i, j].grid()
            ax[i, j].set(
                title=name,
                xlim=(-1, 1) if name in ('linear', 'relu') else (-5, 5),
                xticks=[-1, 0, 1] if name in ('linear', 'relu') else [-5, 0, 5],
                ylim=(-1, 1),
                yticks=[-1, 0, 1],
            )

    plt.show()
    plt.waitforbuttonpress()


def plot_info(analysis_type, model, ax):
    for (plot_x, plot_y, data) in (
        (0, 0, model.loss_history()),
        (0, 1, model.accuracy_history()),
        (1, 0, model.val_loss_history()),
        (1, 1, model.val_accuracy_history()),
    ):
        x, y = data[0], data[1]
        if analysis_type in ('num_layers', 'layer_size', 'initializer'):
            # Only show the last 10 epochs for these analysis types
            x = x[90:]
            y = y[90:]
        ax[plot_x, plot_y].plot(x, y, label=model.name)


def run_analysis(analysis_type):
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = get_mnist_digit_data()

    # Configure multiple different models based on the analysis type
    if analysis_type == 'num_layers':
        models = [MNISTDigitNeuralNetwork(num_layers=n, name=f'{n} layers') for n in NUM_LAYERS_OPTIONS]
    elif analysis_type == 'very_deep':
        models = [MNISTDigitNeuralNetwork(num_layers=n, name=f'{n} layers') for n in VERY_DEEP_NETWORK_OPTIONS]
    elif analysis_type == 'layer_size':
        models = [MNISTDigitNeuralNetwork(layer_size=n, name=f'{n} nodes') for n in LAYER_SIZE_OPTIONS]
    elif analysis_type == 'learning_rate':
        models = [MNISTDigitNeuralNetwork(learning_rate=n, name=f'alpha={n}') for n in LEARNING_RATE_OPTIONS]
    elif analysis_type == 'optimizer':
        models = [MNISTDigitNeuralNetwork(optimizer_cls=o, name=o.__name__) for o in OPTIMIZER_OPTIONS]
    elif analysis_type == 'activation':
        models = [MNISTDigitNeuralNetwork(activation_fn=a, name=a) for a in ACTIVATION_OPTIONS]
    elif analysis_type == 'initialization':
        models = [MNISTDigitNeuralNetwork(kernel_initializer=i, name=i) for i in INITIALIZER_OPTIONS]

    # Prepare the plots
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Neural Network Analysis')

    success_ratios = {}
    for model in models:
        # Fit the model to the entire training set
        model.fit(x_train, y_train, validation_data=(x_val, y_val))

        # Plot training + validation loss/accuracy
        plot_info(analysis_type, model, ax)

        # Plot test performance
        pred = np.argmax(model.predict(x_test), axis=1)
        success_ratios[model.name] = float(np.count_nonzero(pred - y_test == 0)) / pred.shape[0]

    # Loss curve
    ax[0, 0].legend()
    ax[0, 0].set_title('Training Loss')
    ax[0, 0].set(xlabel='epoch', ylabel='loss')

    # Accuracy
    ax[0, 1].legend()
    ax[0, 1].set_title('Accuracy')
    ax[0, 1].set(xlabel='epoch', ylabel='accuracy')

    # Performance on test data
    success_ratio_values = list(success_ratios.values())
    ax[0, 2].bar(list(success_ratios.keys()), success_ratio_values)
    for i, v in enumerate(success_ratio_values):
        ax[0, 2].text(i, v, str(v), ha='center')
    ax[0, 2].set_title('Performance on Test')

    # Validation loss curve
    ax[1, 0].legend()
    ax[1, 0].set_title('Validation Loss')
    ax[1, 0].set(xlabel='epoch', ylabel='validation loss')

    # Validation accuracy
    ax[1, 1].legend()
    ax[1, 1].set_title('Validation Accuracy')
    ax[1, 1].set(xlabel='epoch', ylabel='validation accuracy')

    plt.show()
    plt.waitforbuttonpress()


def get_mnist_digit_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.reshape(60000, 784).astype("float32") / 255)[:1800]
    x_test = (x_test.reshape(10000, 784).astype("float32") / 255)[:200]
    y_train = y_train.astype("float32")[:1800]
    y_test = y_test.astype("float32")[:200]

    x_val = x_train[-200:]
    y_val = y_train[-200:]
    x_train = x_train[:-200]
    y_train = y_train[:-200]
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="run different analyses of MNIST digit classifier neural networks"
    )
    parser.add_argument("analysis_type", default="num_layers")
    args = parser.parse_args()

    # Get deterministic results
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    if args.analysis_type == 'gradient':
        run_gradient_analysis()
    elif args.analysis_type == 'activation_functions':
        plot_activation_functions()
    else:
        run_analysis(args.analysis_type)
