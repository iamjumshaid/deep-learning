"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    x = np.linspace(-4, +4, 101)

    relu = ReLU()
    y = relu(x)

    # plot input and relu output
    plt.plot(x, y)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid(True)
    plt.show()

    # END TODO###################
