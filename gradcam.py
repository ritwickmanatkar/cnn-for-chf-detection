import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import MITBIH_SAMPLING_FREQUENCY


def make_gradcam_heatmap_1d(array_1d, model, layer_name, pred_index=None):
    """ This builds a heatmap based on importance extracted from the last convolution layer.

    :param array_1d: Example to be worked on
    :param model: Model in question
    :param layer_name: Layer name to be evaluated
    :param pred_index:

    :return: Gradient map
    """
    # Create a model mapping the input 1D array to the activations of the last conv layer
    # and the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class with respect to the activations
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(array_1d)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of the class channel with respect to the feature map
    grads = tape.gradient(class_channel, layer_output)

    # Compute channel-wise mean of the gradients (only over the channel axis)
    pooled_grads = tf.reduce_mean(grads, axis=0)

    # Multiply each channel by the corresponding pooled gradient value
    layer_output = layer_output[0]
    heatmap = tf.reduce_sum(layer_output * pooled_grads, axis=-1)

    # Normalize the heatmap to the range [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def display_gradcam_timeseries(input_series, heatmap):
    """ This displays the gradcam gradients overlayed on the input series.

    :param input_series: Input series on which to overlay
    :param heatmap: Gradients to overlay.

    :return: None
    """
    # Ensure heatmap is normalized between 0 and 1
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap_interpolated = np.interp(np.linspace(0, len(heatmap)-1, len(input_series)),
                                     np.arange(len(heatmap)), heatmap)

    cmap = plt.cm.jet
    heatmap_colors = cmap(heatmap_interpolated)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(input_series, color="black", label="Input Time Series", linewidth=1.5)

    # Overlay the heatmap as a background
    for i in range(len(input_series) - 1):
        ax.plot(
            [i, i + 1],
            [input_series[i], input_series[i + 1]],
            color=heatmap_colors[i],
            alpha=0.4,
            linewidth=4,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Grad-CAM Intensity", fontsize=10)

    ax.set_title("Grad-CAM Heatmap", fontsize=14)
    ax.set_xlabel(f"Samples (Sampling rate is {MITBIH_SAMPLING_FREQUENCY} Hz)", fontsize=12)
    ax.set_ylabel("Z-Normalized Signal", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()
