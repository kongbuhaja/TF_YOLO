import tensorflow as tf
from tensorflow.keras.layers import Activation

activations = {
    "ReLU": Activation("ReLU"),
    "LeakyReLU": Activation("LeakyReLU"),
    "ELU": Activation("ELU"),
    "Softmax": Activation("softmax"),
    "Sigmoid": Activation("sigmoid"),
    "SiLU": Activation("silu"),
    "Swish": Activation("swish"),
    "Mish": Activation("mish"),
    "SELU": Activation("selu"),
    "Softplus": Activation("softplus"),
    "Softsign": Activation("softsign"),
    "Tanh": Activation("tanh"),
}

activations.update({key.lower(): value for key, value in activations.items()})