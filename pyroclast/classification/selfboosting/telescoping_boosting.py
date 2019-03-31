import sonnet as snt
import tensorflow as tf

from pyroclast.classification import selfboosting


def learn(data, network, seed, total_epochs):
    selfboosting.build_model()
