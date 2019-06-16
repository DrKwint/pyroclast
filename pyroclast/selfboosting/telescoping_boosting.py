import sonnet as snt
import tensorflow as tf

from pyroclast.classification import telescoping_boosting


def learn(data, network, seed, total_epochs):
    telescoping_boosting.build_model()
