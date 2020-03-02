from pyroclast.common.feature_classifier import FeatureClassifierMixin
import tensorflow as tf


class GenericClassifier(tf.Module, FeatureClassifierMixin):

    def __init__(self, conv_stack, classifier, name):
        super(GenericClassifier, self).__init__(name=name)
        self.conv_stack = conv_stack
        self.classifier = classifier

    def __call__(self, x):
        return self.classifier(self.features(x))

    def features(self, x):
        return tf.squeeze(self.conv_stack(x))

    def classify_features(self, z):
        return self.classifier(z)
