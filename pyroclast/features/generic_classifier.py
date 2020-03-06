from pyroclast.common.feature_classifier import FeatureClassifierMixin
from pyroclast.common.visualizable import VisualizableMixin
import tensorflow as tf


class GenericClassifier(tf.Module, VisualizableMixin, FeatureClassifierMixin):

    def __init__(self, conv_stack, classifier, name):
        super(GenericClassifier, self).__init__(name=name)
        self.conv_stack = conv_stack
        self.classifier = classifier

    def __call__(self, x):
        return self.classifier(self.features(x))

    def features(self, x):
        return self.conv_stack(x)

    def classify_features(self, z):
        return self.classifier(z)

    def logits(self, x):
        return self(x)

    def conv_stack_submodel(self):
        raise NotImplementedError()
