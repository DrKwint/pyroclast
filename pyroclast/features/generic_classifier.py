from pyroclast.common.feature_classifier_mixin import FeatureClassifierMixin
from pyroclast.common.visualizable import VisualizableMixin
import tensorflow as tf


class GenericClassifier(tf.Module, VisualizableMixin, FeatureClassifierMixin):

    def __init__(self, conv_stack, classifier, name):
        super(GenericClassifier, self).__init__(name=name)
        self.conv_stack = conv_stack
        self.classifier = classifier

    def __call__(self, x):
        embed = self.features(x)
        if len(embed.shape) > 2:
            embed = tf.squeeze(embed)
        return self.classifier(embed)

    def features(self, x):
        return self.conv_stack(x)

    def classify_features(self, z):
        return self.classifier(z)

    def get_classification_module(self):
        return self.classifier

    def logits(self, x):
        return self(x)

    def conv_stack_submodel(self):
        return self.conv_stack
