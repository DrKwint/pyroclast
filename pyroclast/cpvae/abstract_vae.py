from abc import ABC, abstractmethod
import tensorflow as tf


class AbstractVAE(ABC, tf.Module):

    @abstractmethod
    def forward_loss(self, inputs):
        pass

    @abstractmethod
    def posterior(self, inputs):
        pass

    @abstractmethod
    def output_distribution(self, inputs):
        pass

    @abstractmethod
    def output_point_estimate(self, inputs):
        pass