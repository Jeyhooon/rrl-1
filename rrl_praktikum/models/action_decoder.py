import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow_probability import distributions as tfd
import numpy as np

from rrl_praktikum.models.base_module import Module
from rrl_praktikum.utilities import distributions


class ActionDecoder(Module):
    def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, distributions.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = distributions.SampleDist(dist)
        elif self._dist == 'onehot':
            x = self.get(f'hout', tfkl.Dense, self._size)(x)
            dist = distributions.OneHotDist(x)
        return dist
