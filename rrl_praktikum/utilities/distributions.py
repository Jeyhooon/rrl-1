import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]
        self._dtype = prec.global_policy().compute_dtype

    @property
    def name(self):
        return 'OneHotDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)


class TanhBijector(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


def _mnd_sample(self, sample_shape=(), seed=None, name='sample'):
    return tf.random.normal(
        tuple(sample_shape) + tuple(self.event_shape),
        self.mean(), self.stddev(), self.dtype, seed, name)


tfd.MultivariateNormalDiag.sample = _mnd_sample


def _cat_sample(self, sample_shape=(), seed=None, name='sample'):
    assert len(sample_shape) in (0, 1), sample_shape
    assert len(self.logits_parameter().shape) == 2
    indices = tf.random.categorical(
        self.logits_parameter(), sample_shape[0] if sample_shape else 1,
        self.dtype, seed, name)
    if not sample_shape:
        indices = indices[..., 0]
    return indices


tfd.Categorical.sample = _cat_sample
