import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal

sys.path.append("../../../")

from gpax.gp import ExactGP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def dummy_mean_fn(x, params):
    return params["a"] * x**params["b"]


def dummy_mean_fn_priors():
    a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
    b = numpyro.sample("b", numpyro.distributions.Normal(3, 1))
    return {"a": a, "b": b}


def gp_kernel_custom_prior():
    length = numpyro.sample("k_length", numpyro.distributions.Uniform(0, 1))
    scale = numpyro.sample("k_scale", numpyro.distributions.LogNormal(0, 1))
    return {"k_length": length, "k_scale": scale}


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("unsqueeze", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit(kernel, jax_ndarray, unsqueeze):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray, unsqueeze)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_get_samples(kernel, jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)
        assert_equal(len(v), 100)


@pytest.mark.parametrize("chain_dim, samples_dim", [(True, 2), (False, 1)])
def test_get_samples_chain_dim(chain_dim, samples_dim):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100, num_chains=2)
    samples = m.get_samples(chain_dim)
    assert_equal(samples["k_scale"].ndim, samples_dim)
    assert_equal(samples["noise"].ndim, samples_dim)
    assert_equal(samples["k_length"].ndim, samples_dim + 1)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_sample_kernel(kernel):
    m = ExactGP(1, kernel)
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    _ = kernel_params.pop('period')
    param_names = ['k_length', 'k_scale']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


def test_sample_periodic_kernel():
    m = ExactGP(1, 'Periodic')
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    param_names = ['k_length', 'k_scale', 'period']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_fit_with_custom_kernel_priors(kernel):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = ExactGP(1, kernel, kernel_prior=gp_kernel_custom_prior)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_get_mvn_posterior():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


@pytest.mark.parametrize("unsqueeze", [True, False])
def test_single_sample_prediction(unsqueeze):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=unsqueeze)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sample = m._predict(rng_key, X_test, params, 1)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sample, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sample.shape, X_test.squeeze().shape)


def test_prediction():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data()
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sampled = m.predict(rng_keys[1], X_test, samples)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict_in_batches(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict_in_batches(rng_keys[1], X_test, batch_size=4)
    assert isinstance(y_pred, onp.ndarray)
    assert isinstance(y_sampled, onp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_prob_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_fit_predict_with_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


def test_fit_predict_with_prob_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))