from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit

from .gp import ExactGP
from .kernels import get_kernel


class DKL(ExactGP):
    """
    Fully Bayesian implementation of deep kernel learning

    Args:
        input_dim: number of input dimensions
        z_dim: latent space dimensionality
        kernel: type of kernel ('RBF', 'Matern', 'Periodic')
        kernel_prior: optional priors over kernel hyperparameters (uses LogNormal(0,1) by default)
        bnn_fn: Custom MLP
        bnn_fn_prior: Bayesian priors over the weights and biases in bnn_fn
        latent_prior: Optional prior over the latent space (BNN embedding)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 bnn_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 bnn_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 latent_prior: Optional[Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        super(DKL, self).__init__(input_dim, kernel, kernel_prior)
        self.bnn = bnn_fn if bnn_fn else bnn
        self.bnn_prior = bnn_fn_prior if bnn_fn_prior else bnn_prior(input_dim, z_dim)
        self.kernel_dim = z_dim
        self.latent_prior = latent_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """DKL probabilistic model"""
        task_dim = X.shape[0]
        # BNN part
        bnn_params = self.bnn_prior(task_dim)
        z = jax.jit(jax.vmap(self.bnn))(X, bnn_params)
        if self.latent_prior:  # Sample latent variable
            z = self.latent_prior(z)
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim)
        # Sample noise
        with numpyro.plate('obs_noise', task_dim):
            noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # GP's mean function
        f_loc = jnp.zeros(z.shape[:2])
        # compute kernel(s)
        k_args = (z, z, kernel_params, noise)
        k = jax.jit(jax.vmap(get_kernel(self.kernel)))(*k_args)
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    @partial(jit, static_argnames='self')
    def _get_mvn_posterior(self,
                           X_train: jnp.ndarray, y_train: jnp.ndarray,
                           X_new: jnp.ndarray, params: Dict[str, jnp.ndarray]
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        # embed data intot the latent space
        z_train = self.bnn(X_train, params)
        z_test = self.bnn(X_new, params)
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(z_test, z_test, params, noise)
        k_pX = get_kernel(self.kernel)(z_test, z_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(z_train, z_train, params, noise)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
        return mean, cov

    @partial(jit, static_argnames='self')
    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using the inferred weights
        of the DKL's Bayesian neural network
        """
        samples = self.get_samples(chain_dim=False)
        predictive = jax.vmap(lambda params: self.bnn(X_new, params))
        z = predictive(samples)
        return z

    def _print_summary(self):
        list_of_keys = ["k_scale", "k_length", "noise", "period"]
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})


def sample_weights(name: str, in_channels: int, out_channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    with numpyro.plate("batch_dim", task_dim, dim=-3):
        w = numpyro.sample(name=name, fn=dist.Normal(
            loc=jnp.zeros((in_channels, out_channels)),
            scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling bias vector"""
    with numpyro.plate("batch_dim", task_dim, dim=-3):
        b = numpyro.sample(name=name, fn=dist.Normal(
            loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def bnn(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Simple MLP for a single MCMC sample of weights and biases"""
    h1 = jnp.tanh(jnp.matmul(X, params["w1"]) + params["b1"])
    h2 = jnp.tanh(jnp.matmul(h1, params["w2"]) + params["b2"])
    h3 = jnp.tanh(jnp.matmul(h2, params["w3"]) + params["b3"])
    z = jnp.matmul(h3, params["w4"]) + params["b4"]
    return z


def bnn_prior(input_dim: int, zdim: int = 2) -> Dict[str, jnp.array]:
    """Priors over weights and biases in the default Bayesian MLP"""
    hdim = [128, 64, 32]

    def _bnn_prior(task_dim: int):
        w1 = sample_weights("w1", input_dim, hdim[0], task_dim)
        b1 = sample_biases("b1", hdim[0], task_dim)
        w2 = sample_weights("w2", hdim[0], hdim[1], task_dim)
        b2 = sample_biases("b2", hdim[1], task_dim)
        w3 = sample_weights("w3", hdim[1], hdim[2], task_dim)
        b3 = sample_biases("b3", hdim[2], task_dim)
        w4 = sample_weights("w4", hdim[2], zdim, task_dim)
        b4 = sample_biases("b4", zdim, task_dim)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3, "w4": w4, "b4": b4}

    return _bnn_prior
