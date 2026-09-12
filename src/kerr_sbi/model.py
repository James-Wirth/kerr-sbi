import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, StandardNormal
from flowjax.flows import masked_autoregressive_flow

from kerr_sbi.config import Config


class Embedding(eqx.Module):
    convolutions: tuple[eqx.nn.Conv2d, ...]
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, cfg: Config, key: jax.Array):
        settings = cfg["training"]
        channels = [1, *settings["conv_channels"]]
        if len(channels) != 6 or any(type(c) is not int or c < 1 for c in channels):
            raise ValueError("embedding requires five positive channel counts")
        keys = jax.random.split(key, 7)
        self.convolutions = tuple(
            eqx.nn.Conv2d(
                channels[index],
                channels[index + 1],
                kernel_size=3,
                stride=1 if index == 0 else 2,
                padding=1,
                key=keys[index],
            )
            for index in range(5)
        )
        self.hidden = eqx.nn.Linear(channels[-1] * 4 * 4, settings["dense_width"], key=keys[5])
        self.output = eqx.nn.Linear(settings["dense_width"], settings["embedding_dim"], key=keys[6])

    def __call__(self, image: jax.Array) -> jax.Array:
        for convolution in self.convolutions:
            image = jax.nn.gelu(convolution(image))
        return self.output(jax.nn.gelu(self.hidden(image.reshape(-1))))


class Posterior(eqx.Module):
    embedding: Embedding
    flow: AbstractDistribution

    def __init__(self, cfg: Config, key: jax.Array):
        embedding_key, flow_key = jax.random.split(key)
        self.embedding = Embedding(cfg, embedding_key)
        settings = cfg["training"]
        self.flow = masked_autoregressive_flow(
            flow_key,
            base_dist=StandardNormal((2,)),
            transformer=RationalQuadraticSpline(
                knots=settings["spline_knots"], interval=settings["spline_interval"]
            ),
            cond_dim=settings["embedding_dim"],
            flow_layers=settings["flow_layers"],
            nn_width=settings["flow_nn_width"],
            nn_depth=settings["flow_nn_depth"],
            invert=True,
        )

    def log_prob(self, z: jax.Array, u: jax.Array) -> jax.Array:
        return self.flow.log_prob(u, condition=jax.vmap(self.embedding)(z))

    def sample(self, z: jax.Array, key: jax.Array, count: int) -> jax.Array:
        return self.flow.sample(key, (count,), condition=jax.vmap(self.embedding)(z))


def masked_nll(model: Posterior, z: jax.Array, u: jax.Array, mask: jax.Array) -> jax.Array:
    values = model.log_prob(z, u)
    return -jnp.where(mask, values, 0).sum() / mask.sum()
