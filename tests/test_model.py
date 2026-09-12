import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.config import Config
from kerr_sbi.model import Posterior, masked_nll


def test_full_embedding_and_flow_shapes(cfg: Config) -> None:
    model = Posterior(cfg, jax.random.PRNGKey(3))
    image = jnp.zeros((1, 64, 64))
    feature = image
    for layer, expected in zip(
        model.embedding.convolutions,
        [(32, 64, 64), (64, 32, 32), (128, 16, 16), (128, 8, 8), (128, 4, 4)],
        strict=True,
    ):
        feature = layer(feature)
        assert feature.shape == expected
    assert model.embedding.hidden.in_features == 2048
    assert model.embedding(image).shape == (64,)
    values = eqx.filter_jit(lambda m: m.log_prob(image[None], jnp.zeros((1, 2))))(model)
    assert values.shape == (1,) and np.isfinite(values).all()
    assert jax.tree_util.tree_leaves(eqx.filter(model.flow.base_dist, eqx.is_inexact_array)) == []


def test_conditional_samples_and_joint_gradients(small_model_cfg: Config) -> None:
    model = Posterior(small_model_cfg, jax.random.PRNGKey(1))
    z = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 64, 64))
    u = jnp.array([[0.3, -0.4], [1.0, 0.8]])
    mask = jnp.array([True, False])
    loss, gradients = eqx.filter_jit(eqx.filter_value_and_grad(masked_nll))(model, z, u, mask)
    assert np.isfinite(loss)
    np.testing.assert_allclose(loss, -model.log_prob(z[:1], u[:1])[0], atol=1e-6)
    for component in (gradients.embedding, gradients.flow):
        leaves = jax.tree_util.tree_leaves(eqx.filter(component, eqx.is_inexact_array))
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)
    sample = eqx.filter_jit(lambda m: m.sample(z, jax.random.key(9), 3))(model)
    assert sample.shape == (3, 2, 2) and np.isfinite(sample).all()
