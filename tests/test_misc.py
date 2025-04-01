from sympol import SYMPOL_RL, Indices
from tests._test_utils import SetupDefaults


class TestSetupDefaults(SetupDefaults):
    def test_setup(self):
        # asser not writatable
        with self.assertRaises(TypeError):
            self._DEFAULT_BATCH["something"] = 1  # pyright: ignore
        with self.assertRaises(TypeError):
            self._DEFAULT_CONFIG_DICT["something"] = 1  # pyright: ignore

    def test_unpack_vs_concat(self):
        import jax.numpy as jnp
        import numpy.testing as npt

        assert self._OBSERVATION_SPACE.shape is not None
        npt.assert_array_equal(
            jnp.zeros((1, 2) + self._OBSERVATION_SPACE.shape),  # noqa: RUF005
            jnp.zeros((1, 2, *self._OBSERVATION_SPACE.shape)),
        )
