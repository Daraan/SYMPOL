import jax.numpy as jnp
import numpy.testing as npt

from ray_utilities.connectors.dummy_connector import DummyNumpyToTensor
from tests._test_utils import SympolSetupDefaults


class TestSetupDefaults(SympolSetupDefaults):
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

    def test_dummy_connector(self):
        connector = DummyNumpyToTensor()
        batch = {"observations": jnp.ones((1, 2, 3))}
        batch_copy = {"observations": jnp.ones((1, 2, 3))}
        converted_batch = connector(batch=batch)
        self.assertIs(converted_batch, batch)
        npt.assert_array_equal(converted_batch["observations"], batch_copy["observations"])
        self.assertEqual(connector(batch=self._DEFAULT_BATCH), self._DEFAULT_BATCH)
