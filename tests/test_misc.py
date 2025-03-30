from tests._test_utils import SetupDefaults


class TestSetupDefaults(SetupDefaults):
    def test_setup(self):
        # asser not writatable
        with self.assertRaises(TypeError):
            self._DEFAULT_BATCH["something"] = 1  # pyright: ignore
        with self.assertRaises(TypeError):
            self._DEFAULT_CONFIG_DICT["something"] = 1  # pyright: ignore
