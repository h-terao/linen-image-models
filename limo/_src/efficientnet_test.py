from __future__ import annotations
from absl.testing import absltest, parameterized

import jax.random as jr
from limo._src import efficientnet as efnet
from limo._src.test_utils import assert_close_outputs
from limo._src.configuration import using_config
import timm


testcases = [
    ("efficientnet_b0",),
    ("efficientnet_b1",),
    ("efficientnet_b2",),
    ("efficientnet_b3",),
    ("efficientnet_b4",),
    ("efficientnet_b5",),
    ("efficientnet_b6",),
    ("efficientnet_b7",),
    ("efficientnet_b8",),
    ("tinynet_a",),
    ("tinynet_b",),
    ("tinynet_c",),
    ("tinynet_d",),
    ("tinynet_e",),
    # ("efficientnetv2_s",),
    # ("efficientnetv2_m",),
    # ("efficientnetv2_l",),
    # ("efficientnetv2_xl",),
    # ("efficientnetv2_s", "tf_efficientnetv2_s"),
    # ("efficientnetv2_m", "tf_efficientnetv2_m"),
    # ("efficientnetv2_l", "tf_efficientnetv2_l"),
    # ("efficientnetv2_xl", "tf_efficientnetv2_xl"),
]


class EfficientNetTest(parameterized.TestCase):
    @parameterized.parameters(*testcases)
    def test_logit(self, limo_name, timm_name=None):
        timm_name = timm_name or limo_name
        torch_like = not timm_name.startswith("tf_")

        with using_config(torch_like=torch_like):
            flax_model = getattr(efnet, limo_name)()
            torch_model = timm.create_model(timm_name, pretrained=True)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    # @parameterized.parameters(*testcases)
    # def test_features(self, arch):
    #     flax_model = getattr(efnet, arch)(num_classes=0)
    #     torch_model = timm.create_model(arch, pretrained=True, num_classes=0)
    #     assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
