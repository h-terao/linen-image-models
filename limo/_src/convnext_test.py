from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import convnext as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class ConvNeXtTest(parameterized.TestCase):
    def test_convnext_tiny(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.convnext_tiny(num_classes=num_classes)
            torch_model = torch_models.convnext_tiny(
                weights=torch_models.ConvNeXt_Tiny_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_convnext_small(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.convnext_small(num_classes=num_classes)
            torch_model = torch_models.convnext_small(
                weights=torch_models.ConvNeXt_Small_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.classifier = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_convnext_base(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.convnext_base(num_classes=num_classes)
            torch_model = torch_models.convnext_base(
                weights=torch_models.ConvNeXt_Base_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.classifier = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_convnext_large(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.convnext_large(num_classes=num_classes)
            torch_model = torch_models.convnext_large(
                weights=torch_models.ConvNeXt_Large_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.classifier = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
