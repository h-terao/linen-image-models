from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import efficientnet as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class EfficientNetTest(parameterized.TestCase):
    def test_efficientnet_b0(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b0(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b0(
                weights=torch_models.EfficientNet_B0_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b1(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b1(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b1(
                weights=torch_models.EfficientNet_B1_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b2(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b2(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b2(
                weights=torch_models.EfficientNet_B2_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b3(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b3(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b3(
                weights=torch_models.EfficientNet_B3_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b4(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b4(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b4(
                weights=torch_models.EfficientNet_B4_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b5(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b5(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b5(
                weights=torch_models.EfficientNet_B5_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b6(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b6(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b6(
                weights=torch_models.EfficientNet_B6_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_b7(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_b7(num_classes=num_classes)
            torch_model = torch_models.efficientnet_b7(
                weights=torch_models.EfficientNet_B7_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_v2_s(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_v2_s(num_classes=num_classes)
            torch_model = torch_models.efficientnet_v2_s(
                weights=torch_models.EfficientNet_V2_S_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_v2_m(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_v2_m(num_classes=num_classes)
            torch_model = torch_models.efficientnet_v2_m(
                weights=torch_models.EfficientNet_V2_M_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_efficientnet_v2_l(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.efficientnet_v2_l(num_classes=num_classes)
            torch_model = torch_models.efficientnet_v2_l(
                weights=torch_models.EfficientNet_V2_L_Weights.DEFAULT
            )

            if num_classes == 0:
                torch_model.classifier = nn.Flatten()

            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
