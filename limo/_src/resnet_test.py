from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import resnet as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class ResNetTest(parameterized.TestCase):
    def test_resnet18(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnet18(num_classes=num_classes)
            torch_model = torch_models.resnet18(weights=torch_models.ResNet18_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnet34(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnet34(num_classes=num_classes)
            torch_model = torch_models.resnet34(weights=torch_models.ResNet34_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnet50(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnet50(num_classes=num_classes)
            torch_model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnet101(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnet101(num_classes=num_classes)
            torch_model = torch_models.resnet101(weights=torch_models.ResNet101_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnet152(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnet152(num_classes=num_classes)
            torch_model = torch_models.resnet152(weights=torch_models.ResNet152_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnext50_32x4d(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnext50_32x4d(num_classes=num_classes)
            torch_model = torch_models.resnext50_32x4d(
                weights=torch_models.ResNeXt50_32X4D_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnext101_32x8d(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnext101_32x8d(num_classes=num_classes)
            torch_model = torch_models.resnext101_32x8d(
                weights=torch_models.ResNeXt101_32X8D_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_resnext101_64x4d(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.resnext101_64x4d(num_classes=num_classes)
            torch_model = torch_models.resnext101_64x4d(
                weights=torch_models.ResNeXt101_64X4D_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_wide_resnet50_2(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.wide_resnet50_2(num_classes=num_classes)
            torch_model = torch_models.wide_resnet50_2(
                weights=torch_models.Wide_ResNet50_2_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_wide_resnet101_2(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.wide_resnet101_2(num_classes=num_classes)
            torch_model = torch_models.wide_resnet101_2(
                weights=torch_models.Wide_ResNet101_2_Weights.DEFAULT
            )
            if num_classes == 0:
                torch_model.fc = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
