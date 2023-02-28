from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import swin_transformer as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class SwinFeature(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.features = swin.features

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SwinTransformerTest(parameterized.TestCase):
    def test_swin_backbone(self):
        flax_model = flax_models.swin_t(num_classes=0)
        torch_model = torch_models.swin_t(weights=torch_models.Swin_T_Weights.DEFAULT)
        torch_model = SwinFeature(torch_model)
        assert_computable(flax_model, is_training=True)
        assert_computable(flax_model, is_training=False)
        assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_swin_t(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_t(num_classes=num_classes)
            torch_model = torch_models.swin_t(weights=torch_models.Swin_T_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_swin_s(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_s(num_classes=num_classes)
            torch_model = torch_models.swin_s(weights=torch_models.Swin_S_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_swin_b(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_b(num_classes=num_classes)
            torch_model = torch_models.swin_b(weights=torch_models.Swin_B_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
