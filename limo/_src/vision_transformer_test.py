from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import vision_transformer as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class VisionTransformerTest(parameterized.TestCase):
    def test_vit_b_16(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.vit_b_16(num_classes=num_classes)
            torch_model = torch_models.vit_b_16(weights=torch_models.ViT_B_16_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.heads = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_vit_b_32(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.vit_b_32(num_classes=num_classes)
            torch_model = torch_models.vit_b_32(weights=torch_models.ViT_B_32_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.heads = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_vit_l_16(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.vit_l_16(num_classes=num_classes)
            torch_model = torch_models.vit_l_16(weights=torch_models.ViT_L_16_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.heads = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_vit_l_32(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.vit_l_32(num_classes=num_classes)
            torch_model = torch_models.vit_l_32(weights=torch_models.ViT_L_32_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.heads = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_vit_h_14(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.vit_h_14(num_classes=num_classes)
            torch_model = torch_models.vit_h_14(weights=torch_models.ViT_H_14_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.heads = nn.Flatten()
            assert_computable(flax_model, True, (518, 518, 3), 2)
            assert_computable(flax_model, False, (518, 518, 3), 2)
            assert_close_outputs(
                jr.PRNGKey(0), flax_model, torch_model, (518, 518, 3), batch_size=2
            )


if __name__ == "__main__":
    absltest.main()
