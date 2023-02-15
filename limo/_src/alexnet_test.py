from __future__ import annotations

import jax.random as jr
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import alexnet as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class AlexNetTest(parameterized.TestCase):
    def test_alexnet(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.alexnet(num_classes=num_classes)
            torch_model = torch_models.alexnet(weights=torch_models.AlexNet_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.classifier = nn.Flatten()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
