from __future__ import annotations

from absl.testing import absltest, parameterized
import jax.random as jr

from timm.models.convnext import ConvNeXtBlock, ConvNeXtStage, ConvNeXt
from timm.layers import LayerNorm2d
from limo._src import convnext
from limo._src.torch_util import assert_equal_preds, torch2flax


class ConvNeXtTest(parameterized.TestCase):
    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_block(self, seed):
        rng, init_rng = jr.split(jr.PRNGKey(seed))
        inputs = jr.uniform(rng, (4, 64, 64, 16))
        torch_block = ConvNeXtBlock(16, conv_mlp=True, ls_init_value=None)
        flax_block = convnext.ConvNextBlock(16, ls_init_value=None)
        variables = flax_block.init(init_rng, inputs)
        variables = torch2flax(variables, torch_block)
        assert_equal_preds(
            flax_block,
            torch_block,
            variables,
            inputs,
        )

    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_stage(self, seed):
        rng, init_rng = jr.split(jr.PRNGKey(seed))

        inputs = jr.uniform(rng, (4, 64, 64, 16))

        torch_stage = ConvNeXtStage(
            16, 16, conv_mlp=True, ls_init_value=1e-6, norm_layer=LayerNorm2d
        )
        flax_stage = convnext.ConvNextStage(16, ls_init_value=1e-6)
        variables = flax_stage.init(init_rng, inputs)
        variables = torch2flax(variables, torch_stage)
        assert_equal_preds(
            flax_stage,
            torch_stage,
            variables,
            inputs,
        )

    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_stage_with_different_inC_outC(self, seed):
        rng, init_rng = jr.split(jr.PRNGKey(seed))

        inputs = jr.uniform(rng, (4, 64, 64, 16))

        torch_stage = ConvNeXtStage(
            16, 32, stride=2, conv_mlp=True, ls_init_value=None, norm_layer=LayerNorm2d
        )
        flax_stage = convnext.ConvNextStage(32, stride=2, ls_init_value=None)
        variables = flax_stage.init(init_rng, inputs)
        variables = torch2flax(variables, torch_stage)
        assert_equal_preds(
            flax_stage,
            torch_stage,
            variables,
            inputs,
        )

    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_stage_with_dilation(self, seed):
        rng, init_rng = jr.split(jr.PRNGKey(seed))

        inputs = jr.uniform(rng, (4, 64, 64, 16))

        torch_stage = ConvNeXtStage(
            16, 16, conv_mlp=True, ls_init_value=None, dilation=(2, 2), norm_layer=LayerNorm2d
        )
        flax_stage = convnext.ConvNextStage(16, dilation=(2, 2), ls_init_value=None)
        variables = flax_stage.init(init_rng, inputs)
        variables = torch2flax(variables, torch_stage)
        assert_equal_preds(
            flax_stage,
            torch_stage,
            variables,
            inputs,
        )

    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_convnext(self, seed):
        rng, init_rng = jr.split(jr.PRNGKey(seed))

        inputs = jr.uniform(rng, (4, 64, 64, 3))

        torch_model = ConvNeXt()
        flax_model = convnext.ConvNeXt()
        variables = flax_model.init(init_rng, inputs)
        variables = torch2flax(variables, torch_model)
        assert_equal_preds(
            flax_model,
            torch_model,
            variables,
            inputs,
        )


if __name__ == "__main__":
    absltest.main()
