from __future__ import annotations

import numpy
import jax.random as jr
import jax.numpy as jnp
import torch
import torch.nn as nn
from torchvision import models as torch_models
from absl.testing import absltest, parameterized

from limo._src import swin_transformer as flax_models
from limo._src.test_utils import assert_close_outputs, assert_computable


class SwinTransformerTest(parameterized.TestCase):
    def test_swin_v1_relative_position_bias_index(self):
        from einops import reduce, rearrange

        window_size = (7, 7)

        # PyTorch.
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        x_torch = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww

        # JAX
        coords_h = jnp.arange(window_size[0])
        coords_w = jnp.arange(window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = rearrange(coords, "n size_h size_w -> n (size_h size_w)")
        relative_coords = rearrange(
            coords_flatten[:, :, None] - coords_flatten[:, None, :],
            "n size1 size2 -> size1 size2 n",
        )
        relative_coords = relative_coords.at[:, :, 0].add(window_size[0] - 1)
        relative_coords = relative_coords.at[:, :, 1].add(window_size[1] - 1)
        relative_coords = relative_coords.at[:, :, 0].multiply(2 * window_size[1] - 1)
        x_jax = reduce(relative_coords, "size1 size2 n -> (size1 size2)", "sum")

        numpy.testing.assert_allclose(x_jax, x_torch, atol=5e-5)

    def test_swin_patch_merging_pad(self):
        from torchvision.models.swin_transformer import _patch_merging_pad

        x = torch.rand((4, 32, 32, 128))
        x_torch = _patch_merging_pad(x)

        x = jnp.array(x.detach().cpu().numpy())
        x_jax = flax_models._patch_merging_pad(x)
        numpy.testing.assert_allclose(x_jax, x_torch, atol=5e-5)

    def test_swin_get_relative_position_bias(self):
        from torchvision.models.swin_transformer import _get_relative_position_bias

        window_size = (7, 7)
        num_heads = 8

        table = torch.rand((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww

        x_torch = _get_relative_position_bias(table, index, window_size)

        table = jnp.array(table.detach().cpu().numpy())
        index = jnp.array(index.detach().cpu().numpy())
        x_jax = flax_models._get_relative_position_bias(table, index, window_size)

        numpy.testing.assert_allclose(x_jax, x_torch, atol=5e-5)

    def test_swin_shifted_window_attention(self):
        from torchvision.models import swin_transformer

        window_size = (7, 7)
        num_heads = 1
        shift_size = [x // 2 for x in window_size]
        attention_dropout = 0.0
        dropout = 0

        in_ch = 16
        hidden_dim = 16
        batch_size = 3

        input = torch.rand((batch_size, 64, 64, in_ch))  # (N, H, W, C)
        qkv_weight = torch.rand((3 * num_heads * hidden_dim, in_ch))  # (outC, inC)
        qkv_bias = torch.rand((3 * num_heads * hidden_dim,))
        proj_weight = torch.rand((in_ch, num_heads * hidden_dim))  # (outC, inC)
        proj_bias = torch.rand((in_ch,))

        relative_position_bias = torch.rand(
            (num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1])
        )

        x_torch = swin_transformer.shifted_window_attention(
            input,
            qkv_weight=qkv_weight,
            proj_weight=proj_weight,
            relative_position_bias=relative_position_bias,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=shift_size,
            attention_dropout=attention_dropout,
            dropout=dropout,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )

        input = jnp.array(input.detach().cpu().numpy())
        qkv_weight = jnp.transpose(jnp.array(qkv_weight.detach().cpu().numpy()), axes=(1, 0))
        qkv_bias = jnp.array(qkv_bias.detach().cpu().numpy())
        proj_weight = jnp.transpose(jnp.array(proj_weight.detach().cpu().numpy()), axes=(1, 0))
        proj_bias = jnp.array(proj_bias.detach().cpu().numpy())
        relative_position_bias = jnp.array(relative_position_bias.detach().cpu().numpy())

        x_jax = flax_models.shifted_window_attention(
            input,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            relative_position_bias,
            window_size,
            shift_size,
            num_heads,
            attention_dropout,
            dropout,
            deterministic=True,
        )

        numpy.testing.assert_allclose(x_jax, x_torch, atol=5e-5)

    def test_swin_shifted_window_attention_module(self):
        from torchvision.models import swin_transformer

        torch_model = swin_transformer.ShiftedWindowAttention(
            dim=16,
            window_size=[7, 7],
            shift_size=[3, 3],
            num_heads=1,
            qkv_bias=True,
            proj_bias=True,
        )
        flax_model = flax_models.ShiftedWindowAttention((7, 7), (3, 3), 1, True, True)

        assert_close_outputs(
            jr.PRNGKey(0), flax_model, torch_model, input_size=(64, 64, 16), tensor_transpose_axes=None
        )

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

    def test_swin_v2_t(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_v2_t(num_classes=num_classes)
            torch_model = torch_models.swin_v2_t(weights=torch_models.Swin_V2_T_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_swin_v2_s(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_v2_s(num_classes=num_classes)
            torch_model = torch_models.swin_v2_s(weights=torch_models.Swin_V2_S_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)

    def test_swin_v2_b(self):
        for num_classes in [0, 1000]:
            flax_model = flax_models.swin_v2_b(num_classes=num_classes)
            torch_model = torch_models.swin_v2_b(weights=torch_models.Swin_V2_B_Weights.DEFAULT)
            if num_classes == 0:
                torch_model.head = nn.Identity()
            assert_computable(flax_model, is_training=True)
            assert_computable(flax_model, is_training=False)
            assert_close_outputs(jr.PRNGKey(0), flax_model, torch_model)


if __name__ == "__main__":
    absltest.main()
