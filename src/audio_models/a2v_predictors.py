# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Different types of a2v predictors that take audio features and predict video features.
"""

import torch
from torch import nn

from src.models.attentive_pooler import AttentivePooler


class A2V2_MLP_Predictor(nn.Module):
    """
    Takes tokens of one modality
    Pools them.
    Uses an MLP to predict pooled features of other modality.
    """

    def __init__(
        self,
        mod1_feature_dim=768,
        mod2_feature_dim=768,
        pooling="mean",
        mlp_ratio=2,
        act_layer=nn.GELU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.mod1_feature_dim = mod1_feature_dim
        self.mod2_feature_dim = mod2_feature_dim
        self.hidden_features = mod1_feature_dim * mlp_ratio
        self.pooling = pooling

        if norm_layer is not None:
            self.predictor_norm = norm_layer(self.mod2_feature_dim)
        else:
            self.predictor_norm = None

        if self.pooling == "attentive":
            self.pooler = AttentivePooler(
                num_queries=1,
                embed_dim=mod1_feature_dim,
                num_heads=4,
                complete_block=True,  # adds a residual MLP branch; KQV stay learnable either way
            )

        self.fc1 = nn.Linear(self.mod1_feature_dim, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.fc3 = nn.Linear(self.hidden_features, self.mod2_feature_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x shape [bs, num_tokens, mod1_feature_dim]
        if self.pooling == "mean":
            x = torch.mean(x, dim=1)  # shape [bs, mod1_feature_dim]
        elif self.pooling == "attentive":
            x = self.pooler(x)  # shape [bs, num_of_query_tokens, mod1_feature_dim]
            x = torch.mean(x, dim=1)  # shape [bs, mod1_feature_dim]

        # NOW MLP predictor on top of pooled input features
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        if self.predictor_norm is not None:
            x = self.predictor_norm(x)

        return x


class MLPPredictor(nn.Module):
    def __init__(
        self,
        in_features,
        mlp_ratio=2,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = in_features * mlp_ratio

        if norm_layer is not None:
            self.predictor_norm = norm_layer(self.out_features)
        else:
            self.predictor_norm = None

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.fc3 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        if self.predictor_norm is not None:
            x = self.predictor_norm(x)

        return x
