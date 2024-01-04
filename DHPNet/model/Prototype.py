import datetime

import torch
import torch.autograd as ag
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from .layers import *
from .coordatt import CoordAtt

class Prototype(nn.Module):
    def __init__(self, proto_size, feature_dim, proto_dim, shink_thres=0, dropout=0.0):
        super(Prototype, self).__init__()
        #Constants
        self.proto_size = proto_size
        self.proto_dim = proto_dim
        self.feature_dim = feature_dim
        self.shink_thres = shink_thres

        self.w_q = nn.Linear(feature_dim, proto_size * proto_dim)
        self.w_k = nn.Linear(feature_dim, proto_size * proto_dim)
        self.w_v = nn.Linear(feature_dim, proto_size * proto_dim)

        self.linear_out = nn.Linear(proto_size * proto_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)
        self.instance_norm = nn.InstanceNorm2d(feature_dim)

    def inter_head_ortho_loss(self, proto_weights, r):
        """
        :param proto_weights: tensor of shape (batch_size, proto_size, h*w, h*w)
        :param r: the constant r
        """
        ortho_loss = 0
        num_heads = proto_weights.size(1)
        I = torch.eye(proto_weights.size(2)).to(proto_weights.device)
        for b in range(proto_weights.size(0)):
            for head in range(proto_weights.size(1)):
                P = proto_weights[b, head, :, :]  # shape: (h*w, h*w)
                PPT = torch.matmul(P, P.t())  # shape: (h*w, h*w)
                ortho_loss += torch.norm(PPT - r ** 2 * I, p='fro')
        return ortho_loss

    def inter_head_separation_loss(self, proto_weights):
        """
        :param proto_weights: tensor of shape (batch_size, proto_size, h*w, h*w)
        """
        batch_size, proto_size, seq_len, _ = proto_weights.size()

        # Normalize proto_weights to make each vector a unit vector
        proto_weights_normalized = proto_weights / torch.norm(proto_weights, dim=-1, keepdim=True)

        # Calculate the cosine similarity between all pairs of prototypes
        similarity_matrix = torch.einsum("bpld,bpjd->bpjl", proto_weights_normalized, proto_weights_normalized)

        # Remove the diagonal elements (similarity of a prototype with itself) by setting them to -1
        eye_matrix = torch.eye(seq_len).to(proto_weights.device).unsqueeze(0).unsqueeze(0) * 2
        eye_matrix = eye_matrix.expand(batch_size, proto_size, seq_len, seq_len)
        similarity_matrix = similarity_matrix - eye_matrix

        # Calculate the separation loss
        separation_loss = torch.sum(similarity_matrix) / (batch_size * seq_len * proto_size * (proto_size - 1))

        return separation_loss


    def forward(self, key, q, k, v, mode="train"):
        batch_size, dims, h, w = q.size()  # b * d * h * w

        q = q.view(batch_size, h * w, dims)
        k = k.view(batch_size, h * w, dims)
        v = v.view(batch_size, h * w, dims)
        # transform qkv to multiheads
        q = self.w_q(q).view(batch_size, -1, self.proto_size, self.proto_dim).transpose(1, 2)   #  (batch_size, hw, proto_size, proto_dim) -> (batch_size, proto_size, h*w, proto_dim)
        k = self.w_k(k).view(batch_size, -1, self.proto_size, self.proto_dim).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.proto_size, self.proto_dim).transpose(1, 2)

        # calculate multi_feature_space weights using scaled dot product
        proto_weights = self.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.proto_dim))   #  (batch_size, proto_size, hw, hw)
        # proto_weights = self.dropout(proto_weights)
        output = proto_weights @ v

        output = output.transpose(1, 2).contiguous()  # (batch_size, h*w, proto_size, proto_dim)
        output = output.view(batch_size, h * w, -1)  # (batch_size, h*w, proto_size * proto_dim)

        # Combine multi-head features using a linear layer
        output = self.linear_out(output)  # (batch_size, h*w, feature_dim)
        output = output.view(batch_size, h, w, -1)  # (batch_size, h, w, feature_dim)
        output = output.permute(0, 3, 1, 2).contiguous()  # (batch_size, feature_dim, h, w)


        if mode == "train":
            separation_loss = self.inter_head_separation_loss(proto_weights)
            ortho_loss = self.inter_head_ortho_loss(proto_weights, 1)

            output = self.instance_norm(output + key)

            return output, separation_loss, ortho_loss

        else:
            output = self.instance_norm(output + key)

            return output