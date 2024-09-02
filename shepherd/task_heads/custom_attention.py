import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoderLayer

import numpy as np
from scipy.stats import rankdata


# !!!!!! THIS IS CHANGED !!!!!!
# THIS IS CHANGED - premtc: GPT GENERATED
# !!!!!! THIS IS CHANGED !!!!!!

class CosineAttention:
    """
    Computes attention between a vector and a matrix using cosine similarity.
    """

    @staticmethod
    def tiny_value_of_dtype(dtype):
        """
        Returns a small value to avoid division by zero. The value is chosen depending on the dtype.
        """
        return torch.finfo(dtype).eps

    def forward(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention scores between the input vector and each row of the matrix using cosine similarity.

        Args:
            vector (torch.Tensor): A tensor of shape (batch_size, dim) representing the vector.
            matrix (torch.Tensor): A tensor of shape (batch_size, num_elements, dim) representing the matrix.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_elements) containing the attention scores.
        """

        # Normalize the vector
        a_norm = vector / (
            vector.norm(p=2, dim=-1, keepdim=True) + self.tiny_value_of_dtype(vector.dtype)
        )

        # Normalize the matrix
        b_norm = matrix / (
            matrix.norm(p=2, dim=-1, keepdim=True) + self.tiny_value_of_dtype(matrix.dtype)
        )

        # Compute cosine similarity using batch matrix-matrix multiplication
        return torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)


class BilinearAttention(nn.Module):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.
    This function has a matrix of weights `W` and a bias `b`, and the similarity between
    the vector `x` and the matrix `y` is computed as `x^T W y + b`.

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`. This is `x.size()[-1]`.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`. This is `y.size()[-1]`.
    activation : `callable`, optional (default=`None`)
        An activation function applied after the `x^T W y + b` calculation. Default is
        linear (i.e., no activation).
    normalize : `bool`, optional (default=`True`)
        If true, normalizes the computed similarities with a softmax to return a probability
        distribution for the attention. If false, computes a similarity score.
    """

    def __init__(
        self,
        vector_dim: int,
        matrix_dim: int,
        activation: callable = None,
        normalize: bool = True,
    ) -> None:
        super(BilinearAttention, self).__init__()
        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation if activation is not None else lambda x: x
        self._normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # Compute the bilinear attention score x^T W y + b
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        scores = intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias

        # Apply activation function
        scores = self._activation(scores)

        # Optionally normalize the scores with softmax
        if self._normalize:
            scores = F.softmax(scores, dim=-1)

        return scores

class AdditiveAttention(nn.Module):
    """
    Computes attention between a vector and a matrix using an additive attention function.
    This function has two matrices `W`, `U`, and a vector `V`. The similarity between
    the vector `x` and the matrix `y` is computed as `V^T tanh(Wx + Uy)`.

    This attention is often referred to as concat or additive attention. It was introduced
    in "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al. (2015).

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector `x`. This is `x.size()[-1]` - the length of the vector that will go into the similarity computation.
    matrix_dim : `int`, required
        The dimension of the matrix `y`. This is `y.size()[-1]` - the length of the vector that will go into the similarity computation.
    normalize : `bool`, optional (default=`True`)
        If true, normalizes the computed similarities with a softmax to return a probability distribution for your attention. If false, computes a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super(AdditiveAttention, self).__init__()
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self._normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)

    def forward(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # Compute the attention scores using the additive attention mechanism
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)
        scores = intermediate.matmul(self._v_vector).squeeze(2)

        # Optionally normalize the scores with softmax
        if self._normalize:
            scores = torch.softmax(scores, dim=-1)

        return scores

class DotProductAttention(nn.Module):
    """
    Computes attention between a vector and a matrix using dot product.

    Reference: "Attention Is All You Need" by Vaswani et al. (2017)
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the dot product attention between a vector and a matrix.

        Parameters:
        vector: torch.Tensor
            A tensor of shape (batch_size, vector_dim) representing the vector.
        matrix: torch.Tensor
            A tensor of shape (batch_size, num_elements, vector_dim) representing the matrix.

        Returns:
        torch.Tensor
            A tensor of shape (batch_size, num_elements) containing the attention scores.
        """
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)

# !!!!!! THIS IS CHANGED !!!!!!
# THIS IS CHANGED - premtc: GPT GENERATED
# !!!!!! THIS IS CHANGED !!!!!!