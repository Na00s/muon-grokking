import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        sequence_length: int = 3,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        # Produces queries, keys, and values together.
        self.qkv_projection = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # Prevent each token from attending to future tokens.
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # B = examples, T = tokens, C = model dimension

        qkv = self.qkv_projection(x)  # (B, T, 3C)

        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, T, C)

        q = q.view(B, T, self.num_heads, self.head_size)  # (B, T, H, D)
        k = k.view(B, T, self.num_heads, self.head_size)  # (B, T, H, D)
        v = v.view(B, T, self.num_heads, self.head_size)  # (B, T, H, D)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, T, D)
        v = v.transpose(1, 2)  # (B, H, T, D)

        attention_scores = q @ k.transpose(-2, -1)  # (B, H, T, T)

        attention_scores = attention_scores / math.sqrt(self.head_size)

        mask = self.causal_mask[:T, :T]  # (T, T)

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )  # (B, H, T, T)

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )  # (B, H, T, T)

        output = attention_weights @ v  # (B, H, T, D)

        output = output.transpose(1, 2)  # (B, T, H, D)

        output = output.contiguous().view(B, T, C)  # (B, T, C)

        output = self.output_projection(output)  # (B, T, C)

        return output


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_mlp: int = 512,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            d_model,
            d_mlp,
            bias=False,
        )

        self.output_projection = nn.Linear(
            d_mlp,
            d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)  # (B, T, C) -> (B, T, M)

        x = F.relu(x)  # (B, T, M)

        x = self.output_projection(x)  # (B, T, M) -> (B, T, C)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_mlp: int = 512,
        sequence_length: int = 3,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            sequence_length=sequence_length,
        )

        self.mlp = MLP(
            d_model=d_model,
            d_mlp=d_mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)  # (B, T, C)

        x = x + self.mlp(x)  # (B, T, C)

        return x


class ModularAdditionTransformer(nn.Module):
    def __init__(
        self,
        modulus: int = 113,
        sequence_length: int = 3,
        d_model: int = 128,
        num_heads: int = 4,
        d_mlp: int = 512,
    ):
        super().__init__()

        self.modulus = modulus
        self.sequence_length = sequence_length

        # Tokens 0 through 112 represent numbers.
        # Token 113 represents "=".
        vocabulary_size = modulus + 1

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            sequence_length,
            d_model,
        )

        self.transformer_block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_mlp=d_mlp,
            sequence_length=sequence_length,
        )

        # Outputs one logit for each possible answer from 0 through 112.
        self.unembedding = nn.Linear(
            d_model,
            modulus,
            bias=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape  # B = examples, T = 3 tokens

        positions = torch.arange(
            T,
            device=token_ids.device,
        )  # (T,)

        token_embeddings = self.token_embedding(token_ids)  # (B, T, C)

        position_embeddings = self.position_embedding(positions)  # (T, C)

        x = token_embeddings + position_embeddings  # (B, T, C)

        x = self.transformer_block(x)  # (B, T, C)

        # The final token is the equals token, so we read its representation.
        final_token = x[:, -1, :]  # (B, C)

        logits = self.unembedding(final_token)  # (B, P), P = 113 answers

        return logits


if __name__ == "__main__":
    model = ModularAdditionTransformer()

    example_inputs = torch.tensor(
        [
            [80, 50, 113],
            [20, 30, 113],
        ],
        dtype=torch.long,
    )  # (B, T), B = 2 examples, T = 3 tokens

    logits = model(example_inputs)  # (B, P), B = 2, P = 113 possible answers

    print("Input shape:", example_inputs.shape)
    print("Logits shape:", logits.shape)
    print("Predictions:", logits.argmax(dim=-1))