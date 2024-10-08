import math

from tokenizers import Tokenizer
import torch
from torch import nn as nn


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.linear_2(self.dropout(self.linear_1(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "dimension must be divisible by number of heads"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        # (b, h, s, d_k) @ (b, h, d_k, s) -> (b, h, s, s)
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill(mask == 0, -1e9)
        # (b, h , sl, sl)
        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        # (b, h, s, s) @ (b, h, s, d_k) -> (b, h, s, d_k)
        return attention_score @ v

    def forward(self, q, k, v, mask):
        # (b, s, d)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # (b, h, s, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        # x (b, h, s, d_k), attention_score (b, h, s, s)
        x = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        # (b, s, d)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # (b, s, d)
        return self.w_o(x)


class Residual(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([Residual(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feedforward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feedforward
        self.residuals = nn.ModuleList([Residual(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residuals[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (b, s, d) -> (b, s, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_size)


class PositionalEncoding(nn.Module):

    # Positional encoding for transformer model
    # formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    #          PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    def __init__(self, embedding_size: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, embedding_size)
        # (seqlen,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1)

        # (embedding_size/2)
        deno = torch.exp(
            torch.arange(0, embedding_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / embedding_size)
        )

        positional_encoding[:, 0::2] = torch.sin(position * deno)
        positional_encoding[:, 1::2] = torch.cos(position * deno)

        # (1, seqlen, embedding_size)
        positional_encoding = positional_encoding.unsqueeze(dim=0)

        self.register_buffer("pe", positional_encoding)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbedding,
        tgt_embedding: InputEmbedding,
        srcPosision: PositionalEncoding,
        tgtPosition: PositionalEncoding,
        projection: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.projection = projection
        self.tgtPosition = tgtPosition
        self.srcPosision = srcPosision
        self.tgt_embedding = tgt_embedding
        self.src_embedding = src_embedding
        self.decoder = decoder
        self.encoder = encoder

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.srcPosision(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):

        tgt = self.tgt_embedding(tgt)
        tgt = self.tgtPosition(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seg_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
):

    src_embedding = InputEmbedding(src_vocab_size, d_model)
    tgt_embedding = InputEmbedding(tgt_vocab_size, d_model)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seg_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention, cross_attention, feed_forward, dropout
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer


def causal_mask(input_len: int):
    mask = torch.triu(torch.ones(1, input_len, input_len), diagonal=1)
    return mask == 0


def greedy_decode(
    model, source, source_mask, tokenizer_src: Tokenizer, tokenizer_tgt, max_len, device
):

    sos_token = tokenizer_src.token_to_id("[SOS]")
    eos_token = tokenizer_src.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_token).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])

        # print(f'valid: {out.shape} {prob.shape} {out[:,-1].shape}')

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_token:
            break

    return decoder_input.squeeze(0)


