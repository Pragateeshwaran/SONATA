import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass



@dataclass
class Configuration:
    n_encoder_layers: int = 8
    n_decoder_layers: int = 16
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_length: int = 1500
    vocab_size: int = 51876
    n_conv_channels: list = (80, 256, 512, 1024)
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class EncoderLayer(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.Q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.V = nn.Linear(config.d_model, config.d_model, bias=False)
        self.K = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)
        self.activation = nn.SiLU()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.ffn_layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q, k, v = self.Q(x), self.K(x), self.V(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.dropout(self.Out(attn_output))
        x = self.attn_layer_norm(x + attn_output)

        # Feed-forward network
        ffn_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = self.ffn_layer_norm(x + ffn_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention Link 
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward network
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class SONATA_Encoder(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.conv1 = nn.Conv1d(config.n_conv_channels[0], config.n_conv_channels[1], 3, 1, 1)
        self.conv2 = nn.Conv1d(config.n_conv_channels[1], config.n_conv_channels[2], 3, 1, 1)
        self.conv3 = nn.Conv1d(config.n_conv_channels[2], config.n_conv_channels[3], 3, 2, 1)
        self.projection = nn.Linear(config.n_conv_channels[3], config.d_model)
        self.pos_embed = PositionalEncoding(config.max_seq_length, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)

class SONATA_Decoder(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.max_seq_length, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed_tokens(x)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)  # Change to (seq_len, batch_size, d_model)
        encoder_output = encoder_output.transpose(0, 1)  # Change to (seq_len, batch_size, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        x = x.transpose(0, 1)  # Change back to (batch_size, seq_len, d_model)
        return self.layer_norm(x)

class SONATA(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.Encoder = SONATA_Encoder(self.config)
        self.Decoder = SONATA_Decoder(self.config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        encoder_output = self.Encoder(src)
        decoder_output = self.Decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.proj_out(decoder_output)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

 
config = Configuration()
model = SONATA(config)
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device) 


print(f"The model has {count_parameters(model):,} trainable parameters")

# Dummy input
src = torch.randn(32, 80, 1000)  # (batch_size, features, time_steps)
tgt = torch.randint(0, config.vocab_size, (32, 100))  # (batch_size, seq_len)
src, tgt = src.to(device), tgt.to(device)
output = model(src=src, tgt=tgt)
print(output.shape)  # Should be (32, 100, 51876)
