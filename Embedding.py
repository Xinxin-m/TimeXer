import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, N = x.shape  # [B, L, N]
        glb = self.glb_token.repeat((B, 1, 1, 1))  # [B, 1, 1, d_model]
        # Permute to get [B, N, L] before splitting into patches
        # num_patches = L // self.patch_len
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, N, num_patches, patch_len]
        x = torch.reshape(x, (B * N, x.shape[2], x.shape[3]))  # [B*N, num_patches, patch_len]
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)  # [B*N, num_patches, d_model]
        x = torch.reshape(x, (-1, N, x.shape[-2], x.shape[-1]))  # [B, N, num_patches, d_model]
        x = torch.cat([x, glb], dim=2)  # [B, N, num_patches+1, d_model]
        x = torch.reshape(x, (B * N, x.shape[2], x.shape[3]))  # [B*N, num_patches+1, d_model]
        
        return self.dropout(x), N
    
    
class DataEmbedding(nn.Module): # for exogeneous variables
    # self.ex_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.timestep, configs.dropout)

    def __init__(self, d_in, d_model, timestep=3600, dropout=0.1, use_datetime=False):
        super(DataEmbedding, self).__init__()
        self.timestep = timestep
        self.value_embedding = nn.Linear(d_in, d_model) # nn.Linear(in_features, out_features)
        self.time_embedding = nn.Linear(d_in, d_model) # learn a different linear embedding for time
        self.dropout = nn.Dropout(p=dropout)
        self.use_datetime = use_datetime

    def forward(self, x, x_mark): # x_mark is the timestamps
        # x: [batch_size, max_len, num_var]:= [B,L,N]
        # [B,L,N] -> [B,N,L]
        x = x.permute(0, 2, 1)
        # x_mark is [B,L] if not use_datetime, [B,L,N] if use_datetime
        
        if self.use_datetime:
            x_mark = x_mark.permute(0, 2, 1) # [B,L,N] -> [B,N,L]
            # x = torch.cat([x, datetime.permute(0, 2, 1)], 1) # [B,N,L] -> [B,N+N_time,L]
        # Ensure x_mark are integers (torch changed integer to floats when importing)
        # x_mark = x_mark.long() # not needed since we aren't using TimeEmbedding
        x = self.value_embedding(x) + self.time_embedding(x_mark) # map each variable seq of length L onto d_model
        return self.dropout(x)
    

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_window=2048, seq_len=None):
        super(TimeEmbedding, self).__init__()
        if seq_len is None:
            seq_len = 5000 # used 2*seq_len in frequency 
        position = torch.arange(0, max_window).float().unsqueeze(1)  # shape: (max_window, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(2*seq_len) / d_model))  # (2L)^(-i/d_model) for i in [0, d_model/2]

        pe = torch.zeros(max_window, d_model)  # shape: (max_window, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (non-trainable, saved in state_dict)
        self.register_buffer("pe", pe)

    def forward(self, t):
        """
        t: Tensor of shape (batch_size, seq_len) containing timestamp indices
        Returns: Tensor of shape (batch_size, seq_len, d_model)
        """
        # t should have been normalized to start at 0 for each sample in batch
        # t = (t - t[:, 0].unsqueeze(1)) / self.timestep  # this is done in data_loader_dt.py
        return self.pe[t]

class HourEmbedding(nn.Module):
    def __init__(self, d_model, num_hours=24):
        super(HourEmbedding, self).__init__()
        # Precompute the embedding matrix for 24 hours
        w = torch.zeros(num_hours, d_model).float()
        w.require_grad = False

        # Compute frequencies: 2^(i/half_dim) for i in [0, half_dim-1]
        half_dim = d_model // 2
        freqs = torch.arange().float() / half_dim
        freqs = 2 ** freqs  # Exponential frequencies: 2^0, 2^1, ...

        # Compute cyclical embeddings: sin(2πh/24 * f_i) and cos(2πh/24 * f_i)
        hours = torch.arange(num_hours).float().unsqueeze(-1)  # Shape: (24, 1)
        hourly_emb = 2 * math.pi * hours * freqs.unsqueeze(0) / 24.0  # Shape: (24, half_dim)
        w[:, 0::2] = torch.sin(hourly_emb)
        w[:, 1::2] = torch.cos(hourly_emb)

        # Register the precomputed embeddings as a buffer
        self.register_buffer('hour_emb', w)

    def forward(self, hours):
        """
        Args:
            hours: Tensor of shape (batch_size, seq_len) with hour indices (0-23)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with cyclical embeddings
        """
        return self.hour_emb[hours]

class ContinuousHourEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        half_dim = d_model // 2
        freqs = torch.arange(half_dim).float() / half_dim
        self.register_buffer("hour_freqs", 2 ** freqs * (2 * math.pi / 24))  # scale by 2π/24

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (batch_size, seq_len) with float hours in [0, 24)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        angles = t.float().unsqueeze(-1) * self.hour_freqs  # shape: (batch, seq_len, half_dim)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
