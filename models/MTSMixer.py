import torch
import torch.nn as nn
from layers.RevIN import RevIN
from utils.decomposition import svd_denoise, NMF

class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        # self.dropouts = nn.ModuleList()
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else: x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling) :
        super().__init__()

        # assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling + int(input_dim % sampling > idx), mlp_dim) for idx in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C, configs.down_sampling_window, configs.use_norm) for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.enc_in) if configs.use_norm else None

        self.task_name = configs.task_name
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
            self.rev = RevIN(configs.enc_in) if configs.use_revin else None
        elif self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
        
    def forecast(self, x):
        x = self.rev(x, 'norm') if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, 'denorm') if self.rev else x

        return x
    

    def classification(self, x, x_mark_enc):
        for block in self.mlp_blocks:
            x = block(x)
        x = self.norm(x) if self.norm else x

        # referred from ETSformer classification ftn
        output = self.act(x)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        
        return output
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            out = self.forecast(x_enc)
        elif self.task_name == 'classification':  # referred from ETSformer classification ftn
            out = self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f'Invalid task name: {self.task_name}')
        return out