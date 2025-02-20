import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from mamba_ssm import Mamba, Mamba2
from layers.MambaBlock import Mamba_TimeVariant
SSM_LAYER_LIST = ["Mamba", "Mamba2", "Mamba_TimeVariant"]


class TokenEmbedding_modified(nn.Module):
    def __init__(self, c_in, d_model, d_kernel=3):  # original TokenEmbedding use d_kernel=3
        super().__init__()
        padding = d_kernel - 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=d_kernel, padding=padding, padding_mode='replicate', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        seq_len = x.size(1)
        x = self.tokenConv(x.permute(0, 2, 1))[..., :seq_len].transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1, d_kernel=3, temporal_emb=False):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding_modified(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        if temporal_emb:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
                if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.temporal_embedding is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)




class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        if self.task_name in ['classification', 'anomaly_detection']:
            self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.seq_len,
                                           configs.embed, configs.freq, 
                                           configs.dropout, configs.num_kernels, False)
        elif self.task_name in ['short_term_forecast', 'long_term_forecast']:
            self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.seq_len,
                                           configs.embed, configs.freq, 
                                           configs.dropout, configs.num_kernels, True)
            
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model = configs.d_model,
                d_state = configs.d_ff,
                d_conv = configs.d_conv,
                expand = configs.expand,
                timevariant_dt= bool(configs.tv_dt),    # only available in Mamba_TimeVariant
                timevariant_B= bool(configs.tv_B),      # only available in Mamba_TimeVariant
                timevariant_C= bool(configs.tv_C),      # only available in Mamba_TimeVariant
                device = configs.device,
            ),
            nn.LayerNorm(configs.d_model),
            nn.Dropout(configs.dropout)
        )
        
        if self.task_name in ['classification']:  # one class per one sequence
            self.out_layer = nn.Linear(configs.d_model, configs.num_class, bias=False)
            self.impact_factor = nn.Linear(configs.d_model, 1, bias=True)
            self.impact_factor.weight.data.fill_(0.0)
            self.impact_factor.bias.data.fill_(1.0)
            
        elif self.task_name in ['anomaly_detection', 'short_term_forecast', 'long_term_forecast']:
            self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['anomaly_detection']:
            # seq_len(L_in)  equals to  pred_len(L_out)
            mamba_in = self.embedding(x_enc, None)  # actually, x_mark_enc is None. (B, L, D)
            mamba_out = self.mamba(mamba_in)  # (B, L, C_out)

            return mamba_out


        elif self.task_name in ['classification']:
            mamba_in = self.embedding(x_enc, None)  # (B, L_in, D)
            mamba_out = self.mamba(mamba_in)  # (B, L_in, D)

            ### 1) use the last hidden state to make the final prediction
            # out = mamba_out[:, -1, :]  # (B, D)
            # out = self.out_layer(x_out)  # (B, D) -> (B, C_out)
            
            ### 2) use the average of the hidden states to make the final prediction
            # out = self.out_layer(mamba_out)  # (B, L_in, D) -> (B, L_in, C_out)
            # out = out * x_mark_enc.unsqueeze(2)  # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
            # out = out.mean(1)  # (B, C_out)
            
            ### 3) use the average of the hidden states to make the final prediction
            # softmax 계열이 아닌 sigmoid와 같은 단독 활성화 함수를 사용할 경우, 오버피팅이 발생함. 이는 학습 과정에서 지나치게 많은 timestamp 정보를 0으로 만들어서 학습데이터의 극히 일부만 사용하게 만들기 때문으로 추정.
            logit_out = self.out_layer(mamba_out)  # (B, L_in, D/2) -> (B, L_in, C_out)
            w_out = F.softmax(self.impact_factor(mamba_out).squeeze(2), dim=1)  # (B, L_in, D) -> (B, L_in, 1) -> (B, L_in)
            # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
            logit_out = logit_out * x_mark_enc.unsqueeze(2)  # (B, L_in, C_out)
            w_out = w_out * x_mark_enc  
            w_out = w_out / w_out.sum(1, keepdim=True)
            # calculate the weighted average of the hidden states to make the final prediction
            out = logit_out * w_out.unsqueeze(2)  # (B, L_in, C_out)
            out = out.sum(1)  # (B, C_out)
            # to reduce the gradient for w_out(impact factor), we use simple trick as follows.
            if self.training:
                prob = torch.rand(1, device=out.device)
                out = prob * out + (1 - prob) * logit_out.mean(1)  # (B, C_out)

            return out



        elif self.task_name in ['short_term_forecast', 'long_term_forecast']:
            # x_enc : (B, L_in, C_in)
            mamba_in = self.embedding(x_enc, x_mark_enc)  # (B, L_in, D)
            mamba_outs = [mamba_in]
            for i in range((self.pred_len - 1) // self.seq_len + 1):
                mamba_out = self.mamba(mamba_outs[-1])
                mamba_outs.append(mamba_out)
            mamba_out = torch.cat(mamba_outs[1:], dim=1)[:,:self.pred_len,:]  # (B, L_out, D)
            x_out = self.out_layer(mamba_out)  # (B, L_out, D) -> (B, L_out, C_out)

            return x_out
        