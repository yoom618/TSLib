import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding
from mamba_ssm import Mamba, Mamba2
from layers.MambaBlock import Mamba_TimeVariant
SSM_LAYER_LIST = ["Mamba", "Mamba2", "Mamba_TimeVariant"]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, d_kernel=3):
        super(TokenEmbedding, self).__init__()
        padding = d_kernel - 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=d_kernel, padding=padding, padding_mode='replicate', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, seqlen):
        x = self.tokenConv(x.permute(0, 2, 1))[..., :seqlen].transpose(1, 2)
        return x


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # self.embedding = DataEmbedding(configs.enc_in, configs.d_model, 
        #                                configs.embed, configs.freq, configs.dropout)
        self.embedding = TokenEmbedding(configs.enc_in, configs.d_model, configs.num_kernels)
        self.embedding_pos = PositionalEmbedding(configs.d_model, max_len=configs.seq_len)
        self.dropout = nn.Dropout(configs.dropout)
        
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model = configs.d_model,
                d_state = configs.d_ff,
                d_conv = configs.d_conv,
                expand = configs.expand,
                timevariant_dt= bool(configs.tv_dt),
                timevariant_B= bool(configs.tv_B),
                timevariant_C= bool(configs.tv_C),
                device = configs.device,
            ),
            nn.LayerNorm(configs.d_model)
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
            # mamba_in = self.embedding(x_enc)  # (B, L_in, D)
            mamba_in = self.embedding(x_enc, self.seq_len) + self.embedding_pos(x_enc)  # (B, L_in, D)
            mamba_out = self.mamba(mamba_in)  # (B, L_in, D)
            mamba_out = self.dropout(mamba_out)  # (B, L_in, D). 이 편이 성능이 더 많이 오름

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
        