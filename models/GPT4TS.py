import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from layers.Embed import DataEmbedding


def set_huggingface_cache_dir(cache_dir):
    os.system(f'export HF_HOME={cache_dir}')    # !export HF_HOME = cache_dir
    os.environ["HF_HOME"] = cache_dir
    # os.system('huggingface-cli whoami')         # !huggingface-cli whoami
    
    with open(os.path.join(cache_dir, 'token'), 'r') as f:
        token = f.read().strip()

    return token


class Model(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.feat_dim = configs.enc_in
        self.num_classes = configs.num_class

        self.gpt_layers = configs.e_layers   # default was 6 in One-fits-all. it should be <= 12 since pre-trained GPT2 has 12 layers
        self.d_model = configs.d_model  # d_model should be fixed to 768 since it will be used in pre-trained GPT2
        self.d_ff = configs.d_ff  # d_ff should be smaller than d_model(=768)

        if self.task_name != 'anomaly_detection':
            if self.task_name == 'classification':
                self.patch_size = configs.patch_size
                self.stride = configs.patch_stride
                self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
                self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
                self.patch_num += 1
            else:
                self.patch_size = 1
                self.stride = 1
                self.patch_num = self.seq_len
            self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, configs.d_model, configs.dropout)

        self.token = set_huggingface_cache_dir(configs.huggingface_cache_dir)
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_hidden_states=True,
                                              cache_dir=configs.huggingface_cache_dir, token=self.token)
        self.gpt2.wte = None
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out)
        elif self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.ln_proj = nn.LayerNorm(configs.d_model * self.patch_num)
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.num_class)

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None


    def classification(self, x_enc):
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x, None)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs
    

    def imputation(self, x_enc, x_mark_enc, mask):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out


    def forecast(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = self.out_layer(dec_out)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = outputs[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * (stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = dec_out + (means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        return dec_out