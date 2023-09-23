import torch
import torch.nn as nn
import numpy as np
from custom_layers.attention import Aspect_Attention_op2
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HiPAMA(nn.Module):
    def __init__(self, embed_dim, depth, input_dim=84, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.conv_dim = 25

        # phone projection
        self.phn_proj = nn.Linear(40, embed_dim)

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)

        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=depth, batch_first=True)
        self.attn = Attention(
            embed_dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.) # Multi-head self-attention
        self.conv = torch.nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=5, padding=2) # seq_len, 

        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification
        self.rep_w1 = nn.Linear(embed_dim, embed_dim)
        self.rep_w2 = nn.Linear(embed_dim, embed_dim)
        self.rep_w3 = nn.Linear(embed_dim, embed_dim)

        self.attn_tmp = Aspect_Attention_op2(embed_dim)
        
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.w_attn = Attention(
            embed_dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.)

        # utterance level
        self.rep_utt1 = nn.Linear(embed_dim, embed_dim)
        self.rep_utt2 = nn.Linear(embed_dim, embed_dim)
        self.rep_utt3 = nn.Linear(embed_dim, embed_dim)
        self.rep_utt4 = nn.Linear(embed_dim, embed_dim)
        self.rep_utt5 = nn.Linear(embed_dim, embed_dim)
        self.utt_attn_tmp = Aspect_Attention_op2(embed_dim)
        
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    # get the output of the last valid token
    def get_last_valid(self, input, mask):
        output = []
        B = input.shape[0]
        seq_len = input.shape[1]
        for i in range(B):
            for j in range(seq_len):
                if mask[i, j] == 0:
                    output.append(input[i, j-1])
                    break
                if j == seq_len - 1:
                    print('append')
                    output.append(input[i, j])
        output = torch.stack(output, dim=0)
        return output.unsqueeze(1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn):

        # batch size
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn>=0)

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot =  torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        
        x = self.attn(x)
        
        x = self.conv(x.transpose(1,2)) # x.transpose in shape [batch, feat_dim, seq_len]
        x = x.transpose(1,2)

        ### Fisrt output phn score 
        p = self.mlp_head_phn(x).reshape(B, seq_len, 1) 
        
        ### Second output word score with aspects attention
        w1 = self.rep_w1(x) 
        w2 = self.rep_w2(x)
        w3 = self.rep_w3(x)
        
        w_list = (w1, w2, w3)
        w_attns = []
        for i in range(len(w_list)):
            target_w = w_list[i]
            non_target_w = torch.cat((w_list[:i] + w_list[i+1:]), dim=1)
            w_attn = self.attn_tmp(target_w, non_target_w)
            w = target_w + w_attn
            w_attns.append(w)

        w1 = self.mlp_head_word1(w_attns[0]).reshape(B, seq_len, 1)
        w2 = self.mlp_head_word2(w_attns[1]).reshape(B, seq_len, 1)
        w3 = self.mlp_head_word3(w_attns[2]).reshape(B, seq_len, 1)
        
        ### Third output utterance score using words representation
        rep = (w_attns[0] + w_attns[1] + w_attns[2]) / 3
        rep = self.w_attn(rep)
        
        u1 = self.rep_utt1(rep)
        u2 = self.rep_utt2(rep)
        u3 = self.rep_utt3(rep)
        u4 = self.rep_utt4(rep)
        u5 = self.rep_utt5(rep)
        
        utt_list = (u1, u2, u3, u4, u5)
        utt_attns = []
        for i in range(len(utt_list)):
            target_utt = utt_list[i]
            non_target_utt = torch.cat((utt_list[:i] + utt_list[i+1:]), dim=1)
            utt_attn = self.utt_attn_tmp(target_utt, non_target_utt)
            utt = target_utt + utt_attn
            utt_attns.append(utt)
        

        u1 = self.get_last_valid(self.mlp_head_utt1(utt_attns[0]).reshape(B, seq_len), valid_tok_mask)
        u2 = self.get_last_valid(self.mlp_head_utt2(utt_attns[1]).reshape(B, seq_len), valid_tok_mask)
        u3 = self.get_last_valid(self.mlp_head_utt3(utt_attns[2]).reshape(B, seq_len), valid_tok_mask)
        u4 = self.get_last_valid(self.mlp_head_utt4(utt_attns[3]).reshape(B, seq_len), valid_tok_mask)
        u5 = self.get_last_valid(self.mlp_head_utt5(utt_attns[4]).reshape(B, seq_len), valid_tok_mask)

        return u1, u2, u3, u4, u5, p, w1, w2, w3