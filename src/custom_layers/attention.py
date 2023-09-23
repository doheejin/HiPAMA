import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class Aspect_Attention_op2(nn.Module):
    def __init__(self, embed_dim, op='none', activation='tanh', init_stdev=0.01):
        super().__init__()
        self.supports_masking = True
        assert op in {'attsum', 'attmean','none'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self._reset_parameters(embed_dim)

    def _reset_parameters(self, embed_dim):
        init_val_v = (torch.randn(embed_dim) * self.init_stdev)
        self.att_v = nn.Parameter(init_val_v)
        init_val_W = (torch.randn(embed_dim, embed_dim) * self.init_stdev)
        self.att_W = nn.Parameter(init_val_W)
        self.built = True
        
    def forward(self, x, x2, mask=None):
        
        y = torch.matmul(x2, self.att_W)
        
        if not self.activation:
            weights = torch.tensordot(self.att_v, y, dims=([0], [2]))
        elif self.activation == 'tanh':
            weights = torch.tensordot(self.att_v, torch.tanh(y), dims=([0], [2]))

        weights = F.softmax(weights, dim=0)
        out = x2 * weights.repeat(1, x2.shape[2]).reshape(weights.shape[0], weights.shape[1], x2.shape[2])
        
        batch_size, hidden_dim, input_size = x.size(0), x.size(2), x2.size(1)

        self.score = torch.bmm(x, out.transpose(1, 2))
        self.attn = F.softmax(self.score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(self.attn, out)
        
        return context

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention_tmp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
