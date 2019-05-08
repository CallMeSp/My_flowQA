import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import logging


# For attending the span in document from the query
class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:
    o_i = x_i'Wy for x_i in X.
    """

    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        # x = dropout(x, p=my_dropout_p, training=self.training)
        # y = dropout(y, p=my_dropout_p, training=self.training)
        print('x:', x.size())
        print('y:', y.size())
        print('x_mask:', x_mask.size(), x_mask)
        Wy = self.linear(y) if self.linear is not None else y
        print('wy', Wy.size())
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))

        print('xWy', xWy.size())

        return xWy


class GetSpanStartEnd(nn.Module):
    # supports MLP attention and GRU for pointer network updating
    def __init__(self,
                 x_size,
                 h_size,
                 do_indep_attn=True,
                 attn_type="Bilinear",
                 do_ptr_update=True):
        super(GetSpanStartEnd, self).__init__()

        self.attn = BilinearSeqAttn(x_size, h_size)
        self.attn2 = BilinearSeqAttn(x_size, h_size) if do_indep_attn else None

        self.rnn = nn.GRUCell(x_size, h_size) if do_ptr_update else None

    def forward(self, x, h0, x_mask):
        """
        x = batch * len * x_size
        h0 = batch * h_size
        x_mask = batch * len
        """
        print('span,x:', x.size())
        st_scores = self.attn(x, h0, x_mask)
        # st_scores = batch * len

        if self.rnn is not None:
            ptr_net_in = torch.bmm(
                F.softmax(st_scores, dim=1).unsqueeze(1), x).squeeze(1)
            # ptr_net_in = dropout(
            #     ptr_net_in, p=my_dropout_p, training=self.training)
            # h0 = dropout(h0, p=my_dropout_p, training=self.training)
            h1 = self.rnn(ptr_net_in, h0)
            # h1 same size as h0
        else:
            h1 = h0

        end_scores = self.attn(x, h1, x_mask) if self.attn2 is None else \
            self.attn2(x, h1, x_mask)
        print('st,end', st_scores.size(), st_scores.argmax(),
              end_scores.size(), end_scores.argmax())
        exit(0)
        # end_scores = batch * len
        return st_scores, end_scores


if __name__ == '__main__':
    x = torch.from_numpy(np.random.rand(2, 3, 6)).float()
    y = torch.from_numpy(np.random.rand(2, 9)).float()
    x_mask = torch.IntTensor([[0, 0, 1], [0, 1, 0]])

    gs = GetSpanStartEnd(6, 9)

    print(gs(x, y, x_mask).size())
