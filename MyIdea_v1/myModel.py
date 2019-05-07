import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_pretrained_bert import TransfoXLModel
from torch.autograd import Variable


# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """Set docstring here.

        Parameters
        ----------
        layer: 实际的encoder层
        N: 层数

        Returns
        -------
        经过N层layer得到的vector:[bsz,seq_len,d_model]
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # pass the input (and mask) through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # std为标准差
        std = x.std(-1, keepdim=True)
        # 0均值标准化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply residual connection to any sublayer with the same size.
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """
    生成n个相同的层
    :param module:层
    :param N:需要生成的个数
    :return:nn.moduleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        """Set docstring here.

        Parameters
        ----------
        size: 一般为d_model
        self_attn: multihead-selfAttention层
        feed_forward: 前馈网络层
        dropout: default=0.1 

        Returns
        -------
        输出应该size维持不变
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # follow Figure 1(left) for connections
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    # 屏蔽掉当前位置后面的序列
    attn_shape = (1, size, size)
    # triu 返回上三角矩阵，k=1为主对角线
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# Attention(Q,K,V)=softmax(QK^T/sqrt(d_k)) * V
def attention(query, key, value, mask=None, dropout=None):
    # Compute 'Scaled Dot Product Attention'
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 把mask==0的索引位置替换为value
        scores = scores.masked_fill(mask == 0, value=-1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        # print('in attention before dropout',p_attn)
        dropout = nn.Dropout(dropout)
        p_attn = dropout(p_attn)
        # print('in attention after dropout', p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h:并行的head数量
        :param d_model:原始word的embeddingsize或者一个encoder的输出维度
        :param dropout:dropout比例
        """
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = dropout
        assert d_model % h == 0
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，
        # 而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，
        # 把tensor变成在内存中连续分布的形式。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.h * self.d_k)
        return self.linears[-1](x)


# FFN(x)=max(0,xW1+b1)W2+b2
class PositionwiseFeedForward(nn.Module):
    # Implements FFN equation.
    def __init__(self, d_model, d_ff, dropout=0.1):
        """Set docstring here.

        Parameters
        ----------
        d_model: 原始word的embeddingsize或者一个encoder的输出维度或者multi-head-SelfAtt的输出维度
        d_ff: hidden层的维度
        dropout=0.1: 

        Returns
        -------
        返回还是d_model维度的
        [bsz,seq_len,d_model]
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 如果要使用预训练的则
        # nn.Embedding(vocab, d_model).weight.data.copy_(torch.from_numpy(W))
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        newx = x.long()
        embeddingMat = self.lut(newx) * math.sqrt(self.d_model)
        # print('x_embedding:', x.shape, embeddingMat.shape)
        return embeddingMat


class PositionalEncoding(nn.Module):
    # Implement the PE function.

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Set docstring here.

        Parameters
        ----------
        d_model: 原始word的embeddingsize
        dropout: default=0.1
        max_len: 序列的最大长度default=5000

        Returns
        -------
        和输入的x size相同
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            -(torch.arange(0., d_model, 2) / d_model) * math.log(10000.0))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(-2)], requires_grad=False)
        return self.dropout(x)


class SelfAttLayer(nn.Module):
    def __init__(self, data, N=6, head=8, ff_size=2048):
        """Set docstring here.

        Parameters
        ----------
        data: [bsz,seq_len,d_model]
        N: 叠加的层数default=6 
        head: multihead的头数
        ff_size:  FFN的隐藏层单元数

        Returns
        -------
        shape和输入相同
        """
        super(SelfAttLayer, self).__init__()
        self.N = N
        self.head = head
        self.d_model = data.size(-1)
        self.ff_size = ff_size
        self.d_k = self.d_model // self.head
        self.c = copy.deepcopy
        self.selfattn = MultiHeadedAttention(self.head, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.ff_size)
        self.position = PositionalEncoding(self.d_model)
        self.encoder = Encoder(
            EncoderLayer(self.d_model, self.selfattn, self.ff), self.N)

    def forward(self, data):
        return self.encoder(self.position(data), mask=None)


class QuestionFlowLayer(nn.Module):
    def __init__(self, questions, mode='onlyQ'):
        """
        :param questions: [bsz?,max_len,d_model]
        :param mode:选择进行only question flow 还是answer-aware question flow
        :return:[bsz?,max_len,2*d_model]
        """
        super(QuestionFlowLayer, self).__init__()
        self.questions = questions
        self.mode = mode
        self.d_model = questions.size(-1)
        self.result = torch.zeros(questions.size())
        self.question_num = questions.size(-3)

        self.result[0] = questions[0]
        for i in range(1, self.question_num):
            self.result[i] = questions[:i].mean(0)

    def forward(self):
        return torch.cat((self.questions, self.result), dim=-1)


class AttentionScore(nn.Module):
    """
    sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, attention_hidden_size, similarity_score=False):
        super(AttentionScore, self).__init__()
        self.linear = nn.Linear(input_size, attention_hidden_size, bias=False)

        if similarity_score:
            self.linear_final = Parameter(torch.ones(1, 1, 1) / (attention_hidden_size ** 0.5), requires_grad=False)
        else:
            self.linear_final = Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        scores: batch * len1 * len2 <the scores are not masked>
        """
        # x1 = dropout(x1, p=my_dropout_p, training=self.training)
        # x2 = dropout(x2, p=my_dropout_p, training=self.training)

        x1_rep = self.linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
        x2_rep = self.linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)

        x1_rep = F.relu(x1_rep)
        x2_rep = F.relu(x2_rep)
        final_v = self.linear_final.expand_as(x2_rep)

        x2_rep_v = final_v * x2_rep
        # bmm:batch matrix multiply
        scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
        return scores


class AwareIntegration(nn.Module):
    def __init__(self, context, questions):
        """
        实现对单一的context和它对应的questions进行integration
        :param context:[c_max_len,c_emb]
        :param questions:[q_num,q_max_len,q_emb]
        """
        super(AwareIntegration, self).__init__()
        self.context = context
        self.questions = questions

    # 空函数待补充
    def forward(self):
        return self.context


class QuestionAwareContextLayer(nn.Module):
    def __int__(self, contexts, questions, tags, mode='onlyQ'):
        """
        :param contexts:[bsz,c_max_len,c_emb]
        :param questions:[q_num,q_max_len,q_emb]
        :param tags:[q_num] an indicator that show which context that current question belong to
        :param mode:'onlyQ' or 'answer-aware'
        :return:[bsz,max_q_num,c_max_len,emb?]
        """
        super(QuestionAwareContextLayer, self).__init__()
        self.contexts = contexts
        self.questions = questions
        self.tags = tags
        max_q_num = 0
        for i in range(len(contexts)):
            cur_q_len = len(questions[tags == i])
            if cur_q_len > max_q_num:
                max_q_num = cur_q_len
        self.aware_contexts = torch.Tensor(contexts.size(0), max_q_num, contexts.size(-2), contexts.size(-1))
        for i in range(len(contexts)):
            cur_context_que = questions[tags == i]
            before_aware_questions = QuestionFlowLayer(cur_context_que)
            cur_aware_context = AwareIntegration(contexts[i], before_aware_questions)
            self.aware_contexts[i][:len(cur_context_que)] = cur_aware_context

    def forward(self):
        return self.aware_contexts


if __name__ == '__main__':
    print(__name__)
    # selfAttnLayer test
    # testdata = torch.Tensor(2, 3000, 2048).cuda()
    # ff = PositionwiseFeedForward(2048, 2048).cuda()
    # SALayer = SelfAttLayer(ff(testdata), 1).cuda()
    # print(SALayer.forward(testdata).size())

    # QuestionFlow Attn
    # x = torch.from_numpy(np.random.randint(10, size=[64, 2000, 2000])).float()
    # qf = QuestionFlowLayer(x)
    # print(qf.forward().size())

    # QuestionAwareContextLayer test
    input_size = 6
    attention_hidden_size = 10
    ats = AttentionScore(input_size, attention_hidden_size)
    linear = nn.Linear(input_size, attention_hidden_size, bias=False)
    x1 = torch.from_numpy(np.random.rand(4, 5, 6)).float()
    x2 = torch.from_numpy(np.random.rand(4, 7, 6)).float()
    linear_final = Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad=True)
    print('linear weight:',linear.weight.size())
    x1_rep = linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
    x2_rep = linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)
    x1_rep = F.relu(x1_rep)
    x2_rep = F.relu(x2_rep)
    print('x1:', x1_rep.size())
    final_v = linear_final.expand_as(x2_rep)
    print('final_v:', final_v.size())
    print('x2_rep:', x2_rep.size())
    x2_rep_v = final_v * x2_rep
    scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
    print('x2_rev_v:', x2_rep_v.size())
    print(F.softmax(scores,dim=-1))
    # print(x2)
    # print(ats(x1, x2))
