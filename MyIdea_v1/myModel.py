import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_pretrained_bert import TransfoXLModel
from torch.autograd import Variable
import logging
from MyIdea_v1 import myDataUtils
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
class Encoder(nn.Module):
    def __init__(self, layer, N, opt):
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
        self.usegpu = opt['cuda']
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size, opt)

    def forward(self, x, mask):
        # pass the input (and mask) through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Encoder_v2(nn.Module):
    def __init__(self, layer, N, opt):
        """Set docstring here.

        Parameters
        ----------
        layer: 实际的encoder层
        N: 层数

        Returns
        -------
        经过N层layer得到的vector:[bsz,seq_len,d_model]
        """
        super(Encoder_v2, self).__init__()
        self.usegpu = opt['cuda']
        self.layers = clones(layer, N)
        self.norm = LayerNorm(opt['hidden_size'] * 2, opt)

    def forward(self, x, mask):
        # pass the input (and mask) through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """

    def __init__(self, features, opt, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.usegpu = opt['cuda']
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        if self.usegpu:
            self.a_2 = nn.Parameter(torch.ones(features).cuda())
            self.b_2 = nn.Parameter(torch.zeros(features).cuda())
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

    def __init__(self, size, dropout, opt):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, opt)
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
    def __init__(self, size, self_attn, feed_forward, opt, dropout=0.1):
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
        self.usegpu = opt['cuda']
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, opt), 2)
        if self.usegpu:
            self.sublayer = self.sublayer.cuda()
        self.size = size

    def forward(self, x, mask):
        # follow Figure 1(left) for connections
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class EncoderLayer_v2(nn.Module):
    def __init__(self, size, self_attn, feed_forward, opt, dropout=0.1):
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
        super(EncoderLayer_v2, self).__init__()
        self.usegpu = opt['cuda']
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, opt), 2)
        if self.usegpu:
            self.sublayer = self.sublayer.cuda()
        self.size = size

    def forward(self, x, mask):
        # follow Figure 1(left) for connections
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.feed_forward(x)


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
        edge = d_model // 2 if d_model % 2 == 0 else d_model // 2
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-(torch.arange(0., d_model, 2) / d_model) * math.log(10000.0))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, 0:edge]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('before:', x.size())
        x = x + Variable(self.pe[:, :x.size(-2)], requires_grad=False)
        # print('after:', x.size())
        return self.dropout(x)


class SelfAttLayer(nn.Module):
    def __init__(self, dmodel, opt, N=2, head=4, ff_size=2048):
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
        self.usegpu = opt['cuda']
        self.N = N
        self.head = head
        self.d_model = dmodel
        self.ff_size = ff_size
        self.d_k = self.d_model // self.head
        self.c = copy.deepcopy
        self.selfattn = MultiHeadedAttention(self.head, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.ff_size)
        self.position = PositionalEncoding(self.d_model)
        if self.usegpu:
            self.selfattn = self.selfattn.cuda()
            self.ff = self.ff.cuda()
            self.position = self.position.cuda()
        self.encoder = Encoder(
            EncoderLayer(self.d_model, self.selfattn, self.ff, opt), self.N, opt)

    def forward(self, data):
        return self.encoder(self.position(data), mask=None)


# FFN(x)=max(0,xW1+b1)W2+b2
class PositionwiseFeedForward_v2(nn.Module):
    # Implements FFN equation.
    def __init__(self, d_model, d_ff, d_output, dropout=0.1):
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
        super(PositionwiseFeedForward_v2, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_output)
        self.w_3 = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_3(x)


class SelfAttLayerFF(nn.Module):
    def __init__(self, dmodel, opt, N=6, head=2, ff_size=200):
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
        super(SelfAttLayerFF, self).__init__()
        self.usegpu = opt['cuda']
        self.N = N
        self.head = head
        self.d_model = dmodel
        self.ff_size = ff_size
        self.d_k = self.d_model // self.head
        self.c = copy.deepcopy
        self.d_output = opt['hidden_size'] * 2
        self.selfattn = MultiHeadedAttention(self.head, self.d_model)
        self.ff = PositionwiseFeedForward_v2(self.d_model, self.ff_size, self.d_output)
        self.position = PositionalEncoding(self.d_model)
        if self.usegpu:
            self.selfattn = self.selfattn.cuda()
            self.ff = self.ff.cuda()
            self.position = self.position.cuda()
        self.encoder = Encoder_v2(
            EncoderLayer_v2(self.d_model, self.selfattn, self.ff, opt), self.N, opt)

    def forward(self, data):
        return self.encoder(self.position(data), mask=None)


class QuestionFlowLayer(nn.Module):
    def __init__(self, opt, mode='onlyQ'):
        """
        :param questions: [bsz?,max_len,d_model]
        :param mode:选择进行only question flow 还是answer-aware question flow
        :return:[bsz?,max_len,2*d_model]
        """
        super(QuestionFlowLayer, self).__init__()
        self.usegpu = opt['cuda']
        # self.questions = questions
        self.mode = mode
        # self.d_model = questions.size(-1)
        # self.result = torch.zeros(questions.size())
        # if self.usegpu:
        #     # self.questions = self.questions.cuda()
        #     self.result = self.result.cuda()
        # self.question_num = questions.size(-3)
        #
        # self.result[0] = questions[0]
        # for i in range(1, self.question_num):
        #     self.result[i] = questions[:i].mean(0)

    def forward(self, questions):
        avgques = torch.zeros(questions.size())
        if self.usegpu:
            avgques = avgques.cuda()
        for i in range(1, questions.size(0)):
            avgques[i] = questions[:i].mean(0)
        return torch.cat((questions, avgques), dim=-1)


class AttentionScore(nn.Module):
    """
    sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, x1_d, x2_d, attention_hidden_size, opt, similarity_score=False):
        super(AttentionScore, self).__init__()
        self.usegpu = opt['cuda']
        self.linear1 = nn.Linear(x1_d, attention_hidden_size, bias=False)
        self.linear2 = nn.Linear(x2_d, attention_hidden_size, bias=False)
        if similarity_score:
            self.linear_final = Parameter(torch.ones(1, 1, 1) / (attention_hidden_size ** 0.5), requires_grad=False)
        else:
            self.linear_final = Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad=True)

        if self.usegpu:
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            # self.linear_final = self.linear_final.cuda()

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        scores: batch * len1 * len2 <the scores are not masked>
        """
        # x1 = dropout(x1, p=my_dropout_p, training=self.training)
        # x2 = dropout(x2, p=my_dropout_p, training=self.training)
        x1_rep = self.linear1(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
        x2_rep = self.linear2(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)

        x1_rep = F.relu(x1_rep)
        x2_rep = F.relu(x2_rep)
        final_v = self.linear_final.expand_as(x2_rep)

        x2_rep_v = final_v * x2_rep
        # bmm:batch matrix multiply
        scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
        return scores


class AwareIntegration(nn.Module):
    def __init__(self, AttentionScoreLayer, opt):
        """
        实现对单一的context和它对应的questions进行integration
        :param context:[c_max_len,c_emb]
        :param questions:[q_num,q_max_len,q_emb]
        :return:[q_num,c_max_lem.q_emb]
        """
        super(AwareIntegration, self).__init__()
        self.usegpu = opt['cuda']
        # reshape to [1,c_max_len,c_emb]
        # attention_hidden_size = 2 * context.size(-1)
        # self.scoreLayer = AttentionScore(self.context, self.questions, attention_hidden_size)
        self.scoreLayer = AttentionScoreLayer

    def forward(self, context, questions):
        questionsAwareContexts = torch.ones(len(questions), context.size(0), questions.size(-1))
        context = context.unsqueeze(0)
        for i in range(len(questions)):
            question = questions[i].unsqueeze(0)
            score = self.scoreLayer(context, question)
            questionAwareContext = F.softmax(score, dim=-1).bmm(question).squeeze(0)
            questionsAwareContexts[i] = questionAwareContext
        return questionsAwareContexts


class QuestionAwareContextLayer(nn.Module):
    def __init__(self, AwareIntegrationLayer, opt, mode='onlyQ'):
        """
        :param contexts:[bsz,c_max_len,c_emb]
        :param questions:[q_num,q_max_len,q_emb] 原始的，没有经过before-aware的
        :param tags:[q_num] an indicator that show which context that current question belong to
        :param mode:'onlyQ' or 'answer-aware'
        :return:[q_num,c_max_len,2 * q_emb]
        """
        super(QuestionAwareContextLayer, self).__init__()
        self.opt = opt
        self.usegpu = opt['cuda']
        self.QFlowlayer = QuestionFlowLayer(opt)
        self.AwareIntegrationLayer = AwareIntegrationLayer
        if self.usegpu:
            self.QFlowlayer = self.QFlowlayer.cuda()

    def forward(self, contexts, questions, tags):
        start = 0
        if self.usegpu:
            aware_contexts = torch.Tensor(questions.size(0), contexts.size(-2), questions.size(-1) * 2).cuda()
        else:
            aware_contexts = torch.Tensor(questions.size(0), contexts.size(-2), questions.size(-1) * 2)

        for i in range(contexts.size(0)):
            cur_context_que = questions[tags == i]
            if self.usegpu:
                cur_aware_context = self.AwareIntegrationLayer(contexts[i],
                                                               self.QFlowlayer(cur_context_que)).cuda()
                aware_contexts[start:start + len(cur_context_que)] = cur_aware_context
                start += len(cur_context_que)
            else:
                cur_aware_context = self.AwareIntegrationLayer(contexts[i],
                                                               self.QFlowlayer(cur_context_que))
                aware_contexts[start:start + len(cur_context_que)] = cur_aware_context
                start += len(cur_context_que)

        return aware_contexts


class Verifier(nn.Module):
    def __init__(self, answerr_sent, ques, answers, mode='2'):
        pass

    def forward(self, *input):
        pass


def test():
    contexts = myDataUtils.genBatch()
    start = time.time()
    for i in range(5784):
        (contexts_batch, ques_batch, ans_batch, tags_batch,
         contextsLength_batch, quesLength_batch,
         ansLength_batch) = next(contexts)

        context_batch_id = myDataUtils.tokens2ids(contexts_batch)
        ques_batch_id = myDataUtils.tokens2ids(ques_batch)

        context_batch_emb = myDataUtils.ids2embeddings(context_batch_id)
        ques_batch_emb = myDataUtils.ids2embeddings(ques_batch_id)
        tags_batch = torch.IntTensor(tags_batch)

        print('context_batch_emb:', context_batch_emb.size())
        print('ques_batch_emb', ques_batch_emb.size())

        QAC = QuestionAwareContextLayer(
            contexts=context_batch_emb,
            questions=ques_batch_emb,
            tags=tags_batch)

        print(QAC.forward().size())
        exit(0)
        # print(context_batch_emb.size())
        print(i)
    end = time.time()
    print((start - end) / 60)
    exit(0)


if __name__ == '__main__':
    print(__name__)
    # # selfAttnLayer test
    # opt = {}
    # opt['cuda'] = True
    # opt['hidden_size'] = 125
    # testdata = torch.Tensor(20, 550, 2252).cuda()
    # SALayer = SelfAttLayerFF(2252, opt)
    # parameters = [p for p in SALayer.parameters() if p.requires_grad]
    # x = sum([p.nelement() for p in parameters])
    # print(SALayer.forward(testdata).size(), x)

    # # QuestionAwareContextLayer test
    # c = torch.from_numpy(np.random.rand(3, 5, 6)).float().cuda()
    # q = torch.from_numpy(np.random.rand(7, 7, 9)).float().cuda()
    # qtags = torch.IntTensor([0, 0, 1, 1, 1, 2, 2])
    # QAC = QuestionAwareContextLayer(contexts=c, questions=q, tags=qtags, opt=opt)
    # QAC = QAC().cuda()
    # print(QAC.is_cuda)
    # Wqac = nn.Linear(18, 10)
    # if opt['cuda']:
    #     Wqac = Wqac.cuda()
    # print(Wqac(QAC))
    # # QuestionFlow Attn Test
    # x = torch.from_numpy(np.random.randint(10, size=[64, 2000, 2000])).float()
    # qf = QuestionFlowLayer(x)
    # print(qf.forward().size())

    # # QuestionAwareContextLayer test
    # c = torch.from_numpy(np.random.rand(3, 5, 6)).float()
    # q = torch.from_numpy(np.random.rand(7, 7, 9)).float()
    # qtags = torch.IntTensor([0, 0, 1, 1, 1, 2, 2])
    # QAC = QuestionAwareContextLayer(contexts=c, questions=q, tags=qtags)
    # print(QAC.forward().size())
    dmodel = 975
    testdata = torch.Tensor(20, 550, dmodel)
    pe = PositionalEncoding(dmodel)
    print(pe(testdata))
