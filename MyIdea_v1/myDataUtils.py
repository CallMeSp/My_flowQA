import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import argparse
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
QuACdata_path = '../torch_v/QuAC_data/train.json'


def readTrainjson(path):
    contexts = []
    contexts_lens = []
    wholeQues = []
    Ques_lens = []
    wholeAns = []
    Ans_lens = []
    quesTAG = []
    with open(path, encoding='utf8') as fr:
        datas = json.load(fr)['data']
        for i in range(len(datas)):
            context = datas[i]['paragraphs'][0]['context']
            ques = [qa['question'] for qa in datas[i]['paragraphs'][0]['qas']]
            ans = [
                qa['answers'][0]['text']
                for qa in datas[i]['paragraphs'][0]['qas']
            ]
            # 在traindata中，ans和origin ans 相同。
            ori_ans = [
                qa['orig_answer']['text']
                for qa in datas[i]['paragraphs'][0]['qas']
            ]

            # paragraph = []
            # paragraph.append(context)
            # paragraph.append(ques)
            # paragraph.append(ans)

            contexts.append(context)
            contexts_lens.append(len(context))
            wholeQues.extend(ques)
            Ques_lens.extend([len(que) for que in ques])
            wholeAns.extend(ans)
            Ans_lens.extend([len(an) for an in ans])
            quesTAG.extend([i] * len(ques))
    return contexts, wholeQues, wholeAns, quesTAG, contexts_lens, Ques_lens, Ans_lens


def paddingText(rawlist, padding='<PADDING>'):
    maxlen = 0
    for i in range(len(rawlist)):
        if len(rawlist[i]) > maxlen:
            maxlen = len(rawlist[i])
    curlist = [row + [padding] * (maxlen - len(row)) for row in rawlist]
    return curlist


tokenizer = TransfoXLTokenizer.from_pretrained('./pretrained_model')


def tokens2ids(text):
    # context
    tokenized_text = [tokenizer.tokenize(sent) for sent in text]
    tokenized_text = paddingText(tokenized_text)
    # Convert token to vocabulary indices
    indexed_tokens = [
        tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text
    ]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens)
    return tokens_tensor


model = TransfoXLModel.from_pretrained('./pretrained_model')
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


def ids2embeddings(ids):
    if torch.cuda.is_available():
        ids = ids.cuda()
    with torch.no_grad():
        hidden_state, mems = model(ids)
    return hidden_state


def genBatch(mode='train', bsz=2, ismasked=True):
    if mode == 'train':
        contexts, ques, ans, tags, contextsLength, quesLength, ansLength = readTrainjson(
            QuACdata_path)
        nbatches = len(contexts) // bsz + 1
        print('total nbatches:', nbatches)
        for i in range(nbatches):
            # print('batch :', i)
            contexts_batch = contexts[i * bsz:(i + 1) * bsz]
            ques_batch = [
                ques[j] for j in range(len(ques))
                if bsz * i <= tags[j] <= (i + 1) * bsz
            ]
            ans_batch = [
                ans[j] for j in range(len(ans))
                if bsz * i <= tags[j] <= (i + 1) * bsz
            ]
            tags_batch = [
                tags[j] - (i * bsz) for j in range(len(tags))
                if bsz * i <= tags[j] <= (i + 1) * bsz
            ]
            contextsLength_batch = contextsLength[i * bsz:(i + 1) * bsz]
            quesLength_batch = [
                quesLength[j] for j in range(len(quesLength))
                if bsz * i <= tags[j] <= (i + 1) * bsz
            ]
            ansLength_batch = [
                ansLength[j] for j in range(len(ansLength))
                if bsz * i <= tags[j] <= (i + 1) * bsz
            ]
            # src = Variable(data, requires_grad=False).to(device)
            # tgt = Variable(data, requires_grad=False).to(device)
            # print('contexts:', len(contexts_batch))
            # print('question:', len(ques_batch))
            yield (contexts_batch, ques_batch, ans_batch, tags_batch,
                   contextsLength_batch, quesLength_batch, ansLength_batch)


if __name__ == '__main__':
    contexts, ques, ans, tags, contextsLength, quesLength, ansLength = readTrainjson(QuACdata_path)

    contexts = genBatch()
    for i in range(10):
        contextbatch = next(contexts)[0]
        contextbatch = tokens2ids(contextbatch)
        print(contextbatch)
    exit(0)
    contexts = tokens2ids(contexts)
    ques = tokens2ids(ques).size()
    ans = tokens2ids(ans).size()
    logger.info(len(tags))
    model = TransfoXLModel.from_pretrained('pretrained_model')
    model.eval()
    # contexts = contexts.to('cuda')
    # model.to('cuda')
    print('contexts.size:', contexts.size())
    with torch.no_grad():
        logger.info('start model')
        last_state, mems = model(contexts[:4])
        logger.info('hidden_state1:', last_state.size())
        logger.info('        mems1:', len(mems), mems[0].size())
