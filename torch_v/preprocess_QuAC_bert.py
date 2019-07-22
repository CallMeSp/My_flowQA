import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random
import torch
from pytorch_pretrained_bert import BertTokenizer
from allennlp.modules.elmo import batch_to_ids
from general_utils import flatten_json, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, \
    find_answer_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 20 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--vocab_dir', default='../MyIdea_v1/pretrained_model',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')

args = parser.parse_args()
trn_file = 'QuAC_data/train.json'
# trn_file = 'QuAC_data/1ofTraindata.json'
dev_file = 'QuAC_data/dev.json'
# glove
wv_file = args.wv_file
wv_dim = args.wv_dim

nlp = spacy.load('en', disable=['parser'])

random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))


# ===============================================================
# =================== Work on training data =====================
# ===============================================================

def proc_train(ith, article):
    rows = []

    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answers = qa['orig_answer']

            answer = answers['text']
            # 这里坐标的单位是character
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])
            answer_choice = 0 if answer == 'CANNOTANSWER' else \
                1 if qa['yesno'] == 'y' else \
                    2 if qa['yesno'] == 'n' else \
                        3  # Not a yes/no question
            if answer_choice != 0:  # dialog Act
                """
                0: Do not ask a follow up question!
                1: Definitely ask a follow up question!
                2: Not too important, but you can ask a follow up.
                """
                answer_choice += 10 * (0 if qa['followup'] == "n" else \
                                           1 if qa['followup'] == "y" else \
                                               2)  # =='m
            else:
                answer_start, answer_end = -1, -1
            rows.append((ith, question, answer, answer_start, answer_end, answer_choice))
    return rows, context


def paddingText(rawlist, padding='<PADDING>'):
    maxlen = 0
    for i in range(len(rawlist)):
        if len(rawlist[i]) > maxlen:
            maxlen = len(rawlist[i])
    curlist = [row + [padding] * (maxlen - len(row)) for row in rawlist]
    return curlist


tokenizer = BertTokenizer.from_pretrained('../MyIdea_v1/pretrained_model')


def tokens2ids(text):
    # context
    tokenized_text = [tokenizer.tokenize(sent) for sent in text]
    tokenized_text = [["<CLS>"] + doc + ["<SEP>"] for doc in tokenized_text]
    # tokenized_text = paddingText(tokenized_text)
    # Convert token to vocabulary indices
    tokens_tensor = [
        tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text
    ]

    # Convert inputs to PyTorch tensors
    # tokens_tensor = torch.tensor(tokens_tensor)
    return tokens_tensor


# train 为所有的qas,len=83568(因为是extend连接的)，train_context是对应的context,len=11567
train, train_context = flatten_json(trn_file, proc_train)

train = pd.DataFrame(train, columns=['context_idx', 'question', 'answer',
                                     'answer_start', 'answer_end', 'answer_choice'])

log.info('train json data flattened.')

trC_iter = (pre_proc(c) for c in train_context)
trQ_iter = (pre_proc(q) for q in train.question)
trC_docs = [doc for doc in nlp.pipe(trC_iter, batch_size=64, n_threads=args.threads)]
trQ_docs = [doc for doc in nlp.pipe(trQ_iter, batch_size=64, n_threads=args.threads)]
trC_docs_forbert = [pre_proc(c) for c in train_context]
trQ_docs_forbert = [pre_proc(q) for q in train.question]
# tokens
trC_tokens = [[normalize_text(w.text) for w in doc] for doc in trC_docs]
trQ_tokens = [[normalize_text(w.text) for w in doc] for doc in trQ_docs]
trC_unnorm_tokens = [[w.text for w in doc] for doc in trC_docs]
log.info('All tokens for training are obtained.')
# 得到context中除了空白符的其它所有word和标点符号的以character为单位的span
train_context_span = [get_context_span(a, b) for a, b in zip(train_context, trC_unnorm_tokens)]
ans_st_token_ls, ans_end_token_ls = [], []
# 得到以word为单位的answer span
for ans_st, ans_end, idx in zip(train.answer_start, train.answer_end, train.context_idx):
    ans_st_token, ans_end_token = find_answer_span(train_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)
# 添加两列
train['answer_start_token'], train['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(train)
# defalt:axis=0,Drop rows which contain missing values;how='any',Determine if row or column is removed from DataFrame,
# when we have at least one NA or all NA
# 把所有含None的行删掉
train.dropna(inplace=True)  # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(train), initial_len))
log.info('answer span for training is generated.')
# features
print('args.nomatch:', args.no_match)

# trc_tags:list of 每个word的tag
# trc_ents:list of 每个word的实体识别，如果不是实体就=''
# trc_features:list of 每个word为一个元组(match_origin, match_lower, match_lemma, term_freq)
trC_tags, trC_ents, trC_features = feature_gen(trC_docs, train.context_idx, trQ_docs, args.no_match)
log.info('features for training is generated: {}, {}, {}'.format(len(trC_tags), len(trC_ents), len(trC_features)))

# tr_vocab = build_train_vocab(trQ_tokens, trC_tokens)
# trC_ids = token2id(trC_tokens, tr_vocab, unk_id=1)
# trQ_ids = token2id(trQ_tokens, tr_vocab, unk_id=1)
trC_ids = tokens2ids(trC_docs_forbert)
trQ_ids = tokens2ids(trQ_docs_forbert)
log.info('build index has been built by bert tokenization')

# tags
vocab_tag = [''] + list(nlp.tagger.labels)
trC_tag_ids = token2id(trC_tags, vocab_tag)
# entities
vocab_ent = list(set([ent for sent in trC_ents for ent in sent]))
trC_ent_ids = token2id(trC_ents, vocab_ent, unk_id=0)

log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

# don't store row name in csv
# train.to_csv('QuAC_data/train.csv', index=False, encoding='utf8')


prev_CID, first_question = -1, []
for i, CID in enumerate(train.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

result = {
    'question_ids': trQ_ids,
    'context_ids': trC_ids,
    'context_features': trC_features,  # exact match, tf
    'context_tags': trC_tag_ids,  # POS tagging
    'context_ents': trC_ent_ids,  # Entity recognition
    'context': train_context,
    'context_span': train_context_span,
    '1st_question': first_question,
    'question_CID': train.context_idx.tolist(),
    'question': train.question.tolist(),
    'answer': train.answer.tolist(),
    'answer_start': train.answer_start_token.tolist(),
    'answer_end': train.answer_end_token.tolist(),
    'answer_choice': train.answer_choice.tolist(),
    'context_tokenized': trC_tokens,
    'question_tokenized': trQ_tokens
}
with open('QuAC_data/train_data_bybert.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved training to disk.')


# ==========================================================
# =================== Work on dev data =====================
# ==========================================================

def proc_dev(ith, article):
    rows = []

    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answers = qa['orig_answer']

            answer = answers['text']
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])
            answer_choice = 0 if answer == 'CANNOTANSWER' else \
                1 if qa['yesno'] == 'y' else \
                    2 if qa['yesno'] == 'n' else \
                        3  # Not a yes/no question
            if answer_choice != 0:
                """
                0: Do not ask a follow up question!
                1: Definitely ask a follow up question!
                2: Not too important, but you can ask a follow up.
                """
                answer_choice += 10 * (0 if qa['followup'] == "n" else \
                                           1 if qa['followup'] == "y" else \
                                               2)
            else:
                answer_start, answer_end = -1, -1

            ans_ls = []
            for ans in qa['answers']:
                ans_ls.append(ans['text'])

            rows.append((ith, question, answer, answer_start, answer_end, answer_choice, ans_ls))
    return rows, context


dev, dev_context = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer',
                                 'answer_start', 'answer_end', 'answer_choice', 'all_answer'])
log.info('dev json data flattened.')

devC_iter = (pre_proc(c) for c in dev_context)
devQ_iter = (pre_proc(q) for q in dev.question)
devC_docs = [doc for doc in nlp.pipe(
    devC_iter, batch_size=64, n_threads=args.threads)]
devQ_docs = [doc for doc in nlp.pipe(
    devQ_iter, batch_size=64, n_threads=args.threads)]
devC_docs_forbert = [pre_proc(c) for c in dev_context]
devQ_docs_forbert = [pre_proc(q) for q in dev.question]
# tokens
devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
log.info('All tokens for dev are obtained.')

dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
log.info('context span for dev is generated.')

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
    ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(dev)
dev.dropna(inplace=True)  # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
log.info('answer span for dev is generated.')

# features
devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, args.no_match)
log.info('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))

devC_ids = tokens2ids(devC_docs_forbert)
devQ_ids = tokens2ids(devQ_docs_forbert)
# tags
devC_tag_ids = token2id(devC_tags, vocab_tag)  # vocab_tag same as training
# entities
devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0)  # vocab_ent same as training
log.info('vocabulary for dev is built.')

prev_CID, first_question = -1, []
for i, CID in enumerate(dev.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

result = {
    'question_ids': devQ_ids,
    'context_ids': devC_ids,
    'context_features': devC_features,  # exact match, tf
    'context_tags': devC_tag_ids,  # POS tagging
    'context_ents': devC_ent_ids,  # Entity recognition
    'context': dev_context,
    'context_span': dev_context_span,
    '1st_question': first_question,
    'question_CID': dev.context_idx.tolist(),
    'question': dev.question.tolist(),
    'answer': dev.answer.tolist(),
    'answer_start': dev.answer_start_token.tolist(),
    'answer_end': dev.answer_end_token.tolist(),
    'answer_choice': dev.answer_choice.tolist(),
    'all_answer': dev.all_answer.tolist(),
    'context_tokenized': devC_tokens,
    'question_tokenized': devQ_tokens
}
with open('QuAC_data/dev_data_bybert.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved dev to disk.')
