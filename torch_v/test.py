import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, BertTokenizer

tokenizer = TransfoXLTokenizer.from_pretrained('../MyIdea_v1/pretrained_model')
tokenizer2 = BertTokenizer.from_pretrained('../MyIdea_v1/pretrained_model')
# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
tokenized_text2 = tokenizer2.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
indexed_tokens2 = tokenizer2.convert_tokens_to_ids(tokenized_text2)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print(indexed_tokens)
print(indexed_tokens2)
