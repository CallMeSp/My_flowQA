# @Time : 2019/5/6 23:30
# @Author : 史鹏
# @Site : 
# @File : train.py
# @Software: PyCharm
import time
import myDataUtils
import myModel


def test():
    contexts = myDataUtils.genBatch()
    start = time.time()
    for i in range(5784):
        (contexts_batch, ques_batch, ans_batch, tags_batch,
         contextsLength_batch, quesLength_batch, ansLength_batch) = next(contexts)
        # print(len(contexts_batch))
        context_batch_id = myDataUtils.tokens2ids(contexts_batch)
        # print(context_batch_id.size())
        context_batch_emb = myDataUtils.ids2embeddings(context_batch_id)
        # print(context_batch_emb.size())
        print(i)
    end = time.time()
    print((start - end) / 60)
    exit(0)


if __name__ == '__main__':
    print('program start..')
    test()
