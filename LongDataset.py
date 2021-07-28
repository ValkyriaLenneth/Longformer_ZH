# 改写 Dataset 和 DataLoader， fn_collator 来实现中文 Whole Word Mask

import torch
import os
import transformers
import jieba
import random
import re
import pickle
import time
import logging
import random
import collections
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)

from filelock import FileLock

from torch.utils.data.dataset import Dataset
# 首先改写 TextDataset, 完成对原始数据的处理， 最后返回处理后的数据和 mask 矩阵
""" 
输入是每一行的文本，
输出是 max_seq_len 长度的实例，而且经过 vocab2id, 并且经过 mask， 返回 [original_tokens, masked_tokens, labels]
"""
class TextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 file_path,
                 block_size,
                 masked_lm_prob,
                 dupe_factor,
                 max_prediction_per_sentence,
                 overwrite_cache=False):
        assert os.path.isfile(file_path)
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename, ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            # Load the existed cache_file
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.instances = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                # # Create new samples
                # self.instances = self.create_instance_from_document(tokenizer,file_path,block_size,masked_lm_prob,max_prediction_per_sentence)

                # In order to implement dynamic masking, the function would be used dupe_factor times
                self.instances = []
                print('creating instances')
                for i in trange(dupe_factor):
                    seed = int(random.random() * 1000)
                    instances = self.create_instance_from_document(tokenizer, file_path, block_size, masked_lm_prob, max_prediction_per_sentence, seed)
                    self.instances.extend(instances)

                random.shuffle(self.instances)
                start = time.time()
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.instances, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


    def create_instance_from_document(self, tokenizer,file_path,block_size,masked_lm_prob, max_prediction_per_sentence, seed):
        """
        文件格式为每行代表一个纯文本，如果小于 block_size 则自动拼接到下一行
        """

        instances = []
        print('get raw instance')
        raw_text_list_list = self.get_raw_instance(file_path, block_size, tokenizer)
        print('creating instances for each block')
        for j, raw_text_list in enumerate(tqdm(raw_text_list_list)):
            original_tokens = tokenizer.convert_tokens_to_ids(raw_text_list)
            raw_text_list = self.get_new_segment(raw_text_list) # 结合中文分词，在 whole mask 位置加上 ##
            # 设置 token， segment_ids
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in raw_text_list:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append('[SEP]')
            segment_ids.append(0)
            # 调用原有的方法创建instance
            vocab_words = list(tokenizer.get_vocab().keys())
            rng = random.Random(seed)
            # print('original text: ', tokens)
            (tokens, masked_lm_positions, masked_lm_labels) = self.create_masked_lm_predictions(tokens, masked_lm_prob,
                                                                                                max_prediction_per_sentence,
                                                                                                vocab_words, rng, True)
            """ 
            tokens: ['[CLS]', '斩', '[MASK]', '[MASK]', '者', '称', '号', '怎', '么', '得', '来', '[SEP]']
            masked_lm_position: [2, 3]
            masked_lm_labels: ['魔', '##仙']
            """
            # 接下来的问题是如何整合到 transformers 的本有接口
            # 每次处理一个 instance,
            # 原来 TextDataset 返回的是一个 torch.tensor
            # 那么我们现在就应该返回 一个 tensor 代表没有经过 mask 的原句子
            # 一个 masked 之后的sample
            # 一个 masked_pos => tensor 代表 mask——matrix
            # 最后用 getItem 时候返回一个 tuple，然后用 fn_collator 组装一下
            # 最后 data_collator 返回的是一个 dict{input_ids: batch * padded_seqs, labels: batch * labels}
            # 那么在这里，直接就组装好 input_id 和 labels 就行了
            # print('tokens: ', tokens)
            # print('masked_lm_position: ', masked_lm_positions)
            # print('masked_lm_labels: ', masked_lm_labels)

            masked_tokens = tokenizer.convert_tokens_to_ids(tokens)
            original_tokens = torch.tensor(original_tokens, dtype=torch.long)
            masked_tokens = torch.tensor(masked_tokens, dtype=torch.long)

            # generate mask_label[-100 id ... ]
            mask_label = torch.full(masked_tokens.shape, -100, dtype=torch.long)
            for pos in masked_lm_positions:
                mask_label[pos] = original_tokens[pos - 1]

            # print('maked_tokens id: ', masked_tokens)
            # print('mask_label: ', mask_label)
            # instances.append([original_tokens, masked_tokens, mask_label])
            cur_inst_len = len(instances)
            if cur_inst_len % 2000 == 0:
                logger.info("current instances: {x}".format(x=cur_inst_len))
            instances.append([masked_tokens, mask_label])

            """ 
            TextDataset: 
                每次返回一个 torch.LongTensor, 代表原始的 tokens_ids
            DataCollator:
                1. 把 batch_size 个 tokens_ids 进行 pad。 -> Tensor: B * Seq
                2. 把这个 batch 进行 mask， 最后返回 
                        inputs：Tensor: B * Seq， 代表 masked_tokens_ids
                        labels: Tensor: B * Seq, 所有被 mask 的元素为 原来的token_id， 其他为 -100
            """
            """ 
            1. 现在的 masked_tokens 可以直接作为 inputs
            2. 为了获得 labels：
                1. 首先在 TextDataset 里面， 对 original_tokens 和 masked_lm_labels 来生成一个 tensor: [-100 id ... ]， 记为 mask_label
                    注意 original_tokens 和 masked_tokens 的 shape 区别， 最后应该和 masked_tokens.shape 相同
                2. 在经过 DataCollator 的 pad 之后， 应该返回两个tensor：
                    1. inputs： 由 B * masked_tokens 直接获得
                    2. labels: 由 B * mask_label 获得
                    3. 检查 labels 里面的 PAD_TOK， 把他们换成 -100 
            """

        return instances


    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    def create_masked_lm_predictions(self, tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng, do_whole_word_mask):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (do_whole_word_mask and len(cand_indexes) >= 1 and
                    token.startswith("##")):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        rng.shuffle(cand_indexes)

        output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 else \
                        tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

                output_tokens[index] = masked_token

                masked_lms.append(self.MaskedLmInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def get_new_segment(self, segment):
        """
        输入一句话，返回一句经过处理的话。
        为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
        现在的 segment 的长度应该是 block_size， 而且是减掉了 special_token 后的
        直接用 roberta_zh 原版可以
        """
        seq_cws = jieba.lcut("".join(segment))
        seq_cws_dict = {x: 1 for x in seq_cws}
        new_segment = []
        i = 0
        while i < len(segment):
            if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 不是中文的，原文加进去。
                new_segment.append(segment[i])
                i += 1
                continue

            has_add = False
            for length in range(3, 0, -1):
                if i + length > len(segment):
                    continue
                if ''.join(segment[i:i + length]) in seq_cws_dict:
                    new_segment.append(segment[i])
                    for l in range(1, length):
                        new_segment.append('##' + segment[i + l])
                    i += length
                    has_add = True
                    break
            if not has_add:
                new_segment.append(segment[i])
                i += 1
        return new_segment

    def get_raw_instance(self, file_path, block_size, tokenizer):
        """
        把整个文件， 按照 block_size 切分成多个部分， 然后返回一个 list,
        没必要按照 Google 那样按照每一行， 直接把整段话切割
        :return raw_instances: list[list] 每个元素是一个 block_size 的 tokens.
        """
        raw_instances = []
        with open(file_path, encoding='utf-8', mode='r') as reader:
            text = reader.read()
            print('tokenizing')
            tokenized_text = tokenizer.tokenize(text)
            print('done')
            for i in trange(0, len(tokenized_text) - block_size + 1, block_size):
                instance = tokenized_text[i: i + block_size] # 一个 block_size 长度的 tokens
                raw_instances.append(instance)

        # instance_sizes = [len(ins) for ins in raw_instances]
        print(len(raw_instances))
        return raw_instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        # [original_tokens: tensor, masked_tokens: tensor, mask_label: tensor]
        # [masked_tokens: tensor, mask_label: tensor]
        return self.instances[i]


if __name__ == '__main__':
    data = r'./dataset/demo.txt'
    from transformers import BertTokenizerFast
    pretrained_tokenizer = r'roberta_checkpoint/'
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer, model_max_length=512)
    dataset = TextDataset(tokenizer, data, 7, 0.1, 5, 2)
    print(dataset.__len__())
