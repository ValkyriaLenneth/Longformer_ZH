"""
python: 3.8
transformers: 3.0.2
JioNlp
"""
import logging
import os
import copy
import math
import torch
import argparse

from dataclasses import  dataclass, field
from transformers import BertForMaskedLM, BertTokenizerFast, BertConfig
from transformers import Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

from LongDataset import TextDataset
from data_Collator import DataCollatorForLanguageModeling
""" 
RobertaLong
    RobertaLongForMaskedLM: the long version of the ROBERTa model.
    Replaces BertSelfAttention with RoberaLongSelfAttention
"""
class BertLongSelfAttention(LongformerSelfAttention):
    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False
                ):
        return super().forward(hidden_states,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions)

# change it into bert-version
class BertLongForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Replace self-attention with long-attention
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)

""" 
From Roberta check point, convert the model into a Long-Version.
    1. Extend the position embedding from 512 to max_pos=4096
    2. Initialize additional positional embedding by copying first 512 position embedding, which is crucial for performance.
    3. Replace modeling_bert.BertSelfAttention with modeling_longformer.LongformerSelfAttention with attention window size attention_window
"""
def create_long_model(save_model_to, attention_window, max_pos, pretrained_config, pretrained_checkpoint, pretrained_tokenizer):
    """
    Convert RoBERTa into Long-Version
    :param save_model_to: the model save path
    :param attention_window: the long-attention defined above
    :param max_pos: extend the position embedding to max_pos=4096
    :return: modified model and tokenizer
    """
    config = BertConfig.from_pretrained(pretrained_config)
    model = BertForMaskedLM.from_pretrained(pretrained_checkpoint, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer, model_max_length=max_pos)

    # extend position embedding
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    # RoBERTa has position 0,1 reserved, embedding size = max_pos + 2
    #max_pos += 2 # ??? is this fit for BERT-based RoBerta_zh?
    """ 
    RoBERTa reserved position 0 1,
    However, Bert-based RoBERTa_zh did not.
    """
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos

    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # init by duplication
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k: (k+step)] = model.bert.embeddings.position_embeddings.weight[0:]
        k += step
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed

    # The next problem is that: BERT_Based RoBERTa has not attribute [position_ids] for [bert.embeddings]
    # model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the modeling_bert.BertSelfAttention obj with LongformerSelfAttention
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i , layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def copy_proj_layers(model):
    """
    copy Q,K,V to the global counterpart projection matrices.
    :param model: the model
    :return: copied model
    """
    for i, layer in enumerate(model.bert.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

# pretrain and evaluate model on MLM
def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    """
    pretrain the model on Mask Language Model
    :param args: the arguments
    :param model: the model you want to pretrain
    :param tokenizer: the tokenizer
    :param eval_onlu: switcher for only evaluation
    :param model_path: the save path
    :return:
    """
    # A TextDataset simply splits the text into consecutive "blocks" of certain (token) length
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.max_len,
                              masked_lm_prob=0.1,
                              dupe_factor=2,
                              max_prediction_per_sentence=400,
                              )

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.max_len,
                                    masked_lm_prob=0.1,
                                    dupe_factor=2,
                                    max_prediction_per_sentence=400,
                                    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model,
                      args=args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      prediction_loss_only=True)
    eval_loss = trainer.evaluate()
    print(eval_loss)
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')

"""
Hyperparameters:
    tokens per batch: 2-18
    65K steps
    lr-scheduler: polynomial_decay with power 3 over 65K || constant lr-scheduler for 3K
    times: 2days on 1*32GB GPU
    CUDA_VISIBLE_DEVICES: GPUs
"""
@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Max position"})

def prepariation(pretrained_tokenizer, pretrained_config, pretrained_checkpoint, training_args, model_args, model_path):
    # 1. Evaluation roberta-base on MLM to establish the baseline.
    # RoBERTa_zh is based on BERT not RoBERTa, it is necessary to load the pretrained model with BERT.from_pretrained
    #
    roberta_base_config = BertConfig.from_pretrained(pretrained_config)
    roberta_base = BertForMaskedLM.from_pretrained(pretrained_checkpoint, config=roberta_base_config)
    roberta_base_tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer, model_max_length=512)

    logger.info('Evaluating roberta-base, seq_len: 512 for reference')
    pretrain_and_evaluate(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)

    # 2. Convert roberta-base into roberta-base-4096, the long-version instance of RobertaLong and save it.

    logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(save_model_to=model_path,
                                         attention_window=model_args.attention_window,
                                         max_pos=model_args.max_pos,
                                         pretrained_config=pretrained_config,
                                         pretrained_checkpoint=pretrained_checkpoint,
                                         pretrained_tokenizer=pretrained_tokenizer)


def main(dataset):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # the defination for pretrained-checkpoint , data_path
    logger.info("Setting the dataset")
    validation_data_path = r'./dataset/preprocessed/' + dataset + '_val'
    training_data_path = r'./dataset/preprocessed/' + dataset  + '_train'

    logger.info("setting the arguments")
    parser = HfArgumentParser((TrainingArguments, ModelArgs))
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'save_model',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_gpu_eval_batch_size', '4',
        '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
        '--gradient_accumulation_steps', '8',
        '--fp16',  # Add fp16 training
        '--evaluate_during_training',
        '--do_train',
        '--do_eval',
    ])
    training_args.val_datapath = validation_data_path
    training_args.train_datapath = training_data_path

    logger.info("loading the model")
    model_path = f'{training_args.output_dir}/longformer_mix/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    logger.info("loading the checkpoints")
    pretrained_tokenizer = r'save_model/roberta_512/' 
    pretrained_config = r'save_model/roberta_512/config.json'
    pretrained_checkpoint = r'save_model/roberta_512/'

    logger.info("Evaluate baseline and create the model")
    # 1&2。 Evaluate baseline and create the model for only one time
    if not os.path.exists(model_path + '/pytorch_model.bin'):
        prepariation(pretrained_tokenizer, pretrained_config, pretrained_checkpoint, training_args, model_args,
                     model_path)

    logger.info("Load the model and pertrain with preprocessed dataset")

    # 3. Load roberta-base-4096 from disk.
    logger.info(f'Loading the model from {model_path}')
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertLongForMaskedLM.from_pretrained(model_path)


    # 4. Pretrain roberta-base-4096 for 3K steps, each step has 2-18 tokens
    """
        1. training_args.max_step = 3 only for testing
        2. 3K steps take 2days on 32G gpu
        3. tokenizing take 5-10 mins
    """
    logger.info(f'Pretraining roberta-base-{model_args.max_pos}')
    # training_args.max_steps = 3 # for test
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    # print(tokenizer.max_len)
    # 5.Copy global projection layers.
    logger.info(f'Copying local projection layers into global projection layers')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

    # # 6.Congratulation
    logger.info(f"Loading the model from {model_path}")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertLongForMaskedLM.from_pretrained(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='select the dataset')
    # parser.add_argument('--is_valid', type=bool, required=True, help='is Val_set or not')
    # parser.add_argument('--order', type=int, required=True, help='the order of sub-dataset, ignored for valid set')
    args = parser.parse_args()


    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, filename='pretrain.log')

    main(args.dataset)