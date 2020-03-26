import argparse
import logging
import random
import numpy as np
import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, MSELoss
from logger import Logger

from dataset import BatchType
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, BertPreTrainedModel, BertPreTrainingHeads, BertLMPredictionHead
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from transformers.modeling_electra import ElectraModel, ElectraConfig 
from transformers.modeling_electra import ElectraPreTrainedModel 




class ElectraPretrainingLoss(ElectraPreTrainedModel):
    def __init__(self, electra_encoder, config):
        super(ElectraPretrainingLoss, self).__init__(config)
        self.electra = electra_encoder
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        """
        Computing the loss for Electra pretraining 

        Inputs:
        @input_ids: tensor of shape (batch_size, max_seq_len), contains integers indicating the ids of input tokens, usually after masking (id is from [0, vac_size-1])
        @token_type_ids: tensor of shape (batch_size, max_seq_len), contains integers with either 0 or 1, indicating which sentence (first or the second or padding) each token belongs to (0 means the first sentence, 1 means the second sentence, and then 0 at the end means padding)
        @attention_mask: tensor of shape (batch_size, max_seq_len), contains integers with either 0 or 1, indicating padding tokens (1 means real sentence tokens and 0 means padding tokens)
        @masked_lm_labels: tensor of shape (batch_size, max_seq_len), contains integers, indicating the ids of masked tokens. (-1 means non-masked tokens, any value from [0, vocab_size-1] means the true ids of masked tokens) 
        @next_sentence_label: tensor of shape (batch_size, 1), contains either 0 or 1. 

        Returns:
        @total_loss: scalar tensor, represents summation of generator loss (i.e., mlm_loss) and discriminator loss (i.e., disc_loss). 
        """

        batch_size, sequence_length = input_ids.shape

        #masked_lm_positions = torch.zeros(batch_size).reshape(-1, 1) + torch.arange(sequence_length) 
        masked_lm_positions = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, sequence_length).to(masked_lm_labels.device)
        masked_lm_ids = masked_lm_labels

        # generator
        _, _, _, mlm_logits, _, _, mlm_loss = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_lm_positions=masked_lm_positions, masked_lm_ids=masked_lm_ids)

        # sample fake data from the output of the generator
        fake_data_ids, is_fake_tokens = self._get_fake_data(input_ids, masked_lm_labels, mlm_logits)

        # discriminator
        _, _, _, _, _, disc_loss = self.electra(input_ids=fake_data_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, fake_token_labels=is_fake_tokens)

        total_loss = mlm_loss + disc_loss

        #torch.cuda.synchronize()
        #print('Done')
        #import pdb;pdb.set_trace()

        return total_loss

    def _get_fake_data(self, input_ids, masked_lm_labels, mlm_logits):
        """
        Sample fake data 

        Inputs:
        @input_ids: tensor of shape (batch_size, max_seq_len), contains integers indicating the ids of input tokens, usually after masking (id is from [0, vocab_size-1])
        @masked_lm_labels: tensor of shape (batch_size, max_seq_len), contains integers, indicating the ids of masked tokens. (-1 means non-masked tokens, any value from [0, vocab_size-1] means the true ids of masked tokens)
        @mlm_logits: tensor of shape (batch_size, max_seq_len, vocab_size), contains unnormalized probality for each token from vocabulary (of size vocab_size) at each location (from 0 to max_seq_len-1) for each input sequence.

        Returns:
        @fake_data_ids: tensor of shape (batch_size, max_seq_len), contains integers indicating the ids of fake tokens (each id is from [0, vocab_size-1])
        @is_fake_tokens: tensor of shape (batch_size, max_seq_len), contains either 0 or 1 indicating whether each token is different from the original unmasked input. (1 means different, 0 means the same.)
        """
        cat_dist = torch.distributions.categorical.Categorical(logits=mlm_logits)
        sampled_ids = cat_dist.sample()
        masked_ids_selector = (masked_lm_labels != -1)
        original_input_ids = input_ids.clone()
        original_input_ids[masked_ids_selector] = masked_lm_labels[masked_ids_selector]

        fake_data_ids = original_input_ids.clone()
        fake_data_ids[masked_ids_selector] = sampled_ids[masked_ids_selector]

        is_fake_tokens = (fake_data_ids!=original_input_ids).long()
        return (fake_data_ids.detach(), is_fake_tokens.detach())

class BertPretrainingLoss(BertPreTrainedModel):
    def __init__(self, bert_encoder, config):
        super(BertPretrainingLoss, self).__init__(config)
        self.bert = bert_encoder
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.cls.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        torch.cuda.synchronize()
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class MTLRouting(nn.Module):
    """This setup is to add MultiTask Training support in BERT Training. 
    """
    def __init__(self, encoder: BertModel, write_log, summary_writer):
        super(MTLRouting, self).__init__()
        self.bert_encoder = encoder
        self._batch_loss_calculation = nn.ModuleDict()
        self._batch_counter = {}
        self._batch_module_name = {}
        self._batch_name = {}
        self.write_log = write_log
        self.logger = Logger(cuda=torch.cuda.is_available())
        self.summary_writer = summary_writer

    def register_batch(self, batch_type, module_name, loss_calculation: nn.Module):
        assert isinstance(loss_calculation, nn.Module)
        self._batch_loss_calculation[str(batch_type.value)] = loss_calculation
        self._batch_counter[batch_type] = 0
        self._batch_module_name[batch_type] = module_name

    def log_summary_writer(self, batch_type, logs: dict, base='Train'):
        if self.write_log:
            counter = self._batch_counter[batch_type]
            module_name = self._batch_module_name.get(
                batch_type, self._get_batch_type_error(batch_type))
            for key, log in logs.items():
                self.summary_writer.add_scalar(
                    f'{base}/{module_name}/{key}', log, counter)
            self._batch_counter[batch_type] = counter + 1

    def _get_batch_type_error(self, batch_type):
        def f(*args, **kwargs):
            message = f'Misunderstood batch type of {batch_type}'
            self.logger.error(message)
            raise ValueError(message)
        return f

    def forward(self, batch, log=True):
        batch_type = batch[0][0].item()

        # Pretrain Batch
        if batch_type == BatchType.PRETRAIN_BATCH:
            loss_function = self._batch_loss_calculation[str(batch_type)]

            loss = loss_function(input_ids=batch[1],
                                 token_type_ids=batch[3],
                                 attention_mask=batch[2],
                                 masked_lm_labels=batch[5],
                                 next_sentence_label=batch[4])
            if log:
                self.log_summary_writer(
                    batch_type, logs={'pretrain_loss': loss.item()})
            return loss

CONFIGS = {"bert": BertConfig,
           "electra": ElectraConfig,
           }

MODELS = {"bert": BertModel,
          "electra": ElectraModel,
          }

LOSS_FUNCS = {"bert": BertPretrainingLoss,
              "electra": ElectraPretrainingLoss,
              }

class BertMultiTask:
    def __init__(self, model_name, job_config, use_pretrain, tokenizer, cache_dir, device, write_log, summary_writer):
        self.job_config = job_config

        if not use_pretrain:
            model_config = self.job_config.get_model_config()
            #bert_config = BertConfig(**model_config)
            bert_config = CONFIGS[model_name](**model_config)
            bert_config.vocab_size = len(tokenizer.vocab)

            #self.bert_encoder = BertModel(bert_config)
            self.bert_encoder = MODELS[model_name](bert_config)
        # Use pretrained bert weights
        else:
            self.bert_encoder = MODELS[model_name].from_pretrained(self.job_config.get_model_file_type(), cache_dir=cache_dir)
            bert_config = self.bert_encoder.config

        self.network=MTLRouting(self.bert_encoder, write_log = write_log, summary_writer = summary_writer)

        #config_data=self.config['data']

        # Pretrain Dataset
        #self.network.register_batch(BatchType.PRETRAIN_BATCH, "pretrain_dataset", loss_calculation=BertPretrainingLoss(self.bert_encoder, bert_config))
        self.network.register_batch(BatchType.PRETRAIN_BATCH, "pretrain_dataset", loss_calculation=LOSS_FUNCS[model_name](self.bert_encoder, bert_config))

        self.device=device
        # self.network = self.network.float()
        # print(f"Bert ID: {id(self.bert_encoder)}  from GPU: {dist.get_rank()}")

    def save(self, filename: str):
        network=self.network.module
        return torch.save(network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))

    def move_batch(self, batch, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()
