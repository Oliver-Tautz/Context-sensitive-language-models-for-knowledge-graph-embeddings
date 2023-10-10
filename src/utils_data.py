import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import BertTokenizer

from utils import run_str
from utils_graph import get_random_corrupted_triple


def random_mask(token, mask_token, vocab_size, mask_chance=0.33,
                mask_token_chance=0.9):
    mask_roll = torch.rand(())
    if mask_roll < mask_chance:
        mask_token_roll = torch.rand(())
        if mask_token_roll < mask_token_chance:
            return mask_token, 1
        else:
            return torch.randint(high=vocab_size, size=()), 2

    else:
        return token, 0


def mask_list_of_lists(l, mask_token, vocab_size, special_token_ids,
                       mask_chance=0.33, mask_token_chance=0.9):
    # get random mask for each token, but not for special tokens
    # this is not used ... probably too slow?!
    return torch.tensor(
        [[random_mask(y, mask_token, vocab_size, mask_chance=mask_chance,
                      mask_token_chance=mask_token_chance) if y not in special_token_ids else y
          for y in x] for x in l])


def lm_mask(ids, mask_token, special_token_ids):
    # TODO!
    return None


def mask_list(l, mask_token, vocab_size, special_token_ids, type="MLM",
              typeargs=None, mask_chance=0.33,
              mask_token_chance=0.9, triples=False):
    # get random mask for each token, but not for special tokens
    # if triples: only mask subject or object ... not relation ot both

    if not triples:
        # Random mask all non special tokens.
        if type == "MLM":
            return torch.tensor(
                [
                    random_mask(y, mask_token, vocab_size, mask_chance=mask_chance,
                                mask_token_chance=mask_chance) if y not in special_token_ids else (
                        y, 0) for y in l])
        elif type == "MASS":
            return torch.tensor(
                [
                    random_mask(y, mask_token, vocab_size, mask_chance=mask_chance,
                                mask_token_chance=1) if y not in special_token_ids else (
                        y, 0) for y in l])

    else:
        # only random mask subject or object but not both and never relation.
        decide = np.random.rand()

        if decide > 0.5:
            # mask head
            return torch.tensor(
                [(l[0], 0)] + [random_mask(l[
                                               1], mask_token, vocab_size, mask_chance=1, mask_token_chance=1)] + [
                    (l, 0)
                    for l
                    in
                    l[2:]])
        else:
            return torch.tensor([(l, 0) for l in l[0:3]] + [
                random_mask(l[
                                3], mask_token, vocab_size, mask_chance=1, mask_token_chance=1)] + [
                                    (l[4], 0)])


class DatasetBertTraining(torch.utils.data.Dataset):
    def __init__(self, triples, special_tokens_map, tokenizer=None,
                 max_length=128, dataset_type="MLM",
                 mask_chance=0.33, mask_token_chance=0.9):
        """


        dataset_type: one of MLM,MASS,LM,LP

        MLM: Masked language modeling with mask tokens and random replacements
        MASS: MLM with only mask tokens
        """
        self.mask_chance = mask_chance
        self.mask_token_chance = mask_token_chance

        self.special_tokens_map = special_tokens_map
        self.dataset_type = dataset_type
        if not tokenizer:
            word_level_tokenizer = Tokenizer(WordLevel(unk_token=
                                                       special_tokens_map[
                                                           'unk_token']))
            word_level_trainer = WordLevelTrainer(special_tokens=list(special_tokens_map.values()))
            # Pretokenizer. This is important and could lead to better/worse results!
            word_level_tokenizer.pre_tokenizer = WhitespaceSplit()

            word_level_tokenizer.train_from_iterator(triples, word_level_trainer)

            word_level_tokenizer.post_processor = BertProcessing(
                ("[SEP]", word_level_tokenizer.token_to_id("[SEP]")),
                ('[CLS]', word_level_tokenizer.token_to_id('[CLS]')),
            )
        else:
            word_level_tokenizer = tokenizer

        self.mask_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['mask_token'])
        pad_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['pad_token'])

        word_level_tokenizer.enable_truncation(max_length=max_length)
        word_level_tokenizer.enable_padding(pad_token=special_tokens_map[
            'pad_token'], pad_id=pad_token_id)

        tokenization = word_level_tokenizer.encode_batch(triples)

        self.labels = [x.ids for x in tokenization]
        label_lens = [len(x) for x in self.labels]
        if max(label_lens) == 5:
            self.triples = True
        else:
            self.triples = False

        self.attention_masks = [x.attention_mask for x in tokenization]

        # print('here!!!!!!', len(self.labels[0]))

        self.labels = torch.tensor(self.labels)
        self.attention_masks = torch.tensor(self.attention_masks)
        # print(self.attention_masks.shape)
        # self.attention_masks = torch.stack([torch.ones(len(x)) for x in self.labels])
        self.special_token_ids = [word_level_tokenizer.token_to_id(x) for x in
                                  special_tokens_map.values()]

        self.word_level_tokenizer = word_level_tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.dataset_type == "MLM":

            return mask_list(self.labels[i], self.mask_token_id,
                             self.word_level_tokenizer.get_vocab_size(), self.special_token_ids, triples=self.triples), \
                self.attention_masks[
                    i], \
                self.labels[i]
        elif self.dataset_type == "MASS":
            return mask_list(self.labels[i], self.mask_token_id,
                             self.word_level_tokenizer.get_vocab_size(), self.special_token_ids, type='MASS',
                             triples=self.triples), \
                self.attention_masks[i], \
                self.labels[i]


        else:
            print(f"dataset_type {self.dataset_type} not implemented! Abort :(")
            exit(-1)

    def get_tokenizer(self):
        return self.word_level_tokenizer

    def get_special_tokens_map(self):
        return self.special_tokens_map


class DatasetBertTraining_LM(torch.utils.data.Dataset):
    def __init__(self, input, labels, special_tokens_map, tokenizer=None,
                 max_length=128):
        """

        return X = sentence pairs y = (no_samples)
        """

        self.labels = labels
        self.special_tokens_map = special_tokens_map

        #print(special_tokens_map)
        if not tokenizer:
            word_level_tokenizer = Tokenizer(WordLevel(unk_token=
                                                       special_tokens_map[
                                                           'unk_token']))

            word_level_trainer = WordLevelTrainer(special_tokens=list(special_tokens_map.values()))
            # Pretokenizer. This is important and could lead to better/worse results!
            word_level_tokenizer.pre_tokenizer = WhitespaceSplit()

            word_level_tokenizer.train_from_iterator(input, word_level_trainer)

            word_level_tokenizer.post_processor = BertProcessing(
                ("[SEP]", word_level_tokenizer.token_to_id("[SEP]")),
                ('[CLS]', word_level_tokenizer.token_to_id('[CLS]')),
            )
        else:
            word_level_tokenizer = tokenizer

        self.mask_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['mask_token'])
        pad_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['pad_token'])

        word_level_tokenizer.enable_truncation(max_length=max_length)
        word_level_tokenizer.enable_padding(pad_token=special_tokens_map[
            'pad_token'], pad_id=pad_token_id)

        tokenization = word_level_tokenizer.encode_batch(input)

        self.ids = torch.stack([torch.tensor(e.ids) for e in tokenization])
        self.type_ids = torch.stack([torch.tensor(e.type_ids) for e in
                                     tokenization])
        self.attention_masks = torch.stack([torch.tensor(e.attention_mask) for
                                            e in tokenization])

        # self.attention_masks = torch.stack([torch.ones(len(x)) for x in self.labels])
        self.special_token_ids = [word_level_tokenizer.token_to_id(x) for x in
                                  special_tokens_map.values()]

        self.word_level_tokenizer = word_level_tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            self.ids[i], self.attention_masks[i], self.type_ids[i],
            self.labels[i])

    def get_tokenizer(self):
        return self.word_level_tokenizer

    def get_special_tokens_map(self):
        return self.special_tokens_map


class DatasetBertTraining_LP(torch.utils.data.Dataset):
    def __init__(self, triples, special_tokens_map, tokenizer=None,
                 max_length=128, one_to_n_n=50):
        """


        dataset_type: one of MLM,MASS,LM,LP

        MLM: Masked language modeling with mask tokens and random replacements
        MASS: MLM with only mask tokens
        TODO LM: Language modeling: predict end/beginning of sentence
        TODO LP: Link prediction
        """

        self.one_to_n_n = one_to_n_n
        self.special_tokens_map = special_tokens_map

        if not tokenizer:
            word_level_tokenizer = Tokenizer(WordLevel(unk_token=
                                                       special_tokens_map[
                                                           'unk_token']))
            word_level_trainer = WordLevelTrainer(special_tokens=list(special_tokens_map.values()))
            # Pretokenizer. This is important and could lead to better/worse results!
            word_level_tokenizer.pre_tokenizer = WhitespaceSplit()

            word_level_tokenizer.train_from_iterator(triples, word_level_trainer)

            word_level_tokenizer.post_processor = BertProcessing(
                ("[SEP]", word_level_tokenizer.token_to_id("[SEP]")),
                ('[CLS]', word_level_tokenizer.token_to_id('[CLS]')),
            )
        else:
            word_level_tokenizer = tokenizer

        self.mask_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['mask_token'])
        pad_token_id = word_level_tokenizer.token_to_id(
            special_tokens_map['pad_token'])

        word_level_tokenizer.enable_truncation(max_length=max_length)
        word_level_tokenizer.enable_padding(pad_token=special_tokens_map[
            'pad_token'], pad_id=pad_token_id)

        tokenization = word_level_tokenizer.encode_batch(triples)
        # print(tokenization)
        self.true_triples = [x.ids for x in tokenization]

        self.attention_masks = [x.attention_mask for x in tokenization]

        self.true_triples = torch.tensor(self.true_triples)
        # print(self.true_triples)
        # get all entity ids.
        self.entities = self.true_triples[:, 1:4].flatten().unique()

        self.attention_masks = torch.tensor(self.attention_masks)

        # self.attention_masks = torch.stack([torch.ones(len(x)) for x in self.labels])
        self.special_token_ids = [word_level_tokenizer.token_to_id(x) for x in
                                  special_tokens_map.values()]

        # print(self.entities)

        self.word_level_tokenizer = word_level_tokenizer

    def __len__(self):
        return len(self.true_triples)

    def __getitem__(self, i):
        current_triples = self.true_triples[i]
        if len(current_triples.shape) < 2:
            current_triples = current_triples.unsqueeze(0)
        triples_shape = current_triples.shape

        return current_triples, self.__get_false_triples(current_triples)

    def __get_false_triples(self, true_triples):
        if len(true_triples.shape) == 1:
            true_triples = [true_triples]
        fake_triples = []
        for i in range(self.one_to_n_n):
            fake_triples.extend([get_random_corrupted_triple(x, self.entities)
                                 for x in true_triples])
        return torch.stack(fake_triples)

    def get_tokenizer(self):
        return self.word_level_tokenizer

    def get_special_tokens_map(self):
        return self.special_tokens_map


def get_bert_simple_dataset(graph):
    # Just 's p o' as sentence.
    dataset_most_simple = [' '.join(x) for x in graph]

    # get special BERT tokens
    tz = BertTokenizer.from_pretrained("bert-base-cased")

    special_tokens_map = tz.special_tokens_map_extended

    del (tz)

    dataset = DatasetBertTraining(dataset_most_simple, special_tokens_map)
    tz = dataset.get_tokenizer()

    return dataset, tz


def generate_walks(tmpdir, dataset_file, outname, nproc, depth,
                   no_walks_per_entity,
                   generation_mode="RANDOM_WALKS_DUPLICATE_FREE", light=None):
    # if file does not exist or is empty ...
    print(
        f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -light {light} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
    if not Path(f'{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt').is_file() or os.stat(
            f'{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt') == 0:
        print('')
        if not Path('./jrdf2vec-1.3-SNAPSHOT.jar').is_file():
            run_str("wget -q -nc https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar")
        print(tmpdir)

        if light:
            print(
                f"java -Xmx100g -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -light {light} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
            run_str(
                f"java -Xmx100g -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -light {light} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
            run_str(
                f"java -Xmx100g -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -mergeWalks -o {tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt")
        else:

            print(
                f"java -Xmx100g -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
            run_str(
                f"java -Xmx100g -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
            run_str(
                f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -mergeWalks -o {tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt")

        # remove unused tmpfiles
        for file in os.listdir(tmpdir):
            if Path(file).suffix == '.gz':
                os.remove(f'{tmpdir}/{file}')

    return f"{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt"
