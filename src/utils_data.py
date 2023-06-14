from settings import VECTOR_SIZE
from utils import choose
from utils_graph import get_random_corrupted_triple
import torch
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from transformers import BertTokenizer, EncoderDecoderModel, BertForTokenClassification
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import BertProcessing
from transformers import BertConfig, BertModel, AutoModel
from rdflib import Graph
import numpy as np


class Dataset11(torch.utils.data.Dataset):
    def __init__(self, graph, vec_mapping, entities, corrupt='random', vector_size=VECTOR_SIZE):
        """
        graph: graph to train on
        vec_mapping: function that returns vectos from URIs
        entities_or_relations: iterable of all entities_or_relations to build fake triples
        """

        self.entities = vec_mapping(np.array(entities))
        self.graph = get_vectors_fast(graph, vec_mapping)
        self.len = len(graph)
        self.vec_mapping = vec_mapping
        self.corrupt = corrupt
        self.vector_size = vector_size

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        return get_1_1_dataset_embedded(self.graph[ix], self.entities, self.corrupt, self.vector_size)


def random_mask(token, mask_token, vocab_size, mask_chance=0.33, mask_token_chance=0.9):
    mask_roll = torch.rand(())
    if mask_roll < mask_chance:
        mask_token_roll = torch.rand(())
        if mask_token_roll < mask_token_chance:
            return mask_token, 1
        else:
            return torch.randint(high=vocab_size, size=()), 2

    else:
        return token, 0


def mask_list_of_lists(l, mask_token, vocab_size, special_token_ids, mask_chance = 0.33, mask_token_chance = 0.9):
    # get random mask for each token, but not for special tokens
    # this is not used ... probably too slow?!
    return torch.tensor(
        [[random_mask(y, mask_token, vocab_size,mask_chance= mask_chance, mask_token_chance = mask_token_chance) if y not in special_token_ids else y for y in x] for x in l])

def lm_mask(ids,mask_token,special_token_ids):
    #TODO!
    return None

def mask_list(l, mask_token, vocab_size, special_token_ids, type="MLM",typeargs=None, mask_chance = 0.33, mask_token_chance = 0.9):
    # get random mask for each token, but not for special tokens
    if type =="MLM":
        return torch.tensor(
            [random_mask(y, mask_token, vocab_size,mask_chance = mask_chance, mask_token_chance= mask_chance) if y not in special_token_ids else (y, 0) for y in l])
    elif type == "MASS":
        return torch.tensor(
            [random_mask(y, mask_token, vocab_size,mask_chance = mask_chance,mask_token_chance=1) if y not in special_token_ids else (y, 0) for y in l])
    elif type == "LM":
        lm_mask(l,mask_token,special_token_ids)


class DatasetBertTraining(torch.utils.data.Dataset):
    def __init__(self, triples, special_tokens_map, tokenizer=None, max_length=128, dataset_type="MLM", mask_chance = 0.33, mask_token_chance = 0.9):
        """


        dataset_type: one of MLM,MASS,LM,LP

        MLM: Masked language modeling with mask tokens and random replacements
        MASS: MLM with only mask tokens
        TODO LM: Language modeling: predict end/beginning of sentence
        TODO LP: Link prediction
        """
        self.mask_chance = mask_chance
        self.mask_token_chance = mask_token_chance

        self.special_tokens_map = special_tokens_map
        self.dataset_type = dataset_type
        if not tokenizer:
            word_level_tokenizer = Tokenizer(WordLevel(unk_token=special_tokens_map['unk_token']))
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


        self.mask_token_id = word_level_tokenizer.token_to_id(special_tokens_map['mask_token'])
        pad_token_id = word_level_tokenizer.token_to_id(special_tokens_map['pad_token'])

        word_level_tokenizer.enable_truncation(max_length=max_length)
        word_level_tokenizer.enable_padding(pad_token=special_tokens_map['pad_token'], pad_id=pad_token_id)

        tokenization = word_level_tokenizer.encode_batch(triples)

        self.labels = [x.ids for x in tokenization]
        self.attention_masks = [x.attention_mask for x in tokenization]

        self.labels = torch.tensor(self.labels)
        self.attention_masks = torch.tensor(self.attention_masks)
        print(self.attention_masks.shape)
        # self.attention_masks = torch.stack([torch.ones(len(x)) for x in self.labels])
        self.special_token_ids = [word_level_tokenizer.token_to_id(x) for x in special_tokens_map.values()]

        self.word_level_tokenizer = word_level_tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.dataset_type == "MLM":

            return mask_list(self.labels[i], self.mask_token_id,
                             self.word_level_tokenizer.get_vocab_size(), self.special_token_ids), self.attention_masks[
                i], \
                self.labels[i]
        elif self.dataset_type == "MASS":
            return mask_list(self.labels[i], self.mask_token_id,
                                  self.word_level_tokenizer.get_vocab_size(), self.special_token_ids,type='MASS'), \
            self.attention_masks[i], \
                self.labels[i]


        else:
            print(f"dataset_type {self.dataset_type} not implemented! Abort :(")
            exit(-1)

    def get_tokenizer(self):
        return self.word_level_tokenizer
    def get_special_tokens_map(self):
        return self.special_tokens_map

class DatasetBertTraining_LP(torch.utils.data.Dataset):
    def __init__(self, triples, special_tokens_map, tokenizer=None, max_length=128,one_to_n_n=50):
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
            word_level_tokenizer = Tokenizer(WordLevel(unk_token=special_tokens_map['unk_token']))
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


        self.mask_token_id = word_level_tokenizer.token_to_id(special_tokens_map['mask_token'])
        pad_token_id = word_level_tokenizer.token_to_id(special_tokens_map['pad_token'])

        word_level_tokenizer.enable_truncation(max_length=max_length)
        word_level_tokenizer.enable_padding(pad_token=special_tokens_map['pad_token'], pad_id=pad_token_id)

        tokenization = word_level_tokenizer.encode_batch(triples)
        print(tokenization)
        self.true_triples = [x.ids for x in tokenization]

        self.attention_masks = [x.attention_mask for x in tokenization]

        self.true_triples = torch.tensor(self.true_triples)
        print(self.true_triples)
        # get all entity ids.
        self.entities = self.true_triples[:,1:4].flatten().unique()


        self.attention_masks = torch.tensor(self.attention_masks)

        # self.attention_masks = torch.stack([torch.ones(len(x)) for x in self.labels])
        self.special_token_ids = [word_level_tokenizer.token_to_id(x) for x in special_tokens_map.values()]

        print(self.entities)

        self.word_level_tokenizer = word_level_tokenizer

    def __len__(self):
        return len(self.true_triples)

    def __getitem__(self, i):
        current_triples = self.true_triples[i]
        if len(current_triples.shape)<2:
            current_triples = current_triples.unsqueeze(0)
        triples_shape = current_triples.shape

        return  current_triples, self.__get_false_triples(current_triples)

    def __get_false_triples(self,true_triples):
        if len(true_triples.shape) == 1:
            true_triples = [true_triples]
        fake_triples = []
        for i in range(self.one_to_n_n):
            fake_triples.extend([get_random_corrupted_triple(x,self.entities) for x in true_triples])
        return torch.stack(fake_triples)

    def get_tokenizer(self):
        return self.word_level_tokenizer
    def get_special_tokens_map(self):
        return self.special_tokens_map

def get_1_1_dataset(graph, entities, entity_vec_mapping, corrupt='random'):
    original_triple_len = len(graph)
    # get triples
    X = list(graph)
    no_t = len(X)

    corrupted_triples = [get_random_corrupted_triple(x, entities, corrupt=corrupt) for x in X]
    X = X + corrupted_triples

    # convert uris to strings

    X = get_vectors_fast(X, entity_vec_mapping)

    # stack them

    Y = np.concatenate((np.ones(no_t), np.zeros(no_t))).astype(np.uint8)

    return X, Y


def get_1_1_dataset_embedded(graph, entities, corrupt='random', vector_size=VECTOR_SIZE):
    """
    graph: numpy array of shape (samples,3*vecsize)
    """

    if len(graph.shape) == 1:
        graph = np.expand_dims(graph, 0)
    # print(graph.shape)
    no_t = len(graph)
    corrupted_triples = [get_random_corrupted_triple_embedded(x, entities, corrupt=corrupt, vector_size=vector_size) for
                         x in graph]
    X = np.concatenate((graph, corrupted_triples), axis=0)

    Y = np.concatenate((np.ones(no_t), np.zeros(no_t))).astype(np.uint8)

    return X, Y


def get_vectors_fast(triples, entity_vec_mapping, vector_size=VECTOR_SIZE):
    # ~20-30% faster
    X = np.array(triples)
    X = entity_vec_mapping(X.flatten()).reshape(len(triples), vector_size * 3)

    return X


def get_random_corrupted_triple_embedded(triple, entities, corrupt='object', vector_size=VECTOR_SIZE):
    """
    corrupt = one of 'subject', 'object', 'both'

    return corrupted triple with random path
    """

    s = triple[0:VECTOR_SIZE]
    p = triple[VECTOR_SIZE:VECTOR_SIZE * 2]
    o = triple[VECTOR_SIZE * 2:]

    # set up as the same
    s_corr = s[:]
    o_corr = o[:]

    if corrupt == 'subject':

        # corrupt only the subject
        while (s_corr == s).all():
            s_corr = choose(entities)
    elif corrupt == 'object':
        # corrupt only the object
        while (o_corr == o).all():
            o_corr = choose(entities)
    elif corrupt == 'random':
        # corrupt one or both randomly
        ch = np.random.randint(3)

        if ch == 0:
            while (s_corr == s).all():
                s_corr = choose(entities)
        if ch == 1:
            while (o_corr == o).all():
                o_corr = choose(entities)
        if ch == 2:

            while (s_corr == s).all() or (o_corr == o).all():
                s_corr = choose(entities)
                o_corr = choose(entities)
    else:

        while (s_corr == s).all() or (o_corr == o).all():
            s_corr = choose(entities)
            o_corr = choose(entities)

    return np.concatenate((s_corr, p, o_corr), axis=0)


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



