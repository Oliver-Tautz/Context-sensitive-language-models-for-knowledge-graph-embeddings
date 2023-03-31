import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, AutoModel
from pathlib import Path
from utils import batch, merge_lists
from utils_graph import parse_kg
from random import randint
from tqdm import tqdm, trange
import json


def get_embeddings(entities, bert_model, tokenizer):
    entities = [tokenizer.encode(x) for x in np.array(entities)]
    embeddings = bert_model(torch.tensor(entities))
    embeddings = embeddings['last_hidden_state'][:, 1]
    return embeddings


class BertKGEmb():
    def __init__(self, path, datapath=None, depth=3):
        self.depth = depth
        path = Path(path)
        with open(path / 'special_tokens_map.json', 'r') as file:
            self.special_tokens_map = json.load(file)

        try:
            tz = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"),
                                         pad_token=self.special_tokens_map['pad_token'])
            model = AutoModel.from_pretrained(path / "model")
        except Exception as e:
            print("Cant load Bert model!")
            print('Exception was: ', e)
            exit(-1)

        self.model = model
        self.tz = tz

        self.pad_token = self.special_tokens_map['pad_token']
        self.pad_token_id = self.tz.convert_tokens_to_ids(self.special_tokens_map['pad_token'])

        if datapath:

            # get representation of graph
            entities, predicates, edges, edge_type = parse_kg(datapath)

            # preprocess datapath into ent_tokens, rel_tokens, edges

            # generate walks for all entities_or_relations
            walks_entities = []
            walks_rels = []

            for e in trange(len(entities)):
                es, rs = self.__gen_random_walk_entity(e, edges, edge_type, depth=self.depth)
                walks_entities.append(es)
                walks_rels.append(rs)

            walks_entities = [entities[x].tolist() for x in walks_entities]
            walks_rels = [predicates[x].tolist() for x in walks_rels]
            walks = self.__merge_walks(walks_entities, walks_rels)

            self.entity_walks = dict(zip(entities, walks))

            # generate walks for all relations.
            walks_entities = []
            walks_rels = []

            for p in trange(len(predicates)):
                es, rs = self.__gen_random_walk_relation(p, edges, edge_type, depth=self.depth)
                walks_entities.append(es)
                walks_rels.append(rs)

            walks_entities = [entities[x].tolist() for x in walks_entities]
            walks_rels = [predicates[x].tolist() for x in walks_rels]
            walks = self.__merge_walks(walks_entities, walks_rels)

            self.relation_walks = dict(zip(predicates, walks))

    def get_embeddings(self, entities_or_relations, mode='single', batchsize=100, relations=False):
        """
        modes:
        single :: use [CLS],ent,[SEP] as input for each entity/relation.

        """

        def bert_process(input, column=1):

            # transformers padding works now :)
            # pad manually on string level... this is pretty bad practice.

            # lens = torch.tensor([len(x.split()) for x in input])
            # maxlen = torch.max(lens)

            # for i,en in enumerate(input):
            #    if lens[i] < maxlen:
            #        input[i] = input[i]+' '.join([self.pad_token]*(maxlen-lens[i]).item())

            input = self.tz(list(input), padding='longest', return_tensors='pt')

            ids = input['input_ids']
            masks = input['attention_mask']

            ids = batch(ids, batchsize)
            masks = batch(masks, batchsize)

            batches = []
            for i, m in zip(ids, masks):
                batches.append(self.model(input_ids=i, attention_mask=m)['last_hidden_state'][:, column])
            embeddings = torch.cat(batches)

            return embeddings

        if mode == 'single':
            embeddings = bert_process(entities_or_relations)

            return embeddings

        if mode == 'walks':
            if not relations:
                entities_or_relations = [self.entity_walks[e] for e in entities_or_relations]
                # use column 1, because first entity after CLS is used
                embeddings = bert_process(entities_or_relations, 1)
                return embeddings

            else:
                entities_or_relations = [self.relation_walks[r] for r in entities_or_relations]
                # use columne 2, because first relation is used
                embeddings = bert_process(entities_or_relations, 2)
                return embeddings

    def __gen_random_walk_relation(self, relation_id, edges, edge_type, depth, max_retries=5, stepback=5):

        walk_entities = []
        walk_relations = [relation_id]
        retries = 0

        # setup first edge
        possible_edges = (edge_type == relation_id).nonzero()[0]
        chosen_edge = possible_edges[randint(0, len(possible_edges) - 1)]

        walk_entities.append(edges[chosen_edge][0])
        walk_entities.append(edges[chosen_edge][1])
        depth -= 1

        # continue as if using entities_or_relations.

        entity_id = edges[chosen_edge][1]

        while depth > 0:
            # consider ix of edges starting in entity
            print(entity_id)
            possible_edges = (edges[:, 0] == entity_id).nonzero()[0]

            # choose one randomly

            # if not empty
            if len(possible_edges) > 0:
                chosen_edge = possible_edges[randint(0, len(possible_edges) - 1)]
            else:
                # dead end ... so take a step back.
                if stepback > 0 and len(walk_entities) > 1:
                    # not implemented. Needed/sensible?
                    pass
                return walk_entities, walk_relations

            # if already in walk, retry.
            if edges[chosen_edge][1] in walk_entities:
                retries += 1
                if retries >= max_retries:
                    return walk_entities, walk_relations
                continue

            # save info
            walk_relations.append(edge_type[chosen_edge])

            entity_id = edges[chosen_edge][1]
            walk_entities.append(entity_id)
            depth -= 1

        return np.array(walk_entities), np.array(walk_relations)

    def __gen_random_walk_entity(self, entity_id, edges, edge_type, depth, max_retries=5, stepback=5):
        walk_entities = [entity_id]
        walk_relations = []
        retries = 0

        while depth > 0:
            # consider ix of edges starting in entity
            possible_edges = (edges[:, 0] == entity_id).nonzero()[0]

            # choose one randomly

            # if not empty
            if len(possible_edges) > 0:
                chosen_edge = possible_edges[randint(0, len(possible_edges) - 1)]
            else:
                # dead end ... so take a step back.
                if stepback > 0 and len(walk_entities) > 1:
                    # not implemented. Needed/sensible?
                    pass
                return walk_entities, walk_relations

            # if already in walk, retry.
            if edges[chosen_edge][1] in walk_entities:
                retries += 1
                if retries >= max_retries:
                    return walk_entities, walk_relations
                continue

            # save info
            walk_relations.append(edge_type[chosen_edge])

            entity_id = edges[chosen_edge][1]
            walk_entities.append(entity_id)
            depth -= 1

        return np.array(walk_entities), np.array(walk_relations)

    def __merge_walks(self, walks_entities, walks_rels):
        walks = []
        for ens, rels in zip(walks_entities, walks_rels):
            walk = ' '.join(merge_lists(ens, rels))
            walks.append(walk)
        return walks


if __name__ == "__main__":
    datapath = '/home/olli/gits/Better_Knowledge_Graph_Embeddings/src/FB15k-237/train.nt'
    datapath_test = '/home/olli/gits/Better_Knowledge_Graph_Embeddings/src/FB15k-237/test_1000.nt'

    # create model from checkpoint with integrated dataset
    bert_embeddings = BertKGEmb(
        path='tiny_bert_from_scratch_simple_ep100_vec128_walks_d=4_w=100_g=RANDOM_WALKS_DUPLICATE_FREE_dataset=MLM',
        datapath=datapath)
    # bert_embeddings = BertKGEmb(
    #    path='tiny_bert_from_scratch_simple_ep100_vec128_walks_d=4_w=100_g=RANDOM_WALKS_DUPLICATE_FREE_dataset=MLM')
    # get some entites and predicated
    entities, predicates, _, _ = parse_kg(datapath_test)

    # get some embeddings for entities and relations
    print(bert_embeddings.get_embeddings(entities, mode='walks').shape)
    print(bert_embeddings.get_embeddings(predicates, mode='walks', relations=True).shape)

    # get simple embeddings
    print(bert_embeddings.get_embeddings(entities, mode='single').shape)
    print(bert_embeddings.get_embeddings(predicates, mode='single', relations=True).shape)
