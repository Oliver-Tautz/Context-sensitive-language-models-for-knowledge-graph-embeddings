import argparse
import textwrap
from pathlib import Path

import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from tqdm import tqdm


def main(args):
    train_file = args.train_triples  # '/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/train.txt'
    test_file = args.test_triples  # '/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/test.txt'
    vector_file = args.vector_file  # f'/home/olli//gits/Better_Knowledge_Graph_Embeddings/test.txt'
    pykeen_model_name = args.model_name_pykeen
    epochs = args.epochs

    training = TriplesFactory.from_path(
        train_file,
        create_inverse_triples=True)

    testing = TriplesFactory.from_path(
        test_file,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
        create_inverse_triples=True,
    )

    uri_2_embedding = {}

    with open(vector_file) as embeddings_in:
        for line in embeddings_in:
            line = line.strip()
            uri_and_embedding = line.split(' ')
            uri_2_embedding[uri_and_embedding[0]] = [float(x) for x in
                                                     uri_and_embedding[1:]]
    embedding_dim = len(list(uri_2_embedding.values())[0])

    num_entities = len(training.entity_to_id.keys())
    pretrained_embedding_tensor = torch.rand(num_entities, embedding_dim)

    key_errors = 0
    for entity, index in tqdm(training.entity_to_id.items()):
        if entity in uri_2_embedding:
            pretrained_embedding_tensor[index] = torch.tensor(
                uri_2_embedding[entity])
        else:
            key_errors += 1
    print('Ratio of entities without feature vector:', key_errors / pretrained_embedding_tensor.size(0))

    def initialize_from_pretrained(x: torch.FloatTensor) -> torch.FloatTensor:
        return pretrained_embedding_tensor

    print('Start training.')
    result = pipeline(
        training=train_file,
        testing=test_file,
        model=pykeen_model_name,
        epochs=epochs,
        device='gpu',
        model_kwargs=dict(
            embedding_dim=pretrained_embedding_tensor.shape[-1],
            entity_initializer=initialize_from_pretrained,
        ),
    )

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--train-triples", help="Filepath of the config file. See example config.", type=str,
                        )

    parser.add_argument("--test-triples", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--vector-file", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--model-name-pykeen", help="Filepath of the config file. See example config.", type=str,
                        default='transE'
                        )
    parser.add_argument("--epochs", help="Filepath of the config file. See example config.", type=int, default=50
                        )

    args = parser.parse_args()
    result = main(args)

    print(f'Vector_file={Path(args.vector_file).name},dataset={Path(args.train_triples).name}')
    print('MRR', result.get_metric('mrr'))
    print('MR', result.get_metric('mr'))
    print('Hits@10', result.get_metric('hits@10'))
    print('Hits@3', result.get_metric('hits@3'))
    print('Hits@1', result.get_metric('hits@1'))
