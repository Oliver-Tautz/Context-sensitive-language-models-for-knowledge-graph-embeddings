import argparse
import textwrap

from pykeen.pipeline import pipeline


def main(args):
    modelname = args.modelname  # 'transe'
    epochs = args.epochs
    # dataset = 'relevant_dbpedia_2016-04.tsv'
    # dataset_test = 'relevant_dbpedia_2016-04.tsv'

    dataset = args.dataset
    result = pipeline(dataset=dataset, model=modelname, epochs=epochs, model_kwargs={
        'embedding_dim': 200})

    print(f'Results {modelname} on {dataset}:')
    print('MRR: ', result.get_metric('mrr'))
    print('MR: ', result.get_metric('mr'))
    print('hits1: ', result.get_metric('hits@1'))
    print('hits10: ', result.get_metric('hits@10'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--modelname", default='transE', type=str,
                        )
    parser.add_argument("--dataset", default='fb15k237', type=str,
                        )
    parser.add_argument("--epochs", default=50, type=int,
                        )

    args = parser.parse_args()

    print("passed args: " + str(args))
    main(args)
