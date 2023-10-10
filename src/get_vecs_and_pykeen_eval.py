import argparse
import subprocess
import textwrap
from pathlib import Path


def main(args):
    modelpath = args.bert_path  #
    # '/home/olli/remote/techfak/compute/Better_Knowledge_Graph_Embeddings/codex_models/tiny_bert_from_scratch_simple_codex_ep1000_vec200_walks_d=3_w=100_g=RANDOM_WALKS_dataset=MLM_mask-chance=0.33_mask-token-chance=1.0'
    entity_filepath = args.entity_filepath  # '/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/codex_l_ents.txt'

    pykeen_train_triples = args.pykeen_train_triples  # "/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/train.txt"
    pykeen_test_triples = args.pykeen_test_triples  # "/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/test.txt"

    vector_filepath = f'{Path(modelpath).name}_' \
                      f'{Path(pykeen_train_triples).name}_vectors.txt'
    print(vector_filepath)
    if not Path(vector_filepath).is_file():
        if args.add_base_url != None:

            subprocess.run(
                ['python', 'generate_bert_vectors.py', '--bert-path', modelpath,
                 '--kg-path', pykeen_train_triples, '--out-path',
                 vector_filepath, '--add-base-url', args.add_base_url,
                 '--bert-walks', str(args.bert_walks), '--bert-mode-depth',
                 str(args.bert_mode_depth)])
        else:
            print(['python', 'generate_bert_vectors.py', '--bert-path', modelpath,
                 '--kg-path', pykeen_train_triples, '--out-path',
                 vector_filepath, '--add-base-url', 'False',
                 '--bert-walks', str(args.bert_walks), '--bert-mode-depth',
                 str(args.bert_mode_depth)])
            subprocess.run(
                ['python', 'generate_bert_vectors.py', '--bert-path', modelpath,
                 '--kg-path', pykeen_train_triples, '--out-path',
                 vector_filepath, '--bert-walks', str(args.bert_walks),
                 '--bert-mode-depth',
                 str(args.bert_mode_depth)])

    subprocess.run(
        ['python', 'evaluate_embeddings_in_pykeen.py', '--train-triples',
         pykeen_train_triples, '--test-triples',
         pykeen_test_triples,
         '--vector-file', vector_filepath, '--model-name-pykeen', 'ERMLP',
         '--epochs', '25'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--bert-path", help="Filepath of the config file. See "
                                    "example config.", type=str,
                        )

    parser.add_argument("--entity_filepath", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--pykeen-train-triples", help="Filepath of the "
                                                       "config file. See example config.", type=str,
                        )
    parser.add_argument("--pykeen-test-triples", help="Filepath of the "
                                                      "config file. See example config.", type=str,
                        )
    parser.add_argument("--add-base-url", help="Filepath of the config file. See example config.", type=str,
                        default=None
                        )

    parser.add_argument('--bert-walks', type=str, default=False)
    parser.add_argument('--bert-best-eval', action='store_true',
                        help='use checkpoint with best eval loss using early stopping.')
    parser.add_argument('--bert-mode-depth', type=int, default=3,
                        help="sdepth of walks for embedding.")

    args = parser.parse_args()

    main(args)
