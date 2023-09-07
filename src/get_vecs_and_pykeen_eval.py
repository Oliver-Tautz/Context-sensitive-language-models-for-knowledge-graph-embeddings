import subprocess
from pathlib import Path
import argparse
import textwrap


def main(args):
    modelpath = args.modelpath# '/home/olli/remote/techfak/compute/Better_Knowledge_Graph_Embeddings/codex_models/tiny_bert_from_scratch_simple_codex_ep1000_vec200_walks_d=3_w=100_g=RANDOM_WALKS_dataset=MLM_mask-chance=0.33_mask-token-chance=1.0'
    entity_filepath = args.entity_filepath#'/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/codex_l_ents.txt'

    pykeen_train_triples = args.pykeen_train_triples#"/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/train.txt"
    pykeen_test_triples = args.pykeen_test_triples#"/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-l/test.txt"

    vector_filepath = f'{Path(modelpath).name}_{Path(entity_filepath).name}_vectors.txt'

    if not Path(vector_filepath).is_file():
        subprocess.run(
            ['python', 'vectors_fast.py', '--bert-path', modelpath, '--kg-path', entity_filepath, '--out-path',
             vector_filepath])
    subprocess.run(
        ['python', 'evaluate_embeddings_in_pykeen.py', '--train-triples', pykeen_train_triples, '--test-triples',
         pykeen_test_triples,
         '--vector-file', vector_filepath, '--model-name-pykeen', 'ERMLP', '--epochs', '25'])

    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--modelpath", help="Filepath of the config file. See example config.", type=str,
                        )

    parser.add_argument("--entity_filepath", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--pykeen_train_triples", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--pykeen_test_triples", help="Filepath of the config file. See example config.", type=str,
                        )



    args = parser.parse_args()

    main(args)
