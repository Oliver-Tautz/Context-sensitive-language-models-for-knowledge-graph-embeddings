from utils_bert_models import BertKGEmb
from utils_graph import parse_kg_fast
import argparse
import textwrap
from urllib.parse import urljoin
from tqdm import tqdm
def write_vectors_to_file(dic, vec_filepath):

    f = open(vec_filepath, 'w')
    for k, v in tqdm(dic.items(),total=len(dic)):
        f.write(k)
        f.write(" ")
        for x in v:
            f.write(f"{x.item():f} ")
        f.write("\n")



def get_embeddings_from_kg(kgpath,modelpath):
    print('parsing graph')
    entities, predicates, edges, predicate_ix = parse_kg_fast(kgpath)
    #print(entities)
    print('loading model')
    bert = BertKGEmb(modelpath)
    print('predict embeddings')
    entity_embs = bert.get_embeddings(entities)
    predicate_embs = bert.get_embeddings(predicates)

    entity_embs = dict(zip(entities,entity_embs))
    predicate_embs = dict(zip(predicates,predicate_embs))
    entity_embs.update(predicate_embs)
    return entity_embs

def get_embeddings_from_kg_fast(kgpath,modelpath,base_url=None,datapath=None,depth=3):
    print('parsing graph')
    #entities, predicates, edges, predicate_ix = parse_kg_fast(kgpath)

    f = open(kgpath,'r')
    entities = f.readlines()
    entities = [x.strip() for x in entities]

    #for line in f.readlines:
    #    en
    #print(entities)
    print('loading model')
    if datapath != None:
        bert = BertKGEmb(modelpath,datapath=datapath,depth=depth,mode='walks')
    else:
        bert = BertKGEmb(modelpath)
    print('predict embeddings')
    if base_url!= None:
        entity_embs = bert.get_embeddings([urljoin(base_url,e) for e in entities])
    else:
        entity_embs = bert.get_embeddings(entities)
    #predicate_embs = bert.get_embeddings(predicates)

    entity_embs = dict(zip(entities,entity_embs))
    #predicate_embs = dict(zip(predicates,predicate_embs))
    #entity_embs.update(predicate_embs)
    return entity_embs

def write_vector_file(kg_path, bert_path, vectorfile_path,base_url):

    embs = get_embeddings_from_kg_fast(kg_path, bert_path,base_url)
    print('write file.')

    write_vectors_to_file(embs, vectorfile_path)


def main(args):
    write_vector_file(args.kg_path,args.bert_path,args.out_path,args.add_base_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--bert-path", help="Filepath of the config file. See example config.", type=str,
                        )

    parser.add_argument("--kg-path", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--out-path", help="Filepath of the config file. See example config.", type=str,
                        )
    parser.add_argument("--add-base-url", help="Filepath of the config file. See example config.", type=str, default=None
                        )

    parser.add_argument('--bert-walks', type=str, default=False)
    parser.add_argument('--bert-best-eval', action='store_true',default=False,
                        help='use checkpoint with best eval loss using early stopping.')
    parser.add_argument('--bert-mode-depth', type=int, default=3,
                        help="sdepth of walks for embedding.")


    args = parser.parse_args()



    print("passed args: " + str(args))
    main(args)

