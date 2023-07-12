import pandas as pd
from pathlib import Path
from os import listdir
from tabulate import tabulate
import argparse

def get_dataframe(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    df = df[df['epoch'] == 'test']
    df = df[['mr', 'mrr', 'hits@10', 'hits@3', 'hits@1']]
    return df




def tabulate_results(path):
    path = Path(path)
    rdf2vec_names = ["rdf2vec_VectorReconstructionNet_standard_train_data.csv",
                     "rdf2vec_ClassicLinkPredNet_standard_train_data.csv",
                     "rdf2vec_VectorReconstructionNet_derived_train_data.csv",
                     "rdf2vec_ClassicLinkPredNet_derived_train_data.csv"]

    rdf2vec_values = [[1153.2833, 0.2083, 0.286, 0.226, 0.165],
                      [419.6589, 0.1172, 0.252, 0.119, 0.053],
                      [1146.7583, 0.2054, 0.281, 0.222, 0.162],
                      [427.9076, 0.1228, 0.258, 0.127, 0.057]]

    rdf2vec_table = [[name] + values for name,values in zip(rdf2vec_names,rdf2vec_values)]


    first = True
    table = rdf2vec_table



    for bert_model in listdir(path):
        datapath = path / bert_model / "link_prediction"
        if datapath.is_dir():
            for embedding_type in listdir(datapath):
                embedding_path = datapath / embedding_type
                for file in listdir(embedding_path / "data"):
                    filepath = embedding_path / "data" / file

                    df = get_dataframe(filepath)
                    if first:
                        header = ['model'] + df.columns.tolist()
                        first = False

                    for ix, row in df.iterrows():
                        list = row.tolist()

                    table.append([f"{bert_model}_{embedding_type}_{file}"] + list)

    print(tabulate(sorted(table,key=lambda x: x[1]), headers=header))

# df.to_csv(basepath / 'testscores.csv',index=False,sep='\t')
# print(df.to_string(basepath / 'testscores.csv',index=False,sep='\t'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to prettyprint results from multiple experiments.",

    )

    parser.add_argument("--path", help="Filepath of trained and evaluated models.", type=str,
                        default=".")


    args = parser.parse_args()
    tabulate_results(args.path)