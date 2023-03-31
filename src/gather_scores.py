import pandas as pd
from pathlib import Path
from os import listdir


def get_dataframe(csv_file):
    df = pd.read_csv(file, index_col=0)
    df = df[df['epoch']=='test']
    df = df[['mr','mrr','hits@10','hits@5','hits@3','hits@1']]
    return df

basepath = Path('/home/olli/gits/Better_Knowledge_Graph_Embeddings/tiny_bert_from_scratch_simple_ep25_vec200/link_prediction/data/')
tempdir = Path('/tmp')
file = basepath / listdir(basepath)[0]
df = get_dataframe(file)
df['model'] = file.stem
for file in listdir(basepath)[1:]:
    if file == 'testscores.csv':
        continue
    file = basepath / file
    new_df = get_dataframe(file)
    new_df['model'] = file.stem
    df = df.merge(new_df,'outer')



df.to_csv(basepath / 'testscores.csv',index=False,sep='\t')
df.to_string(basepath / 'testscores.csv',index=False,sep='\t')



