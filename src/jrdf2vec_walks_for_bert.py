from utils import run_str
from pathlib import Path
import os

def generate_walks(tmpdir, dataset_file, outname, nproc, depth, no_walks_per_entity, generation_mode="RANDOM_WALKS_DUPLICATE_FREE"):
    print()
    if not Path(f'{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt').is_file():
        if not Path('./jrdf2vec-1.3-SNAPSHOT.jar').is_file():
            run_str("wget -q -nc https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar")
        print(tmpdir)
        run_str(
            f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
        run_str(f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {str(Path(tmpdir))} -mergeWalks -o {tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt")

        # remove unused tmpfiles
        for file in os.listdir(tmpdir):
            if Path(file).suffix == '.gz':
                os.remove(f'{tmpdir}/{file}')

    return f"{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt"

if __name__ == "__main__":

    # test some walks
    for d in [3,4,5,6,7]:
        generate_walks("/tmp/walks","FB15k-237/train.nt","fb15k-237",12,d,10)
