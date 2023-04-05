from utils import run_str
from pathlib import Path
import os

def generate_walks(tmpdir, dataset_file, outname, nproc, depth, no_walks_per_entity, generation_mode="RANDOM_WALKS_DUPLICATE_FREE"):
    if not Path(f'{tmpdir}/{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt').is_file():
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
