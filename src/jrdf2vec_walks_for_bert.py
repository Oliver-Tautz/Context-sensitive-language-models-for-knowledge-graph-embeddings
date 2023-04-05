from utils import run_str
from pathlib import Path

def generate_walks(tmpdir, dataset_file, outname, nproc, depth, no_walks_per_entity, generation_mode="RANDOM_WALKS_DUPLICATE_FREE"):
    if not Path(f'{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt').is_file():

        run_str(
            f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {tmpdir} -graph {dataset_file} -onlyWalks -threads {nproc} -depth {depth} -numberOfWalks {no_walks_per_entity} -walkGenerationMode {generation_mode}")
        run_str(f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory {tmpdir} -mergeWalks -o {outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt")
        run_str(f"rm -rvf {tmpdir}/*.gz")
    return f"{outname}_{generation_mode}_d={depth}_w={no_walks_per_entity}.txt"

if __name__ == "__main__":

    # test some walks
    for d in [3,4,5,6,7]:
        generate_walks("/tmp/walks","FB15k-237/train.nt","fb15k-237",12,d,10)
