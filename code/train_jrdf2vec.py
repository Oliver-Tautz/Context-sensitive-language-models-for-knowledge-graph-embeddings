from utils import run_str
import os
from settings import VECTOR_SIZE
import multiprocessing


# download jrdf2vec
run_str("wget -q -nc https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar")

# download dataset
run_str("wget  -q -nc --no-check-certificate https://docs.google.com/uc?export=download&id=1pBnn8bjI2VkVvBR33DnvpeyocfDhMCFA -O fb15k-237_nt.zip")

# unzip dataset
run_str("unzip -n fb15k-237_nt.zip")

# clean directory
run_str("rm -rf walks")

# train
run_str(f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory walks -graph FB15k-237/train.nt -threads {multiprocessing.cpu_count()-2} -dimension {VECTOR_SIZE}")

