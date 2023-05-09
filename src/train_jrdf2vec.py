from utils import run_str
import os
from settings import VECTOR_SIZE
import multiprocessing
import gdown
import zipfile

# download jrdf2vec
run_str("wget -q -nc https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar")


gdown.download("https://drive.google.com/file/d/1pBnn8bjI2VkVvBR33DnvpeyocfDhMCFA/view?usp=share_link",
               output="fb15k-237.zip", fuzzy=True)

with zipfile.ZipFile("fb15k-237.zip", 'r') as zip_ref:
    zip_ref.extractall(".")


# clean directory
run_str("rm -rf walks")

# train
run_str(
    f"java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory walks -graph FB15k-237/train.nt -threads {multiprocessing.cpu_count() - 2} -dimension {VECTOR_SIZE}")
