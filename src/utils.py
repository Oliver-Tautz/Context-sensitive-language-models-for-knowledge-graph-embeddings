import subprocess
import pickle
from time import perf_counter
from collections import defaultdict
import numpy as np


VERBOSE=1

def run_str(string,silent = False):
    if not silent:
        return subprocess.run(string.split())
    else:
        return subprocess.run(string.split(),capture_output=True)


def pickle_save(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()

def batch(iterable, n=1):
    # great! taken from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def choose_many_multiple(arrs, n):
    l = len(arrs[0])
    for a in arrs:
        assert len(a) == l, 'Arres not of same length ! :('

    ix = np.random.choice(range(len(a)), n)

    return [np.array(a)[ix] for a in arrs]


def choose_many(a, n):
    ix = np.random.choice(range(len(a)), n)
    return np.array(a)[ix]


def choose(a):
    L = len(a)

    i = np.random.randint(0, L)

    return a[i]

def map_keyed_vectors(word_vectors, iterable):
    """
    for some reason faster than native call :O
    """
    return np.array(list(word_vectors.get_vector(x) for x in iterable))

def verbprint(str):
    if VERBOSE:
        print(str)
def merge_lists(l0,l1):
    l_new = []
    while l0 and l1:
        l_new.append(l0.pop(0))
        l_new.append(l1.pop(0))

    if l0:
        l_new = l_new+l0
    else:
        l_new = l_new+l1

    return l_new
class PerfTimer():
    def __init__(self):
        self.stats_to_track = defaultdict(list)

    def start(self):
        self.start_time = perf_counter()
        self.current_time = self.start_time

    def track(self, stat):
        new_time = perf_counter()
        self.stats_to_track[stat].append(new_time - self.current_time)
        self.current_time = new_time

    def stats(self):
        return self.stats_to_track

    def stats_sum(self):
        return {key: sum(value) for key, value in self.stats().items()}

    def stats_mean(self):
        return {key: np.mean(value) for key, value in self.stats().items()}

    def stats_f(self, f):
        return {key: f(value) for key, value in self.stats().items()}

