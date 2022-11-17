# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from time import perf_counter, sleep
from collections import defaultdict
import numpy as np
import pickle


# %%
def pickle_save(obj,filename):
    file = open(filename,'wb') 
    pickle.dump(obj,file)
    file.close()
  


# %%
class PerfTimer():
    def __init__(self):
        self.stats_to_track = defaultdict(list)

    def start(self):
        self.start_time = perf_counter()
        self.current_time = self.start_time
    def track(self,stat):

            new_time = perf_counter()
            self.stats_to_track[stat].append(new_time-self.current_time)
            self.current_time = new_time
    def stats(self):
        return self.stats_to_track
    def stats_sum(self):
        return {key: sum(value) for key,value in  self.stats().items()}
    def stats_mean(self):
        return {key: np.mean(value) for key,value in  self.stats().items()}
    def stats_f(self,f):
        return {key: f(value) for key,value in  self.stats().items()}
        
    

# %%
if __name__ == "__main__":
    pt = PerfTimer()
    pt.start()
    for i in range(10):
        sleep(0.4)
        pt.track('first_operation')
        sleep(1.2)
        pt.track('second_operation')
    print(pt.stats())
    print(pt.stats_sum())
    print(pt.stats_mean())
    print(pt.stats_f(max))


