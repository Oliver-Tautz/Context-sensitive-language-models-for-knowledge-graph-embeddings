import random
import time
import numpy as np


class Profiler():

    """
    Simple class to keep track of the time of different operations.
    """
    def __init__(self,timers):

        self.timerdict = {}
        for timer in timers:
            self.timerdict[timer] = []

    def timer_start(self,timer):
        self.timerdict[timer].append(time.time())

    def timer_stop(self,timer):
        self.timerdict[timer][-1]=time.time()-self.timerdict[timer][-1]

    def timer_eval(self,timer):
        t = self.timerdict[timer]
        self.timerdict[timer] = {'total' : np.sum(t),'mean' : np.mean(t),'std' : np.std(t)}

    def eval(self):
        for k in self.timerdict:
            self.timer_eval(k)

    def get_profile(self):
        return self.timerdict


if __name__ == "__main__":
    p = Profiler(['i','j'])

    for i in range(3):
        p.timer_start('i')
        time.sleep(random.randint(0,3))
        p.timer_stop('i')

    for j in range(3):
        p.timer_start('j')
        time.sleep(random.randint(0,6))
        p.timer_stop('j')

    p.eval()
    print(p.timerdict)