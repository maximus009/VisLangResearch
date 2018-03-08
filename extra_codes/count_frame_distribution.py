import __init__

base_path = __init__.base_path

import os, sys, numpy as np
from glob import glob

from collections import Counter

framesPath = os.path.join(base_path, 'frames')


cnt = Counter()

def counter(i):
    print i
    path = glob(os.path.join(framesPath,str(i),'*'))
    l = len(path)/2
    return l

from multiprocessing import Pool

p = Pool(8)

hist = p.map(counter,range(5043))

hist = Counter(hist)

f = open('histogram_frame_count.txt','w')
for h in hist:
    f.write(str(h)+':'+str(hist[h])+'\n')
f.close()
