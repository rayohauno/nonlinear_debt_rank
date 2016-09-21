import numpy as np

N = 183
h = np.zeros(N,dtype=np.double)

prob_touched = 0.2

for i in xrange(len(h)):
    if np.random.random() < prob_touched:
        h[i] = 1.0

for i in xrange(len(h)):
    print h[i]
