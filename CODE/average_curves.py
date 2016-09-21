import sys
import numpy as np
import pandas as pd
import simplestats as st

list_filein = sys.argv[1]

dst_num_active = st.DictSimpleStats()
dst_num_stressed = st.DictSimpleStats()
dst_num_defaulted = st.DictSimpleStats()
dst_H = st.DictSimpleStats()

input_files=[]
with open(list_filein,'r') as fh_filein:
    for filein in fh_filein.readlines():
        filein=filein.replace('\n','')
        input_files.append(filein)
        with open(filein,'r') as fh:
            for line in fh.readlines():
                if '#' in line:
                    continue
                cols = line.split()
                if len(cols) < 2:
                    continue
                # 1.t 2.num_active 3.num_stressed 4.num_defaulted 5.min_h 6.mean_h 7.max_h 8.H
                t = int(cols[0])
                num_active = float(cols[1])
                num_stressed = float(cols[2])
                num_defaulted = float(cols[3])
                dst_num_active[t] = num_active
                dst_num_stressed[t] = num_stressed
                dst_num_defaulted[t] = num_defaulted
                try:
                    H = float(cols[7])
                    dst_H[t] = H
                except:
                    pass   

print '# input_files',input_files[:2]
print '# 1.t 2.len_stats 3.mean_num_active 4.std_num_active 5.mean_num_stressed 6.std_num_stressed 7.mean_num_defaulted 8.std_num_defaulted 9.H_mean 10.H_std'
for t in dst_num_active:
    H_mean=None
    H_std=None
    if dst_H[t].len()>0:
        H_mean=dst_H[t].mean()
        H_std=dst_H[t].std()
    print t, dst_num_active[t].len(), dst_num_active[t].mean(), dst_num_active[t].std(), dst_num_stressed[t].mean(), dst_num_stressed[t].std(), dst_num_defaulted[t].mean(), dst_num_defaulted[t].std(), H_mean, H_std
