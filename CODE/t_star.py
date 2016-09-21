import sys
import numpy as np
import pandas as pd
import simplestats as st
from collections import defaultdict

list_filein = sys.argv[1]

num_samples=defaultdict(int)
dst_t_star=st.DictSimpleStats()
dst_A_at_t_star=st.DictSimpleStats()
dst_S_at_t_star=st.DictSimpleStats()
dst_D_at_t_star=st.DictSimpleStats()
dst_H_at_t_star=st.DictSimpleStats()

input_files=[]
with open(list_filein,'r') as fh_filein:
    for filein in fh_filein.readlines():
        filein=filein.replace('\n','')
        input_files.append(filein)
        with open(filein,'r') as fh:
            year=None
            p=None
            rho=None
            p_i_shock=None
            x_shock=None
            alpha=None
            t_star=None
            sample=None
            for line in fh.readlines():
# year 2009
# net 53
# p 0.05
# rho 0.0
# p_i_shock 0.05
# h_shock None
# x_shock 0.005
# alpha 3.9
                cols=line.split()
                if '# year' in line:
                    year=int(cols[2])
                elif '# p ' in line:
                    p=float(cols[2])
                elif '# rho' in line:
                    rho=float(cols[2])
                elif '# p_i_shock' in line:
                    p_i_shock=float(cols[2])
                elif '# x_shock' in line:
                    x_shock=float(cols[2])
                elif '# alpha' in line:
                    alpha=float(cols[2])
                elif '# sample' in line:

                    params=year,p,rho,p_i_shock,x_shock,alpha
                    num_samples[params]+=1

                    sample=int(cols[2])
                    t_star=None

                elif '# t_star' in line:
                    t_star=int(cols[2])

                    assert year is not None
                    assert p is not None
                    assert rho is not None
                    assert p_i_shock is not None
                    assert x_shock is not None
                    assert alpha is not None

                    dst_t_star[params]=t_star
                    dst_A_at_t_star[params]=A
                    dst_S_at_t_star[params]=S
                    dst_D_at_t_star[params]=D
                    dst_H_at_t_star[params]=H

                if '#' in line:
                    #print line,
                    continue

                if len(cols) < 2:
                    continue

                # 1.t 2.num_active 3.num_stressed 4.num_defaulted 5.min_h 6.mean_h 7.max_h 8.H
                t = int(cols[0])
                A = float(cols[1])
                S = float(cols[2])
                D = float(cols[3])
                H = float(cols[7])


print ' '.join([ str(i+1)+'.'+c for i,c in enumerate('year,p,rho,p_i_shock,x_shock,alpha,num_samples,dst_t_star_len,t_star,t_star_std,A_at_t_star,A_at_t_star_std,S_at_t_star,S_at_t_star_std,D_at_t_star,D_at_t_star_std,H_at_t_star,H_at_t_star_std'.split(',')) ])

for params in sorted(num_samples.keys()):

    year,p,rho,p_i_shock,x_shock,alpha=params

    print year,p,rho,p_i_shock,x_shock,alpha,

    try:
        st_t_star_len=dst_t_star[params].len()
    except:
        print '# WARNING: dst_t_star[params] == None for params =',year,p,rho,p_i_shock,x_shock,alpha
        continue

    print num_samples[params],

    print st_t_star_len,
    print dst_t_star[params].mean(),
    print dst_t_star[params].std(),
    print dst_A_at_t_star[params].mean(),
    print dst_A_at_t_star[params].std(),
    print dst_S_at_t_star[params].mean(),
    print dst_S_at_t_star[params].std(),
    print dst_D_at_t_star[params].mean(),
    print dst_D_at_t_star[params].std(),
    print dst_H_at_t_star[params].mean(),
    print dst_H_at_t_star[params].std()
