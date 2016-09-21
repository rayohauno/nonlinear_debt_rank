import numpy as np
import pandas as pd

filein_A_ij="DATA_IN/p_0.05/2008exposure_net1.txt"
filein_bank_specific_data="DATA_IN/bankscopes/bank_data_year2008.dat"

print '# Loading bank-specific data from',filein_bank_specific_data

#bank_id total_assets    equity  inter_bank_assets       inter_bank_liabilities  bank_name
#1       2527465000.0    95685000.0      159769000.0     137316000.0     "HSBC Holdings Plc"
#2       2888526820.49   64294758.6352   96239646.83     260571978.577   "BNP Paribas"

df_bank_specific_data = pd.read_csv(filein_bank_specific_data,sep='\t')
A_i = df_bank_specific_data['total_assets'].values
E_i = df_bank_specific_data['equity'].values
IB_A_i = df_bank_specific_data['inter_bank_assets'].values
IB_L_i = df_bank_specific_data['inter_bank_liabilities'].values

N = len(A_i)

print '# Loading inter-bank assets from',filein_A_ij

#source  target  exposure
#1       2       18804300.1765828
#1       3       593429.0704464162
#1       4       7180905.941936611
#1       5       13568931.097857257

A_ij = np.zeros( (N,N) , dtype=np.double )
df_edges = pd.read_csv(filein_A_ij,sep='\t')
for _,i,j,w in df_edges.itertuples():
    ii = i - 1
    jj = j - 1
    assert ii >= 0 and ii < N
    assert jj >= 0 and jj < N
    assert w > 0
    A_ij[ii,jj] = w

print '# 1.i 2.A_i[i] 3.IB_A_i[i] 4.A_ij[i,:].sum() 5.A_ij[:,i].sum() 6.OK(IB_A_i[i],A_ij[i,:].sum())'
for i in xrange(N):
    def OK(x,y):
        tot=0.5*(abs(x)+abs(y))
        if x==0.0 or y==0.0 or tot==0.0:
            return 'OK'
        if abs(x-y)/tot < 0.05:
            return 'OK'
        return 'BAAAD'
    print i,A_i[i],IB_A_i[i],A_ij[i,:].sum(),A_ij[:,i].sum(),OK(IB_A_i[i],A_ij[i,:].sum())
