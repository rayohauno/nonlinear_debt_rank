import sys
import numpy as np
#import pandas as pd
#import simplestats as st

#filein = sys.argv[1]

#filein_TEMPLATE='DATA_IN/NLDR_v1_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB.dat'
#fileout_TEMPLATE='DATA_OUT/NLDR_v1_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB_average_curves.dat'

filein_TEMPLATE='DATA_IN/NLDR_v2_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB.dat'
fileout_TEMPLATE='DATA_OUT/NLDR_v2_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB_average_curves.dat'

#filein_TEMPLATE='DATA_IN/NLDR_v3_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB.dat'
#fileout_TEMPLATE='DATA_OUT/NLDR_v3_pPPP_yearYYY_netNNN_rhoRRR_alphaAAA_betaBBB_average_curves.dat'

np_num_active = np.empty((100,20),dtype=np.double)
np_num_stressed = np.empty((100,20),dtype=np.double)
np_num_defaulted = np.empty((100,20),dtype=np.double)
np_count = np.empty(20,dtype=np.double)

for AAA in '0.0 0.25 0.5 0.75 1.0 2.0 4.0 8.0 16.0'.split():
    for YYY in '2008'.split(): # 2009 2010 2011 2012 2013'.split():
        for NNN in xrange(1,2): #xrange(1,101):
            NNN=str(NNN)
            for PPP in '0.05'.split(): # 0.2 1.0'.split():
                for BBB in '0.05 0.2 0.5'.split():
                    for RRR in '0.01'.split():
                        filein=filein_TEMPLATE.replace('AAA',AAA).replace('YYY',YYY).replace('NNN',NNN).replace('PPP',PPP).replace('BBB',BBB).replace('RRR',RRR)
                        with open(filein,'r') as fh:
                            np_num_active[:,:]=0.0
                            np_num_stressed[:,:]=0.0
                            np_num_defaulted[:,:]=0.0
                            np_count[:]=0.0
                            for line in fh.readlines():
                                if '#' in line:
                                    continue
                                cols = line.split()
                                if len(cols) < 2:
                                    continue
                                # 1.step 2.num_active 3.num_stressed 4.num_defaulted 5.min_h 6.mean_h 7.max_h
                                step = int(cols[0])
                                count = np_count[step]
                                np_count[step]+=1
                                np_num_active[count,step] = float(cols[1])
                                np_num_stressed[count,step] = float(cols[2])
                                np_num_defaulted[count,step] = float(cols[3])
                        fileout=fileout_TEMPLATE.replace('AAA',AAA).replace('YYY',YYY).replace('NNN',NNN).replace('PPP',PPP).replace('BBB',BBB).replace('RRR',RRR)
                        with open(fileout,'w') as fhw:
                            print >>fhw,'# 1.step 2.mean_num_active 3.std_num_active 4.mean_num_stressed 5.std_num_stressed 6.mean_num_defaulted 7.std_num_defaulted'
                            for _step in xrange(step):
                                print >>fhw,_step, np_num_active[:,_step].mean(), np_num_active[:,_step].std(), np_num_stressed[:,_step].mean(), np_num_stressed[:,_step].std(), np_num_defaulted[:,_step].mean(), np_num_defaulted[:,_step].std()
