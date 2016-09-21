import sys
import os
import doctest
#sys.path.append('../')
#from hierpart import HierarchicalPartition
#from hierpart import save_hierarchical_partition, load_hierarchical_partition
#from hierpart import sub_hierarchical_mutual_information, hierarchical_mutual_information, normalized_hierarchical_mutual_information
#from hierpart import hierpart_doctestme

###############################
# HierarchicalPartition Tests #
###############################

def run_doctests():
#    sys.path.append('../')
    m = __import__('debt_rank')
    doctest.testmod(m)

if __name__=='__main__':
    run_doctests()

