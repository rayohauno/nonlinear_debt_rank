
##########################################################
# To compute simple statistical quantities out of arrays
# of data-points (e.g. mean, std, quartiles, max, min, 
# etc.), one array per tuple-of-parameter-values. In 
# particular, it can be used to compute a function
#     ( tuple of parameters ) -> boxplot
# that can be used to plot curves combined with boxplots.
#
# To implement this module, we took ideas from the blog of
# John D. Cook:
#     http://www.johndcook.com/blog/standard_deviation/
##########################################################

import math
import numpy as np
import scipy as sp
from scipy import stats as sp_stats
from collections import defaultdict

class SimpleStats:
    def __init__(self):
        self.M=0.0
        self.S=0.0
        self.n = 0.0
        self._min=1000000000000000.0
        self._max=-1000000000000000.0
    def push(self,x):
        self.n += 1
        old_M = self.M
        self.M = old_M + (x-old_M)/self.n
        self.S = self.S + (x-old_M)*(x-self.M)
        self._min=min(x,self._min)
        self._max=max(x,self._max)
    def mean(self):
        if self.n > 0.0:
            return self.M
        else:
            return None
    def variance(self):
        if self.n > 1.0:
            return self.S / (self.n - 1.0) # std of the sample
#            return self.S / self.n        # exact std
        else:
            return None
    def std(self):
        var = self.variance()
        if var is None:
            return None
        return math.sqrt(var)
    def std_of_the_mean(self):
        var = self.variance()
        if var is None:
            return None
        return math.sqrt(var/self.n)
    def len(self):
        return self.n
    def min(self):
        return self._min
    def max(self):
        return self._max

class DictSimpleStats:
    def __init__(self):
        self._dict = defaultdict(lambda: SimpleStats())
    def __getitem__(self,key):
        if key in self._dict:
            return self._dict[key]
        return None
    def __setitem__(self,key,value):
         self._dict[key].push(value)
    def __iter__(self):
        for key in sorted(self._dict.keys()):
            yield key
            
class BoxPlot:
    def __init__( self , data = None ):
        self.reset()
        if data is not None:
            self.from_iterable( data )
            
    def reset( self ):
        self._flag_up_to_date = True
        self._data = []
        
    def push( self , d ):
        self._data.append( d )
        self._flag_up_to_date = False        
        
    def from_iterable( self , iterable ):
        for d in iterable:
            self.push( d )
            
    def _update( self ):
        if self._flag_up_to_date is True:
            return
        np_data               = np.array( self._data , dtype = np.double )
        self._mean            = np_data.mean()
        if self.len() > 1:
            self._std         = np_data.std()
            self._sem         = sp_stats.sem( np_data )
        else:
            self._std         = None
            self._sem         = None
        self._min             = np_data.min()
        self._max             = np_data.max()        
        self._first_quartile  = np.percentile( np_data , 25. ) # bottom of the box ; lower_quartile
        self._median          = np.percentile( np_data , 50. ) # second_quartile
        self._third_quartile  = np.percentile( np_data , 75. ) # top of the box ; upper_quartile
        self._IQR             = self._third_quartile - self._first_quartile
        self._whisker_up_IQR  = np_data[ np_data <= self._third_quartile + 1.5 * self._IQR ].max()
        self._whisker_low_IQR = np_data[ np_data >= self._first_quartile - 1.5 * self._IQR ].min()        
        
    def len( self ):
        return len( self._data )
        
    def mean( self ):
        self._update()    
        return self._mean
        
    def std( self ):
        self._update()
        return self._std
        
    def min( self ):
        self._update()
        return self._min
        
    def max( self ):
        self._update()
        return self._max        
        
    def first_quartile( self ):
        self._update()
        return self._first_quartile
        
    def median( self ):
        self._update()
        return self._median
        
    def third_quartile( self ):
        self._update()
        return self._third_quartile        
        
    def whisker_min( self ):
        self._update()
        return self._min
        
    def whisker_max( self ):
        self._update()
        return self._max
        
    def IQR( self ):
        self._update()
        return self._IQR
        
    def whisker_low_IQR( self ):
        self._update()        
        return self._whisker_low_IQR        
        
    def whisker_up_IQR( self ):
        self._update()        
        return self._whisker_up_IQR
        
    def sem( self ):
        """This is the Standard Error of the Mean (SEM).
        
        The SEM is computed as
        
            SEM = S / sqrt( N )
            
        where N is the number of points in the sample and
        
            S = frac{ 1 }{ N - 1 } sum_{ i = 1 }^N ( x_i - < x > )**2
            
        is the corrected sample standard deviation (i.e., the sample-based estimate of the standard deviation of the population). Notice how the estimated_std is obtained by a sum with N terms but "normalized" by N-1 instead of N. This difference is negligible for N large enough.
        
        References
        ----------
        [1] http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html
        [2] http://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
        [3] https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_mean_versus_standard_deviation
        [4] https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
        """
        self._update()
        return self._sem
            
class BoxPlotFunction:
    def __init__( self , list_xy_pairs = None ):
        self.reset()
        if list_xy_pairs is not None:
            self.from_list_of_xy_pairs( list_xy_pairs )
            
    def reset( self ):
        self._x_to_list_y = defaultdict( list )
        
    def len( self ):
        return len( self._x_to_list_y )
        
    def len_at( self , x ):
        if x in self._x_to_list_y:
            return len( self._x_to_list_y[ x ] )
        return 0
        
    def push( self , x , y ):
        y = float( y )
        self._x_to_list_y[ x ].append( y )
        
    def from_list_of_xy_pairs( self , list_xy_pairs ):
        for x , y in list_xy_pairs:
            self.push( x , y )
            
    def __getitem__( self , x ):
        if x in self._x_to_list_y:
            return BoxPlot( self._x_to_list_y[ x ] )
        else:
            return None
            
    def __call__( self , x ):
        return self[ x ]
        
    def __iter__( self ):
        for x in sorted( self._x_to_list_y.keys() ):
            yield x
