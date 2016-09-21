import sys
import datetime
import platform
import numpy as np
#import random as rnd
from collections import defaultdict

def emptystr(s):
    if s is None:
        return True
    return not str(s).replace('\n','')

def rreplace(s,old,new):
    s=str(s)
    old=str(old)
    new=str(new)
    return s[::-1].replace(old[::-1],new[::-1],1)[::-1]

def flush(fh=None):
    if fh is None:
        sys.stdout.flush()
    else:
        sys.stdout.flush(fh)

class fh:
    def __init__(self,filename):
        self._filename = str(filename)
        self._fh = open(self._filename,'r')
        self._line = self._fh.readline()

    def __iter__(self): 
        return self

    def next(self):
        if self._line != '':
            line = self._line
            self._line = self._fh.readline()
            return line.replace('\n','')
        else:
            self._fh.close()
            raise StopIteration

class fhcols:
    def __init__(self,filename,splitby=None):
        if splitby is not None:
            self._splitby = str(splitby)
        else:
            self._splitby = None
        self._filename = str(filename)
        self._fh = open(self._filename,'r')
        self._line = self._fh.readline()

    def __iter__(self):
        return self

    def next(self):
        if self._line != '':
            line = self._line
            self._line = self._fh.readline()
            if '#' in line:
                return ['#',line.replace('\n','')]
            else:
                if self._splitby is None:
                    return line.split()
                else:
                    return line.split(self._splitby)
        else:
            self._fh.close()
            raise StopIteration

def str_to_str(s):
    if s == 'None':
        return None
    else:
        return s

def str_to_float(s):
    if s == 'None':
        return None
    else:
        return float(s)

def str_to_int(s):
    if s == 'None':
        return None
    else:
        return int(s)

def str_to_bool(s):
    if s == 'True' or s == 'T' or s == 'true':
        return True
    elif s == 'False' or s == 'F' or s == 'false':
        return False
    else:
        return None

##############################################################################
# New Comand Line Tools ######################################################
##############################################################################

def _pretty_print_list_2_str(l):
    _str=''
    if len(l)>1:
        _str+='<'+str(l[0])+'>'
        for e in l[1:]:
            _str+=','+str(e)
        return _str
    return str(l[0])

class PathFilenameExtension:
    def __init__(self,pfne=None,path=None,filename=None,extension=None,attr=None,attrval=None):

        assert not ( pfne is None and filename is None ), 'ERROR in __init__(): assert not ( pathfile_name_extension is None and filename is None )'

        if pfne is None:
            self._pfne,self._path,self._filename,self._extension=self._path_filename_extension_2_curated_pathfilenamextension_and_path_filename_extension(path,filename,extension)
        else:
            self._pfne,self._path,self._filename,self._extension=self._pathfilenameextension_2_curated_pathfilenamextension_and_path_filename_extension(pfne)

        if attr is not None:
            try:
                attr=str(attr)
            except:
                assert False, 'ERROR in _curate__init__(): attr is not a str'
            #
            assert attrval is not None,"ERROR in __init__(): attrval is None or attrval==''"
            try:
                attrval=str(attrval)
            except:
                assert False, 'ERROR in _curate__init__(): attrval is not a str'
            #
            self._add_attr_val_2_filename(attr,attrval)

    def _clean_str(self,s):
        try:
            s=str(s)
            assert not ( ' ' in s )
            s.replace('\n','')
        except:
#            pass
            raise
        return s

    def _curate_extension(self,extension):
        if extension is None or emptystr(extension):
            return ''

        try:
            extension=self._clean_str(extension)
        except:
            assert False, 'ERROR in _curate_extension(): extension is not a valid str'
        if '.'==extension[0]:
            extension=extension[1:]
        assert '.' not in extension
        return '.'+extension

    def _curate_filename(self,filename):
        assert not ( ( filename is None ) or emptystr(filename) ), "ERROR in _curate_filename(): assert not ( ( filename is None ) or ( filename == '' ) )"

        try:
            filename=self._clean_str(filename)
        except:
            assert False, 'ERROR in _curate_filename(): filename is not a valid str'
        assert '/' not in filename
        return filename

    def _curate_path(self,path):
        if path is None or emptystr(path):
            return ''

        try:
            path=self._clean_str(path)
        except:
            assert False, 'ERROR in _curate_path(): path is not a valid str'
 
        path=path.replace('//////','/')
        path=path.replace('/////','/')
        path=path.replace('////','/')
        path=path.replace('///','/')
        path=path.replace('//','/')

        if path[-1]!='/':
            path=path+'/'

        return path

    def _pathfilenameextension_2_curated_pathfilenamextension_and_path_filename_extension(self,path_filename_extension):
        try:
            pfe=self._clean_str(path_filename_extension)
        except:
            assert False, 'ERROR in _pathfilenameextension_2_curated_pathfilenamextension_and_path_filename_extension(): path_filename_extension is not a str'

        tmp=pfe.split('.')
        if len(tmp)==1:
            extension=''
        else:
            extension=tmp[-1]
        extension=self._curate_extension(extension)

#        tmp=pfe[::-1].replace(extension[::-1],'',1)[::-1]
        tmp=rreplace(pfe,extension,'')
        tmp2=tmp.split('/')
        if len(tmp2)==1:
            filename=tmp
            filename=self._curate_filename(filename)
            path=''
        else:
            filename=tmp2[-1]
            filename=self._curate_filename(filename)
#            path=tmp[::-1].replace(filename[::-1],'',1)[::-1]
            path=rreplace(tmp,filename,'')
            path=self._curate_path(path)

        return self._path_filename_extension_2_curated_pathfilenamextension_and_path_filename_extension(path,filename,extension)

    def _path_filename_extension_2_curated_pathfilenamextension_and_path_filename_extension(self,path,filename,extension):
        path=self._curate_path(path)
        filename=self._curate_filename(filename)
        extension=self._curate_extension(extension)
        return path + filename + extension, path, filename, extension     

    def _curate_attr_val(self,attr,val):

        assert not ( ( attr is None ) or ( attr == '' ) ), "ERROR in _curate_attr_val(): assert not ( ( attr is None ) or ( attr == '' ) )"
        try:
            attr=str(attr)
        except:
            assert False, 'ERROR in _curate_attr(): attr is not a str'
        assert '/' not in attr
        assert '.' not in attr

        assert not ( ( val is None ) or ( val == '' ) ), "ERROR in _curate_attr_val(): assert not ( ( val is None ) or ( val == '' ) )"
        try:
            val=str(val)
        except:
            assert False, 'ERROR in _curate_val(): val is not a str'
        assert '/' not in val

        return attr,val

    def _add_attr_val_2_filename(self,attr,val):
        attr,val=self._curate_attr_val(attr,val)
        self._filename+='_'+attr+val

    def _is_valid(self):
#        print 'TODO'
        return True

    def __invert__(self):
#        if self._extension!='':
#            return self._path+self._filename+'.'+self._extension
#        return self._path+self._filename
        return self._pfne

    def copy(self):
        return PathFilenameExtension(self.__invert__())
    def replicate(self,path=None,filename=None,extension=None,attr=None,attrval=None):
        if path is None:
            path=self._path
        if filename is None:
            filename=self._filename
        if extension is None:
            extension=self._extension
        return PathFilenameExtension(path=path,filename=filename,extension=extension,attr=attr,attrval=attrval)

    def show(self,verbose=True):
        print self.__invert__()
        if verbose:
            print '# PATH',self.path()
            print '# FILENAME',self.filename()
            print '# EXTENSION',self.extension()
    
    def __repr__(self):
        return 'PathFilenameExtension:'+self.__invert__()

    def path(self):
        return self._path
    def filename(self):
        return self._filename
    def extension(self):
        return self._extension

    def __eq__(self,x):
        if not isinstance(x,PathFilenameExtension):
            try:
                x=PathFilenameExtension(str(x))
            except:
                return False
        if self.path()!=x.path():
            return False
        if self.filename()!=x.filename():
            return False
        if self.extension()!=x.extension():
            return False
        return True

class Option:
    def __init__(self,keywords,values=None,description=''):
        """
        In:
            keywords : a str, of a list of str. The keywords by which the option is identified.
            values : the list of values the option can take, if provided. 
                     By default, the first value in the list, is the default value.
                     If None is provided, then, by default it becomes ['False','True']
                     If 'ANY' is provided as part of the list, then, the option can take any arbitrary value.
        """
        if isinstance(keywords,str):
            self._keywords=[keywords]
        else:
            try:
                self._keywords=[str(kw) for kw in keywords]
            except:
                print "ERROR: keywords has not the appropriate format. A str, or a list of str's"
                assert False

        if values is None:
            self._values=['False','True']
        elif values=='ANY':
            self._values=['ANY']
        else:
            try:
                self._values=[str(v) for v in values]
            except:
                print "ERROR: impossible to create list of values from values =",values
                assert False

        self._description=str(description)
    def keywords(self):
        return self._keywords
    def type(self):
        return self._type
    def values(self):
        return self._values
    def default(self):
        try:
            return self._values[0]
        except:
            return None
    def name(self):
        return self._keywords[0]
    def has_keyword(self,keyword):
        if str(keyword) in self._keywords:
            return True
        return False
    def contains_value(self,value):
        if 'ANY' in self._values:
            return True
        if str(value) in self._values:
            return True
        return False
    def show(self,verbose=True):
        if verbose:
            print _pretty_print_list_2_str(self._keywords)+':'+_pretty_print_list_2_str(self._values)
        else:
            print self._keywords[0]+':'+_pretty_print_list_2_str(self._values)
    def __repr__(self):
        return 'Option:'+str(self._keywords[0])
#    def __str__(self):
#
_help_option=Option( ['help','h','-h','--help'],values=['False','True'], description='The help option. It shows this message')

class CommandLine:
    def __init__(self,options,extra_help_message=''):

        self._option_idx_2_mentioned=defaultdict(lambda: False)
        self._option_idx_2_chosen_value={}

        if isinstance(options,Option):
            self._options=[options]
        else:
            self._options=list(options)

        for option in self._options:
            assert isinstance(option,Option),'ERROR: option is not of type Option.'

        self._options.append( _help_option )

        self._extra_help_message=str(extra_help_message)

        self._keyword_2_option_idx={}
        for option_idx,option in enumerate(self._options):
            _value=option.default()
            if _value=='ANY':
                self._option_idx_2_chosen_value[option_idx]=None
            else:
                self._option_idx_2_chosen_value[option_idx]=_value
            for keyword in option.keywords():
                assert keyword not in self._keyword_2_option_idx.keys(),'ERROR: duplicate keywords in self._options'
                self._keyword_2_option_idx[keyword]=option_idx

        self._args=sys.argv[1:]
        for arg in self._args:

            try:
                keyword,value=arg.split('=')
            except:
                keyword=arg
                value=None

            if value=='ANY':
                print 'ERROR: ANY is a reserved word, and it cannot be used as a value for any Option.'
                self.print_usage(1)

            option_idx=self.keyword_2_option_idx(keyword)
            option=self.keyword_2_option(keyword)

            self._option_idx_2_mentioned[option_idx]=True

            if value is None:
                if not option.contains_value('True'):
                    print 'ERROR: value is None for the ',option,' which has a non empty, non boolean, value list.'
                    self.print_usage(1)
                self._option_idx_2_chosen_value[option_idx]='True'
            else:
                if not option.contains_value(value):
                    print 'ERROR: value',value,'is not available in',option
                    self.print_usage(1)
                self._option_idx_2_chosen_value[option_idx]=value
  
        if self._option_idx_2_mentioned[self.keyword_2_option_idx('help')]:
            self.print_usage(0)

    def keyword_2_option_idx(self,keyword):
        try:
            return self._keyword_2_option_idx[keyword]
        except:
            print 'ERROR: keyword',keyword,'not macching any option in',self._options
            self.print_usage(1)
    def keyword_2_option(self,keyword):
        return self._options[self.keyword_2_option_idx(keyword)]
    def __getitem__(self,keyword):
        option_idx=self.keyword_2_option_idx(keyword)
        value=self._option_idx_2_chosen_value[option_idx]
        if value is None:
            option=self._options[option_idx]
            print 'ERROR: value is None for',option
            self.print_usage(1)
        return value
    def print_usage(self,error_int=0,verbose=False):
        if error_int!=0:
            print 'ERROR_STATUS',error_int
        print 'USAGE [./this.py]'
        for option in self._options:
            option.show(verbose=verbose)
        if self._extra_help_message!='':
            print 'NOTICE:'
            print self._extra_help_message
        sys.exit(error_int)

##############################################################################
# Old version of Command Line Tools' #########################################
##############################################################################

class command_line_arguments:
    def __init__(self,options={'help':'Print this help.',None:['Default argument.','Default_Value','Alternative_Value_1','Alternative_Value_2']},extra_message=None):
        self.extra_message=str(extra_message)
        assert isinstance(options,dict)
        self._options=options
#        self._options_values={option:None for option in self._options.keys()}
        self._options_values={}
        for option in self._options.keys():
            self._options_values[option]=None
        #
        self._descriptions={}
        self._default_values={}
        self._alternative_values={}
        for option,_format in self._options.items():
            if isinstance(_format,list):
                self._descriptions[option]=_format[0]
                self._default_values[option]=_format[1]
                self._alternative_values[option]=_format[2:]
            else:
                self._descriptions[option]=_format
                self._default_values[option]=None
                self._alternative_values[option]=None
        for option,value in self._options_values.items():
            if value is None:
                self._options_values[option]=self._default_values[option]
            elif self._default_values[option] is not None or self._alternative_values[option] is not None:
                if 'ANY' not in self._alternative_values[option]:
                    assert value in self._default_values[option] or value in self._alternative_values[option]
        #
        for arg in sys.argv[1:]:
            if '=' in arg:
                try:
                    option,value=arg.split('=')
                except:
                    print 'ERROR: Invalid argument:',arg
                    self.print_usage(1)
                if option not in self._options.keys():
                    print 'ERROR: Unknown option',option
                    self.print_usage(1)
                else:
                    self._options_values[option]=value
            elif arg in ['help','h','?','-h','--help']:
                self.print_usage(0)
            else:
                self._options_values[None]=arg
        #
#        self._descriptions={}
#        self._default_values={}
#        self._alternative_values={}
#        for option,_format in self._options.items():
#            if isinstance(_format,list):
#                self._descriptions[option]=_format[0]
#                self._default_values[option]=_format[1]
#                self._alternative_values[option]=_format[2:]
#            else:
#                self._descriptions[option]=_format
#                self._default_values[option]=None
#                self._alternative_values[option]=None
#        for option,value in self._options_values.items():
#            if value is None:
#                self._options_values[option]=self._default_values[option]
#            elif self._default_values[option] is not None or self._alternative_values[option] is not None: 
#                if 'ANY' not in self._alternative_values[option]:
#                    assert value in self._default_values[option] or value in self._alternative_values[option]
    def print_usage(self,i):
        print 'Usage:'
        print './[this.py]',
        for option in self._options.keys():
            def_val=self._default_values[option]
            alt_vals=self._alternative_values[option]
            if alt_vals is not None:
                vals='=<'+def_val+'>,'+','.join(alt_vals)
            elif def_val is not None:
                vals='='+def_val
            else:
                vals=''
            print '['+str(option)+vals+']',
        print
        print 'Where:'
        for option in self._options.keys():
            print str(option),':',self._descriptions[option]
        if self.extra_message is not None:
            print self.extra_message
        sys.exit(i)
    def options(self):
        return self._options.keys()
    def assigned_options(self):
        return self._options_values
    def __getitem__(self,x):
        return self._options_values[x]



def fixed_length_numeric_label(n,N=5):
    n=int(n)
    N=int(N)
    assert N>0,"ERROR : N<=0."
    assert 10**N-1>=n,"ERROR: n has more than N digits."
    return '{0}'.format(str(n).zfill(N))

def rand_label(N=5):
    N=int(N)
    return '{0}'.format(str(np.random.randint(10**N)).zfill(N))



class Basename:
    def __init__(self,filename,extension=None,new_path=None):
        self._basemame=None
        self.filename=str(filename)
        if extension is not None:
            self.extension=str(extension)
        else:
            self.extension='.'+self.filename.split('.')[-1]
        if new_path is not None:
            self._new_path=str(new_path)
            if self._new_path[-1]!='/':
                self._new_path+='/'
            self._basename=self._new_path+self.filename.split('/')[-1]
        else:
            try:
                self._basename=self.filename.split('/')[-1]
            except:
                 self._basename=self.filename
     
    def basename(self):
        return self._basename.replace(self.extension,'')
        
    def derivedname(self,description,sample=''):
        """
        sample : "","rand",int()
        """
        description=str(description)
        if sample=='rand':
            sample='_sample'+rand_label()
        elif sample!='':
            sample='_'+sample
        return self._basename.replace(self.extension,'_'+description+sample+self.extension)

if __name__ == '__main__':
    import basescript as bs

    options=[
    bs.Option('filein',values='ANY',description='a file (name).'),
    bs.Option('param1',values='ANY',description='The parameter param1.'),
    bs.Option('optpar1',values=['None','ANY'],description='The optinal parameter optpar1.'),
    ]
    cl=bs.CommandLine(options,
    extra_help_message="""Some extra help message here...
    ... and also down here."""
    )

    filein=cl['filein']
    param1=cl['param1']
    optpar1=cl['optpar1']

    print '# filein',filein
    print '# param1',param1
    print '# optpar1',optpar1
