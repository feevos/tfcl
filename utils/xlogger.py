from datetime import datetime
import os
import pandas as pd 
import ast 
import numpy as np
def read_xlogfile(filename,sep="|",lineterminator='\n'):
    """
    Convenience function to read log files, the user must provide columns that correspond to lists
    """
    df = pd.read_csv(filename,sep=sep,lineterminator=lineterminator)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)))
    return df


class xlogger(object):
    """
    Object to log dictionaries of values. 
    This object constructs files ending in .dat, and is used to log various variables during training. 

    To be used with values that are floats or numpy arrays. 
    Use read_xlogfile to read the outputs of xlogger
    """
    def __init__(self, filename, ending='dat',sep='|',lineterminator='\n'):
        """
        Default separator is "|", works well when storing mixed types, like floats and numpy arrays. 
        """
        self.f = filename
        self.sep = sep
        self.end = lineterminator
        if os.path.exists(self.f):
            print ("Warning, filename::{} exists, renaming to avoid overwriting".format(filename))
            head, tail = os.path.split(self.f)
            ending = tail.split('.')[-1] 

            timenow = datetime.now().strftime("%d-%m-%Y::%Hh-%Mm-%S")
            tail = tail.replace(ending,'_copy_on_{}.dat'.format(timenow))
            self.f = os.path.join(head,tail)
            print ("Logging in filename:{}".format(self.f))

    def write_helper(self,list_of_values, filename, open_mode):
        with open(filename,open_mode) as ff:
            print(*list_of_values, file=ff,flush=True,sep=self.sep,end=self.end)


    def write_header(self,kward):
        tlist = list(kward.keys())
        self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')
            
    def write(self,kward):
        # Trick to store numpy arrays into dictionary
        for k,v in kward.items():
            if isinstance(v,np.ndarray):
                kward[k]=v.tolist()
        if os.path.exists(self.f):
            tlist = kward.values()
            self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')
        else:
            self.write_header(kward)
            tlist = kward.values()
            self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')


