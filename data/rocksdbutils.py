import rocksdb 
import numpy as np
# GPT4 recommended solution for pickle5 / pickle imports when used with different containers
import sys
if sys.version_info < (3, 8):
    import pickle5 as mypickler
else:
    import pickle as mypickler

from ssg2.utils.xlogger import *  # Necessary for saving keys 
import os 
import ast # Reads keys
import pandas as pd # Reads keys

# Parallel WriteBatch
from pathos.pools import ThreadPool as pp 
from multiprocessing import Lock

from collections import OrderedDict

class _RocksDBBase(object):
    """
    Base class with useful defaults 
    Creates a database with two families (columns), of inputs and labels 
    """
    def __init__(self, 
                 lru_cache=1,
                 lru_cache_compr=0.5,
                 num_workers = 16,
                 read_only=True):
        super().__init__()

        GB = 1024**3
        self.lru_cache_GB       = lru_cache         * GB
        self.lru_cache_compr_GB = lru_cache_compr   * GB
        self.num_workers        = num_workers
        self.read_only          = read_only
        
    def _get_db_opts_default(self):
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 3e+5        # Some Defaults 
        opts.write_buffer_size = 67108864 # Some Defaults 
        opts.max_write_buffer_number = 30 # 3 default
        opts.target_file_size_base = 67108864  # default 67108864, value starting 7: 7340032 input.nbytes
        #opts.paranoid_checks=False

        # @@@@@@@@@@@ performance boost @@@@@@@@@@
        opts.IncreaseParallelism(self.num_workers)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        opts.table_factory = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10), # 10 default for 1% error 
                block_cache=rocksdb.LRUCache(self.lru_cache_GB), # 16GB
                block_cache_compressed=rocksdb.LRUCache(self.lru_cache_compr_GB )) # 0.5 GB 

        return opts

    def _get_db_colfamopts_default(self):
        opts = rocksdb.ColumnFamilyOptions()
        opts.write_buffer_size = 67108864 # Some Defaults 
        opts.max_write_buffer_number = 30 # 3 default
        opts.target_file_size_base = 67108864  # default 67108864, value starting 7: 7340032 input.nbytes
        #opts.paranoid_checks=False


        opts.table_factory = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10), # 10 default for 1% error 
                block_cache=rocksdb.LRUCache( self.lru_cache_GB ), # 16GB
                block_cache_compressed=rocksdb.LRUCache( self.lru_cache_compr_GB)) # 0.5 GB 

        return opts
        

    def _open_rocks_write(self,flname_db):
        # Remark: it is fastest if I bundle together inputs and labels, though not generic

        self.opts_db = self._get_db_opts_default()
        self.db = rocksdb.DB(flname_db, self.opts_db, read_only=self.read_only)


        self.cf_opts    = OrderedDict()
        self.cf_objects = OrderedDict()
        for family_name in self.cf_names:
            self.cf_opts[family_name]  = self._get_db_colfamopts_default()
            self.cf_objects[family_name] = self.db.create_column_family(family_name,self.cf_opts[family_name])





    def _open_rocks_read(self,flname_db):
        # Remark: it is fastest if I bundle together inputs and labels, though not generic

        self.opts_db = self._get_db_opts_default()

        self.cf_opts    = OrderedDict()
        for family_name in self.cf_names:
            self.cf_opts[family_name]  = self._get_db_colfamopts_default() # 


        self.db = rocksdb.DB(flname_db, self.opts_db, column_families={b'default':self.opts_db, **self.cf_opts},read_only=self.read_only)






class RocksDBWriter(_RocksDBBase):
    """
    flname_db: Filename for where the database will be written. 
    metadata: anything you want to save to the database (under key: 'meta') along with the data 

    It should contain at a minimum the following keys: inputs_shape, inputs_dtype, labels_shape, labels_dtype. E.g. 
    
    metadata={'inputs':{
                'inputs_shape':(14,256,256),
                'inputs_dtype':np.float32},

                'labels':{'labels_shape':(NClasses,256,256), # Use None for Integer class labels 
                'labels_dtype':np.uint8}
                }
    """
    def __init__(self, 
                 flname_db, 
                 metadata, # Dict of dicts (ordered). 
                 lru_cache=1,
                 lru_cache_compr=0.1,
                 num_workers = 8,
                 read_only=False):
        super().__init__(lru_cache,lru_cache_compr,num_workers,read_only)
        
        # Define Column families 
        self.cf_names = [str.encode(key) for key in metadata.keys()]
    
        #Open Database
        self._open_rocks_write(flname_db)


        # Write meta dictionary in file
        with open(os.path.join(flname_db,'metadata.dat'), 'wb') as handle:
            mypickler.dump(metadata, handle, protocol=mypickler.HIGHEST_PROTOCOL) # Works only for mypickler == pickle  


        # Write in database as well, legacy operation 
        meta_key = b'meta'
        meta_dumps = mypickler.dumps(metadata)
        self.db.put(meta_key,meta_dumps)

        
        # Initialize ascii file that has all keys 
        # Writes all keys EXCEPT b'meta' 
        self.keys_logger = xlogger(os.path.join(flname_db,'keys.dat'))




        self.global_idx = 0
        self.lock = Lock()

    def write_batch(self,batch):
        # batch: iterable of tuples of numpy arrays 
        wb = rocksdb.WriteBatch()

        # Parallel writing of batch of data 
        def writebatch(global_idx,datum):
            key_input = '{}'.format(global_idx).encode('ascii')

            for cfname, tinput in zip(self.cf_names, datum):
                cfinputs = self.db.get_column_family(cfname)
                wb.put((cfinputs ,key_input),  tinput.tobytes())
               
            # Write keys into file, for fast accessing them 
            self.lock.acquire()
            self.keys_logger.write({'keys':key_input})
            self.lock.release()


        pool = pp(nodes=self.num_workers) # with 16 works nice
        global_indices = [self.global_idx + i for i in range(len(batch))]
        
        
        result = pool.map(writebatch,global_indices,batch)
        # Write batch 2 DB
        self.db.write(wb)
        # Update lglobal index 
        self.global_idx = global_indices[-1]+1



class RocksDBReader(_RocksDBBase):
    def __init__(self, 
                 flname_db, 
                 lru_cache=1,
                 lru_cache_compr=0.5,
                 num_workers = 4,
                 read_only=True):
        super().__init__(lru_cache,lru_cache_compr,num_workers,read_only)



        # Read meta dictionary in file
        path = os.path.join(flname_db,'metadata.dat')
        if os.path.exists(path):
            with open( path, 'rb') as handle:
                self.meta = mypickler.load(handle) # Works only for mypickler == pickle  
            # Define Column families 
            self.cf_names = [str.encode(key) for key in self.meta.keys()]
            self._open_rocks_read(flname_db)
        else: # Legacy 
            #Open Database
            self._open_rocks_read(flname_db)
            meta = self.db.get(b'meta')
            self.meta = mypickler.loads(meta) 
            self.cf_names = {b'inputs',b'labels'}

        # Read all keys 
        self.keys = self._read_keys(flname_db)


    def _read_keys(self,flname_db, flname_keys = 'keys.dat',sep="|",lineterminator='\n'):
        # This function works in conjuction with the RocksDBDatasetWriter class, and reads defaults 
        flname_keys = os.path.join( flname_db, flname_keys) 
        df = pd.read_csv(flname_keys ,sep=sep,lineterminator=lineterminator)
        df['keys'] = df['keys'].apply(lambda x: ast.literal_eval(x))
        return df['keys'].tolist()


    def get_inputs_labels(self,idx):

        key = self.keys[idx]


        all_inputs = []
        for cname in self.cf_names:
            tcfinputs = self.db.get_column_family(cname)
            cname = bytes.decode(cname)
            tinputs = self.db.get( (tcfinputs, key) )
            tshape =  self.meta[cname]['{}_shape'.format(cname)]
            if tshape is not None:
                tinputs = np.frombuffer(tinputs, dtype= self.meta[cname]['{}_dtype'.format(cname)]).reshape(*tshape)
            else:
                tinputs = np.frombuffer(tinputs, dtype= self.meta[cname]['{}_dtype'.format(cname)])
            #all_inputs.append(tinputs.copy()) # The np.frombuffer results in a readonly array, that forces pytorch to create warnings. This is usually taken care of in transform method during training
            all_inputs.append(tinputs) # 


        return all_inputs



### Convenience class that writes data into database
from time import time
from datetime import timedelta
from .rasteriter import RasterMaskIterableInMemory 
from ssg2.utils.progressbar import progressbar
class Rasters2RocksDB(object):
    def __init__(self, 
                 lstOfTuplesNames, 
                 names2raster_function, 
                 metadata, 
                 flname_prefix_save, 
                 transform=None, 
                 # Some useful defaults for Remote Sensing (large) imagery
                 batch_size=64,  
                 Filter=256, 
                 stride_divisor=2,
                 train_split=0.9,
                 split_type='sequential'):
        super().__init__()
        
        self.listOfTuplesNames = lstOfTuplesNames
        self.names2raster = names2raster_function
        flname_db_train = os.path.join(flname_prefix_save,'train.db') 
        flname_db_valid = os.path.join(flname_prefix_save,'valid.db')

        self.Filter = Filter
        self.stride_divisor = stride_divisor

        self.dbwriter_train = RocksDBWriter(flname_db_train,metadata)
        self.dbwriter_valid = RocksDBWriter(flname_db_valid,metadata)

        
        self.transform   = transform
        self.batch_size  = batch_size
        self.train_split = train_split
        split_types = ['sequential','random']
        assert split_type in split_types, ValueError("Cannot understand split_type, available options::{}, aborting ...".format(split_types))
        self.split_type = split_type

        print ("Creating databases in location:{}".format(flname_prefix_save))
        print('Database train::{}'.format(flname_db_train))
        print('Database valid::{}'.format(flname_db_valid))


    def write_split_strategy(self,batch_idx, NTrain_Total, some_batch):
        if self.split_type == 'random':
            if np.random.rand() < self.train_split:
                self.dbwriter_train.write_batch(some_batch)
            else:
                self.dbwriter_valid.write_batch(some_batch)
        elif self.split_type == 'sequential':
                if batch_idx < NTrain_Total:
                    self.dbwriter_train.write_batch(some_batch)
                else:
                    self.dbwriter_valid.write_batch(some_batch)


        
    def create_dataset(self):
        # For all triples in list of filenames                      
        tic = time()                                                
        for idx,names in enumerate(self.listOfTuplesNames):            
            print ("============================")                  
            print ("----------------------------")                  
            print ("Processing:: {}/{} Tuple".format(idx+1, len(self.listOfTuplesNames)))
            print ("----------------------------")                  
            for name in names:                                      
                print("Processing File:{}".format(name))            
            print ("****************************")
            
            lst_of_rasters = self.names2raster(names)
            myiterset = RasterMaskIterableInMemory(lst_of_rasters,
                                                   Filter = self.Filter,
                                                   stride_divisor=self.stride_divisor ,
                                                   transform=self.transform,
                                                   batch_size=self.batch_size,
                                                   batch_dimension=False)

            nbset = myiterset.get_len_batch_set()
            train_split = int(self.train_split*nbset)
            for idx2 in progressbar(range(nbset)):
                batch = myiterset.get_batch(idx2)
                self.write_split_strategy(idx2,train_split,batch)

                    
        Dt = time() - tic                                           
        Dt = str(timedelta(seconds=Dt))
        NData = len(self.listOfTuplesNames)
        print("time to WRITE N::{} files, Dt::{}".format(NData,Dt)) 
        
            
        print (" XXXXXXXXXXXXXXXXXXXXXXX Done! XXXXXXXXXXXXXXXXXXXXXX")
                




#### TORCH Dataset Class 
import torch
class RocksDBDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 flname_db,
                 transform=None,
                 lru_cache=0.01,
                 lru_cache_compr=0.01,
                 num_workers = 4,
                 read_only=True):
        super().__init__()

        self.mydbreader = RocksDBReader(flname_db,lru_cache,lru_cache_compr,num_workers,read_only)
        self.length = len(self.mydbreader.keys)
        self.transform = transform
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        data = self.mydbreader.get_inputs_labels(idx)
        
        if self.transform is not None:
            data = self.transform(*data)
            return data

        return data




