#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import print_function

__authors__ = [
    "laurent.faucheux@hotmail.fr",
]

__version__ = '0.1.63'

__all__     = [
    '__authors__',
    '__version__',
    'UniHasher',
    'Cache',
    'Serializer',
    'FileError',
    'DataGetter',
    'DataObject',
    'SpDataObject',
    'GaussianMLARIMA',
    'XACFMaker',
    'Sampler',
    'Metrician',
    'PIntervaler',
    'Presenter',
    'npopts',
    'pdopts',
]

import warnings ; warnings.filterwarnings("ignore")
import matplotlib.patches as mpa
import matplotlib.pyplot as plt
import numdifftools as nd
import collections as cl
import posixpath as px
import functools as ft
import itertools as it
import pickle as ckl
import hashlib as hh
import random as rd
import pandas as pd
import scipy as sc
import pysal as ps
import numpy as np
import errno
import copy
import sys
import os
import re


npopts = lambda p,b,w,f=None:np.set_printoptions(
    precision=p, suppress=b, linewidth=w, formatter=f
)
pdopts = lambda w:pd.set_option(
    'display.width', int(w)
)

name_eq_main = __name__ == '__main__'
#*****************************************************************************#
##    ╦ ╦┌┐┌┬╦ ╦┌─┐┌─┐┬ ┬┌─┐┬─┐
##    ║ ║││││╠═╣├─┤└─┐├─┤├┤ ├┬┘
##    ╚═╝┘└┘┴╩ ╩┴ ┴└─┘┴ ┴└─┘┴└─
_hash_ = lambda o:hh.md5(o).digest().encode("base64")
class UniHasher(object):
    @staticmethod
    def hash_(o):
        r""" Returns a hash of any object.

        Example
        -------
        >>> slib = UniHasher
        
        >>> array = np.arange(10)
        >>> slib.hash_(array)
        'gaTYf5Qzp0xE5BhkFjw2yw==\n'

        >>> dict_of_arrays = {'a1':np.arange(5), 'a2':np.arange(10)}
        >>> slib.hash_(dict_of_arrays)
        'rg0anwYx9IyZNLGvhPmclg==\n'

        >>> list_of_arrays = [np.arange(5), np.arange(10)]
        >>> slib.hash_(list_of_arrays)
        '/LIv0U1I34TJ92FAdrvo/A==\n'

        >>> blob = {
        ...     'str'       : 'abcdef',
        ...     'str_list'  : list('abcdef'),
        ...     'int'       : 1,
        ...     'bool'      : True,
        ...     'none'      : None,
        ...     'int_tuple' : (1,),
        ...     'float_list': [.123, 123.],
        ...     'arr_dict'  : {'a1':np.arange(5), 'a2':np.arange(10)},
        ...     'bool_list' : [True, False, False],
        ...     'none_list' : [None, None, 'None'],
        ...     'blob_list' : [None, False, 123., np.arange(10), [1], (1,), {1:1}],
        ... }
        >>> slib.hash_(blob)
        'X/JnOxedGsKfh1LofptcWA==\n'

        """
        return {
            dict       : lambda d: _hash_(''.join(map(UniHasher.hash_, sorted(d.items())))),
            tuple      : lambda t: _hash_(''.join(map(UniHasher.hash_, t))),
            list       : lambda l: _hash_(''.join(map(UniHasher.hash_, l))),
            type(None) : lambda n: _hash_(str(n)),
            bool       : lambda b: _hash_(str(b)),
            int        : lambda i: _hash_(str(i)),
            float      : lambda f: _hash_(str(f)),
            np.ndarray : UniHasher._array_hash,
            str        : _hash_,
            buffer     : _hash_,
        }[type(o)](o)

    @staticmethod
    def _array_hash(array):
        array.flags.writeable = False
        h = _hash_(array.data)
        array.flags.writeable = True
        return h

    @staticmethod
    def _args_hash(args, kws):
        hh = ''
        for a in args:
            h = UniHasher.hash_(a)
            hh += h
        for kw, a in sorted(kws.items()):
            hh += UniHasher.hash_(a)
        return _hash_(hh)


#*****************************************************************************#
##    ┌─┐┌─┐┌─┐┬ ┬┌─┐
##    │  ├─┤│  ├─┤├┤ 
##    └─┘┴ ┴└─┘┴ ┴└─┘
    
class Cache(UniHasher):
    def __init__(self):
        self._cache = {}
        self._tempo = {}

    @classmethod
    def _dep_property(cls, cid, c='_cache'):
        """ Memoizes outcomes of the so-decorated method, using
        its name plus a suffix which consists of the outcome of
        the attribute whose name is specified via `cid`.
        
        Example
        -------
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {
        ...         'suffix': ' plus an id',
        ...         'attr'  : Cache._dep_property('suffix')(
        ...             meth = lambda cls: rd.random(),
        ...         )
        ...     }, 
        ... )
        >>> o = class_()
        >>> o.attr == o.attr
        True
        >>> o._cache.keys()
        ['<lambda> plus an id']
        """
        def _property_(meth):
            @property
            @ft.wraps(meth)
            def __property(cls):
                mn = '{}{}'.format(
                    meth.__name__,
                    getattr(cls, cid)
                )
                _c = getattr(cls, c)
                if mn not in _c:
                    _c[mn] = meth(cls)
                return _c[mn]
            return __property
        return _property_

    @classmethod
    def __memoize_as_prop(cls, meth, attr):
        """ Memoizes outcomes of the so-decorated method, using
        its name as dict-key identifier.        
        
        Example
        -------
        >>> C = Cache
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {
        ...         'attr':C._Cache__memoize_as_prop(
        ...             meth = lambda cls: rd.random(),
        ...             attr = '_cache'
        ...         )
        ...     }, 
        ... )
        >>> o = class_()
        >>> o.attr == o.attr
        True
        """
        @property
        @ft.wraps(meth)
        def __property(cls):
            mn = meth.__name__
            _c = getattr(cls, attr)
            if mn not in _c:
                _c[mn] = meth(cls)
            return _c[mn]
        return __property
        
    @classmethod
    def _property(cls, meth):
        """ Memoizes outcomes of the so-decorated method.
        NB: is a pre-set wrapper of `__memoize_as_prop`.        
        
        Example
        -------
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {'attr':Cache._property(lambda cls: rd.random())}, 
        ... )
        >>> o = class_()
        >>> o.attr == o.attr
        True
        """
        return cls.__memoize_as_prop(meth, '_cache')

    @classmethod
    def _property_tmp(cls, meth):
        """ Memoizes outcomes of the so-decorated method.
        The only difference with the class method named
        _property is that memoization is intended to be
        temporary.
        NB: is a pre-set wrapper of `__memoize_as_prop`.
        
        Chk
        ---
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {'attr':Cache._property_tmp(lambda cls: rd.random())}, 
        ... )
        >>> o = class_()
        >>> o.attr == o.attr
        True
        """
        return cls.__memoize_as_prop(meth, '_tempo')

    @classmethod
    def _method(cls, meth):
        """ Memoizes outcomes of the so-decorated method, using as
        dict-key identifier its hashed arguments or `cid` if the
        latter is present among arguments.   
        
        Example
        -------
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {
        ...         'meth':Cache._method(
        ...             meth = lambda cls,to_be_hashed: rd.random()
        ...         )
        ...     }, 
        ... )
        >>> o = class_()
        >>> o.meth('to be hashed') == o.meth('to be hashed')
        True
        """
        @ft.wraps(meth)
        def __method(cls, *args, **kws):
            mn = '{}_{}'.format(
                meth.__name__,
                kws.get('cid') or cls._args_hash(args, kws)
            )
            if mn not in cls._cache:
                cls._cache[mn] = meth(cls, *args, **kws)
            return cls._cache[mn]
        return __method

#*****************************************************************************#
##    ╔═╗┌─┐┬─┐┬┌─┐┬  ┬┌─┐┌─┐┬─┐
##    ╚═╗├┤ ├┬┘│├─┤│  │┌─┘├┤ ├┬┘
##    ╚═╝└─┘┴└─┴┴ ┴┴─┘┴└─┘└─┘┴└─
class Serializer(Cache):
    """
    Class which type-aggregates a bunch of methodes used
    to dump/load objects into their instance-like format.
    """
	
    def __init__(self, **kws):
        super(Serializer, self).__init__()
        self._modes = {
            'hickle':{'r':'r', 'w':'w'}, 
            'pickle':{'r':'rb', 'w':'wb'}, 
        }[ckl.__name__]
        self._ext  = kws.get('ext', 'ckl')
        self._sdir = kws.get('sdir')
        self._upd  = kws.get('update') or False
        self._to_upd = kws.get('to_update', [])

    @Cache._property
    def _ckle_save_dir(self):
        sdir = self._sdir or './__serialized/'
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        return sdir

    def _ckle_fname_getter(self, _key_):
        return os.path.join(
            self._ckle_save_dir,
            '{}.{}'.format(_key_, self._ext)
        )
    def _load_ckle(self, _key_):
        fname = self._ckle_fname_getter(_key_)
        with open(fname, self._modes['r']) as ckled:
            return ckl.load(ckled)

    def _dump_ckle(self, _key_, _value_):
        fname = self._ckle_fname_getter(_key_)
        with open(fname, self._modes['w']) as ckled:
            ckl.dump(_value_, ckled)

    def _ckle_exist(self, _key_):
        fname = self._ckle_fname_getter(_key_)
        return 0 if self._upd else os.path.isfile(fname)

    def _may_do(self, _key_, lambda_):
        return self._load_ckle(_key_)\
               if (self._ckle_exist(_key_) and _key_ not in self._to_upd)\
               else lambda_()

    def _may_dump(self, _key_, _value_):
        if not self._ckle_exist(_key_):
            self._dump_ckle(_key_, _value_)

    def _may_do_and_dump(self, _key_, lambda_, _cache):
        _cache[_key_] = self._may_do(_key_, lambda_)
        self._may_dump(_key_, _cache[_key_])

#*****************************************************************************#
##    ╔═╗┬┬  ┌─┐╔═╗┬─┐┬─┐┌─┐┬─┐
##    ╠╣ ││  ├┤ ║╣ ├┬┘├┬┘│ │├┬┘
##    ╚  ┴┴─┘└─┘╚═╝┴└─┴└─└─┘┴└─
class FileError(object):
    def __init__(self, **kws): self.ext = kws.get('ext')

    def _file_not_found(self, name):
        message = ['',
            ' {}.%s does not exist in the working directory.'%self.ext,
            ' Either the file has been renamed or removed.',
            ' This may concern other files as well.'
        ]
        raise IOError(errno.ENOENT, '\n\t'.join(message).format(name))
    def _file_not_unique(self, dirs, **kws):
        enum_dirs = map(lambda(i,d):'{0} - {1}'.format(i+1,d), enumerate(dirs))
        message = ['',
            '{name}.%s is not unique.'%self.ext,
            '{len_} files found.',
            '\n\t\t'.join(['']+enum_dirs)
        ]
        print(IOError('%s2'%self.ext, '\n\t'.join(message).format(**kws)))
        ix = raw_input('\tType the index of the one your want to work with:')
        return int(ix) - 1
    def _unknown_error(self, **kws):        
        message = ['',
            " Something unexpected happened with {name}.%s"%self.ext,
            ' "{exc}"',         
        ]
        raise IOError('%s3'%self.ext, '\n\t'.join(message).format(**kws))

#*****************************************************************************#
##    ╔╦╗┌─┐┌┬┐┌─┐╔═╗┌─┐┌┬┐┌┬┐┌─┐┬─┐
##     ║║├─┤ │ ├─┤║ ╦├┤  │  │ ├┤ ├┬┘
##    ═╩╝┴ ┴ ┴ ┴ ┴╚═╝└─┘ ┴  ┴ └─┘┴└─
class DataGetter(Cache):
    def __init__(self):
        super(DataGetter, self).__init__()
        self._ctwk_dir = os.getcwd()
        self._base_dir = os.path.dirname(__file__)
        self._int_dirs = [
            self._ctwk_dir,
            self._base_dir,
        ]

    _path_stdzer = lambda s,d:px.join(*d.split('\\')).lower()
    def _from_id_to_index(self, dirs, id_):
        r"""
        Method which makes the correspondancy between non-unique file names
        and their associated directory, where the latter is used as a way to
        distinguish them. NB : recall that indexing is 0-based in python.

        Example
        -------
        >>> dg   = DataGetter()
        >>> dirs = sorted(dg._map_all('txt')['non-unique-file'])
        >>> dirs
        ['data\\tests\\non-unique-file.txt', 'data\\tests\\subfolder\\non-unique-file.txt']
        >>> dg._from_id_to_index(dirs, id_='subfolder')\
        ... == dirs.index('data\\tests\\subfolder\\non-unique-file.txt')
        True
        >>> dg._from_id_to_index(dirs, id_=r'sts\no')\
        ... == dirs.index('data\\tests\\non-unique-file.txt')
        True
        """
        if id_ is None:
            return 0
        id_ = self._path_stdzer(id_)
        return dirs.index(
            filter(
                lambda d:id_ in self._path_stdzer(os.path.abspath(d)), dirs
            )[0]
        )

    def _get_dir_of(self, name, ext, **kws):
        """ Methode which gets the targeted file by its name
        and extension.    
        """
        dirs = getattr(self, '_{}_files'.format(ext)).get(name, [])
        len_ = len(dirs)
        file_ix = self._from_id_to_index(dirs, kws.get('file_id'))
        if not len_:
            FileError(ext=ext)._file_not_found(name)            
        elif len_ > 1 and kws.get('file_id') is None:
            file_ix = FileError(ext=ext)._file_not_unique(
                name=name, len_=len_,dirs=dirs
            )
        return os.path._getfullpathname(dirs[file_ix])

    @staticmethod
    def _try_relpath(p):
        """ Returns a relative path if possible, i.e. when the file is
        on the same drive.
        """
        try:
            return os.path.relpath(p)
        except:
            return os.path.abspath(p)
            
    def _map_all(self, ext):
        """ Creates a dict-collection of files
        conditionally upon `ext`.
        """
        dict_ = {}
        len_  = len(ext)+1
        paths = it.chain(
            *map(os.walk, self._int_dirs)
        )
        for dirpath, subdirs, files in paths:
            for file_ in files:
                if file_.endswith(".%s"%ext):
                    key = file_[:-len_]
                    key_= key.lower()
                    if key_ not in dict_:
                        dict_[key_] = dict_['%s.%s'%(key_, ext)] = []
                    rpath = self._try_relpath(
                        os.path.join(dirpath, file_)
                    )
                    if rpath not in dict_[key_]:
                        dict_[key_].append(rpath)
        return dict_

    @Cache._property
    def _dbf_files(self):
        ur""" Dict-aggregates all dbf/path pairs present in
        the current working directory. 

        Example
        -------
        >>> dm = DataGetter()
        >>> sorted(dm._map_all('dbf').items())[0]
        ('columbus', ['data\\COLUMBUS\\columbus.dbf'])
        """
        return self._map_all('dbf')    
    def _get_dbf_dir(self, name, **kws):
        return self._get_dir_of(
            name, 'dbf', **kws
        )

    @Cache._property
    def _shp_files(self):
        ur""" Dict-aggregates all shp/path pairs present in
        the current working directory. 

        Example
        -------
        >>> dm = DataGetter()
        >>> sorted(dm._map_all('shp').items())[0]
        ('columbus', ['data\\COLUMBUS\\columbus.shp'])
        """
        return self._map_all('shp')    
    def _get_shp_dir(self, name, **kws):
        return self._get_dir_of(
            name, 'shp', **kws
        )
        

#*****************************************************************************#
##    ╔╦╗┌─┐┌┬┐┌─┐╔═╗┌┐  ┬┌─┐┌─┐┌┬┐
##     ║║├─┤ │ ├─┤║ ║├┴┐ │├┤ │   │ 
##    ═╩╝┴ ┴ ┴ ┴ ┴╚═╝└─┘└┘└─┘└─┘ ┴ 
class DataObject(DataGetter):

    def __init__(self, data_name, y_name, x_names, **kws):
        super(DataObject, self).__init__()

        if os.path.isfile(data_name):
            self._int_dirs.append(os.path.dirname(data_name))
            data_name  = os.path.basename(data_name)
        self.data_name = os.path.splitext(data_name)[0].lower()
        self.dbf_fname = self._get_dbf_dir(self.data_name)
        self.shp_fname = self._get_shp_dir(self.data_name)
        self._data_dir = os.path.dirname(self.shp_fname)
        
        self.y_name   = y_name        
        self.x_names  = x_names
        self.id_name  = kws.get('id_name')
        self.srid     = kws.get('srid')

    @Cache._property
    def save_dir(self):
        """ Directory used for saving results-related objects,
        for all types, i.e., png, shp, csv, etc..."""
        sdir = os.path.join(
            self._ctwk_dir, 'data', '{}.out'.format(
                self.data_name.capitalize()
            )
        )
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        return sdir

    @Cache._property
    def _szd_dir(self):
        """ Directory used for saving all serialized objects generated.
        """
        elig_sdirs = [
            os.path.join(self.save_dir, '.szd'),
            os.path.join(
                self._base_dir, 'data', '{}.out'.format(
                    self.data_name.capitalize()
                )
            )
        ]
        if os.path.dirname(elig_sdirs[1]) == os.path.dirname(self._data_dir):
            return os.path.join(elig_sdirs[1], '.szd')
        return elig_sdirs[0]

    @Cache._property
    def _szer(self):
        """ Returns a configured instance of Serializer.
        """
        return Serializer(sdir=self._szd_dir)

    @Cache._property
    def dbf_obj(self):
        """Data rendered from a PySAL-opened dbf-file."""
        return ps.open(self.dbf_fname, 'r')

    _shaping_getter = lambda s,n: np.array(s.dbf_obj.by_col(n))[:, None]
    @Cache._property
    def y(self):
        """ Explained/dependent/regressand variable"""
        return self._shaping_getter(self.y_name)
    
    @Cache._property
    def ones(self):
        """ Instrumental vector of ones defined once for all. """
        return np.ones_like(self.y)

    @Cache._property
    def x(self):
        """ Horizontally stacked explanatory variables."""
        if not self.x_names:
            raise ValueError(
                "`x_names` must contain at "
                "least one valid column name"
            )
        return np.hstack(
            [self.ones]
            + map(
                self._shaping_getter,
                self.x_names
            )
        )

    @Cache._property
    def n(self):
        """ Number of individuals."""
        return self.y.shape[0]

    @Cache._property
    def _up2n_line(self):
        """ Multi-purpose auxiliary variable."""
        return np.arange(self.n)

    @Cache._property
    def points_array(self):
        """ Data-object rendered from a PySAL-opened shp-file source."""
        return ps.weights.util.get_points_array_from_shapefile(
            self.shp_fname
        )

    @Cache._property
    def geoids(self):
        """ Names of spatial units."""
        if self.id_name is None:
            return self._up2n_line   
        return ps.weights.util.get_ids(self.shp_fname, self.id_name)
    
    @Cache._property
    def k(self):
        """ Number of predictors."""
        return len(self.x_names) + 1

#*****************************************************************************#
##    ╔═╗┌─┐╔╦╗┌─┐┌┬┐┌─┐╔═╗┌┐  ┬┌─┐┌─┐┌┬┐
##    ╚═╗├─┘ ║║├─┤ │ ├─┤║ ║├┴┐ │├┤ │   │ 
##    ╚═╝┴  ═╩╝┴ ┴ ┴ ┴ ┴╚═╝└─┘└┘└─┘└─┘ ┴
class SpDataObject(DataObject):
        
    def __init__(self, **kws):
        super(SpDataObject, self).__init__(**kws)
        self._ER_ks = kws.get('ER_ks', [])  
        self._AR_ks = kws.get('AR_ks', []) 
        self._MA_ks = kws.get('MA_ks', [])

    def from_scratch(self, **kws):
        """ Cleans cache from all objects that are not permanent
        relatively to a given dataset. One consequence of this is that
        re-executing any instance's methode from scrath returns ols
        like results.
        """
        self._tempo.clear()
        self._ER_ks = kws.get('ER_ks', [])  
        self._AR_ks = kws.get('AR_ks', []) 
        self._MA_ks = kws.get('MA_ks', [])
        

    @staticmethod
    def check_data(data):
        if issubclass(type(data), sc.spatial.KDTree):
            return data
        else:
            try:
                data = np.asarray(data)
                if data.dtype.kind != 'f':
                    data = data.astype(float)
                return sc.spatial.KDTree(data)
            except Exception as err:
                raise ValueError(
                    "{} : Could not make array from data".format(err.message)
                )    
    @property
    def kdtree(self):
        return self.check_data(self.points_array)

    @staticmethod
    def dmat_computer(kdtree, distance_band=float('inf')):
        """ Returns an euclidean-distance matrix.

        Example
        -------
        >>> np.random.seed(0)
        >>> n = 4
        >>> points_array = np.random.random((n, 2))
        >>> points_array
        array([[0.5488135 , 0.71518937],
               [0.60276338, 0.54488318],
               [0.4236548 , 0.64589411],
               [0.43758721, 0.891773  ]])
        >>> slib = SpDataObject
        >>> slib.dmat_computer(
        ...     slib.check_data(points_array)
        ... )
        array([[0.        , 0.1786471 , 0.14306129, 0.20869372],
               [0.1786471 , 0.        , 0.20562852, 0.3842079 ],
               [0.14306129, 0.20562852, 0.        , 0.2462733 ],
               [0.20869372, 0.3842079 , 0.2462733 , 0.        ]])
        """
        return np.asarray(kdtree.sparse_distance_matrix(
            kdtree, max_distance=distance_band
        ).todense())

    @staticmethod
    def smat_computer(dmat):
        """ Returns a transposed matrix of column-based sorting-indices.
         NB: the matrix is transposed since repetitively used as such.

        Example
        -------
        >>> np.random.seed(0)
        >>> n = 4
        >>> points_array = np.random.random((n, 2))
        >>> slib = SpDataObject
        >>> dmat = slib.dmat_computer(
        ...     slib.check_data(points_array)
        ... )
        >>> smat = slib.smat_computer(dmat)
        >>> smat
        array([[0, 1, 2, 3],
               [2, 0, 0, 0],
               [1, 2, 1, 2],
               [3, 3, 3, 1]])
        """
        return dmat.argsort(axis=1, kind="quicksort").T

    @property
    def full_dmat(self):
        """ Full distance matrix.
        """
        _key_ = '{data_name}_DMat_[full]'.format(**self.__dict__)
        if _key_ not in self._cache:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self.dmat_computer(self.kdtree),
                self._cache
            )
##            np.savetxt(
##                os.path.join(self.save_dir, _key_+'.csv'),
##                self._cache[_key_],
##                delimiter=';'
##            )
        return self._cache[_key_]

    @property
    def full_smat(self):
        """ Full transposed matrix of column-based transitory-sorting-indices.
        """
        _key_ = '{data_name}_SMat_[full]'.format(**self.__dict__)
        if _key_ not in self._cache:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self.smat_computer(
                    self.full_dmat
                ),
                self._cache
            )
##            np.savetxt(
##                os.path.join(self.save_dir, _key_+'.csv'),
##                self._cache[_key_],
##                delimiter=';'
##            )
        return self._cache[_key_]

    @staticmethod
    def psw_from_array(arr, **kws):
        """ Converts numpy.ndarray objects into pysal.weights.weights.W ones.

        Example
        -------
        >>> n   = 3
        >>> arr = np.identity(n)
        >>> psw = SpDataObject.psw_from_array(arr)
        >>> psw.full()[0]
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> arr = np.random.random((n,n))
        >>> psw = SpDataObject.psw_from_array(arr)
        >>> (psw.full()[0] == arr).all()
        True
        """
        n,m = arr.shape
        if n!=m:
            raise ValueError(
                "`arr` must be square"
            )            
        tmp_ids   = kws.get('tmp_ids', np.arange(n))
        neighbors = dict([(i,[]) for i in tmp_ids])
        weights   = dict([(i,[]) for i in tmp_ids])
        arr_issym = np.allclose(arr, arr.T, atol=1e-8)
        sp_dk_arr = sc.sparse.dok_matrix(arr)
        for (i, j), v_ij in sp_dk_arr.items():
            if j not in neighbors[i]:
                weights[i].append(v_ij)
                neighbors[i].append(j)
        if arr_issym:
            for (i, j), v_ij in sp_dk_arr.items():            
                if i not in neighbors[j]:
                    weights[j].append(v_ij)
                    neighbors[j].append(i)
        psw = ps.weights.W(neighbors, weights)
        if 'ids' in kws:
            psw.remap_ids(kws['ids'])
        return psw

    @staticmethod
    def _oknn_computer(smat, k, as_psw, **kws):
        """ Returns a `k`-Nearest-Neighbor-Only (OKNN) matrix.

        Example
        -------       
        >>> np.random.seed(0)
        >>> n = 4
        >>> points_array = np.random.random((n, 2))
        >>> slib = SpDataObject
        >>> dmat = slib.dmat_computer(
        ...     slib.check_data(points_array)
        ... )
        >>> dmat
        array([[0.        , 0.1786471 , 0.14306129, 0.20869372],
               [0.1786471 , 0.        , 0.20562852, 0.3842079 ],
               [0.14306129, 0.20562852, 0.        , 0.2462733 ],
               [0.20869372, 0.3842079 , 0.2462733 , 0.        ]])
        >>> smat = slib.smat_computer(dmat)
        >>> o0nn = slib._oknn_computer(
        ...     smat, 0, as_psw=False,
        ... )
        >>> o0nn
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
        >>> o2nn = slib._oknn_computer(
        ...     smat, 2, as_psw=False,
        ... )
        >>> o2nn
        array([[0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])
        """
        mat = np.zeros_like(smat)
        ixs = kws.get('ixs', np.arange(mat.shape[0]))
        mat[ixs, smat[k,:]] = 1.
        if as_psw:
            return SpDataObject.psw_from_array(
                mat, tmp_ids=ixs, ids=kws.get('ids', ixs)
            )
        return mat

    _w_oknn_rex = 'w_o%dnn_matrix'
    def oknn_computer(self, k, as_psw=True):
        """ Caching/serializing wrapper of self._oknn_computer.
        """
        _key_ = self._w_oknn_rex%k
        if as_psw:
            _key_ += '_as_psw'
        else:
            _key_ += '_as_arr'
            
        if _key_ not in self._cache:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self._oknn_computer(
                    self.full_smat, k, as_psw,
                    ixs=self._up2n_line,
                    ids=self.geoids
                ),
                self._cache
            )
        return self._cache[_key_]

    @staticmethod
    def _w_collector(oknn_lambda, ks_list):
        r"""

        Example
        -------       
        >>> np.random.seed(0)
        >>> n = 4
        >>> points_array = np.random.random((n, 2))
        >>> slib = SpDataObject
        >>> dmat = slib.dmat_computer(
        ...     slib.check_data(points_array)
        ... )
        >>> smat = slib.smat_computer(dmat)
        >>> oknn_lambda = lambda k:slib._oknn_computer(
        ...     smat,
        ...     k,
        ...     as_psw=False,
        ... )
        >>> ks_list = [1, 2]
        >>> _w_collection = slib._w_collector(
        ...     oknn_lambda,
        ...     ks_list
        ... )
        >>> _w_collection[0]
        array([[0, 0, 1, 0],
               [1, 0, 0, 0],
               [1, 0, 0, 0],
               [1, 0, 0, 0]])
        >>> _w_collection[1]
        array([[0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])

        Example 2 
        ---------
        >>> o = SpDataObject(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> ks_list       = [0, 4, 22, 47]
        >>> _w_collection = o._w_collector(
        ...     lambda k: o.oknn_computer(k, as_psw=False),
        ...     ks_list
        ... )

        Chk
        ---
        >>> hash_chk = o.hash_(
        ...     _w_collection
        ... )
        >>> hash_ref = 'bc3JZoBk99nnYkC43dgn0Q==\n'
        >>> _w_collection_as_expected = hash_ref == hash_chk
        >>> _w_collection_as_expected
        True
        """
        return map(oknn_lambda, sorted(ks_list))

    @Cache._method
    def w_collector(self, ks_list):
        r""" Caching wrapper of self._w_collector

        Example
        -------
        >>> o = SpDataObject(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> ks_list      = [0, 4, 22, 47]
        >>> w_collection = o.w_collector(ks_list)        

        Chk
        ---
        >>> hash_chk = o.hash_(
        ...     w_collection
        ... )
        >>> hash_ref = 'bc3JZoBk99nnYkC43dgn0Q==\n'
        >>> w_collection_as_expected = hash_ref == hash_chk
        >>> w_collection_as_expected
        True
        """
        return self._w_collector(
            lambda k:self.oknn_computer(k, as_psw=False),
            ks_list
        )

    @Cache._property_tmp
    def ER_p(self):
        """ Number of differencing spatial predictors."""
        return len(self.ER_ks)
    @Cache._property_tmp
    def ER_ks(self):
        """ List of neighbor orders implied in the modeling of the
        error-autoregressive process.
        """
        return filter(lambda k:k>0, sorted(self._ER_ks))
    @Cache._property_tmp
    def ER_w_collection(self):
        """ List of weight matrices implied in the modeling of the
        error-autoregressive process.
        """
        return self.w_collector(self.ER_ks)
    @Cache._property_tmp
    def ER_p_names(self):
        u""" γ-family """
        return map(lambda k:r'\gamma_{%d}'%k, self.ER_ks)

    @Cache._property_tmp
    def AR_p(self):
        """ Number of autoregressive spatial predictors."""
        return len(self.AR_ks)
    @Cache._property_tmp
    def AR_ks(self):
        """ List of neighbor orders implied in the modeling of the
        regressand-autoregressive process.
        """
        return filter(lambda k:k>0, sorted(self._AR_ks))
    @Cache._property_tmp
    def AR_w_collection(self):
        """ List of weight matrices implied in the modeling of the
        regressand-autoregressive process.
        """
        return self.w_collector(self.AR_ks)
    @Cache._property_tmp
    def AR_p_names(self):
        u""" ρ-family """
        return map(lambda k:r'\rho_{%d}'%k, self.AR_ks)

    @Cache._property_tmp
    def MA_p(self):
        """ Number of moving-average spatial predictors."""
        return len(self.MA_ks)     
    @Cache._property_tmp
    def MA_ks(self):
        """ List of neighbor orders implied in the modeling of the
        moving average process.
        """
        return filter(lambda k:k>0, sorted(self._MA_ks))
    @Cache._property_tmp
    def MA_w_collection(self):
        """ List of weight matrices implied in the modeling of the
        moving average process.
        """
        return self.w_collector(self.MA_ks)
    @Cache._property_tmp
    def MA_p_names(self):
        u""" λ-family """
        return map(lambda k:r'\lambda_{%d}'%k, self.MA_ks)

    
    
    @Cache._property_tmp
    def p(self):
        """ Number of spatial predictors."""
        return self.ER_p + self.AR_p + self.MA_p    
    
    @Cache._property_tmp
    def p_names(self):
        u""" List of spatial parameters' names."""
        return self.ER_p_names + self.AR_p_names + self.MA_p_names
    
    @Cache._property_tmp
    def model_id(self):
        return 'ER{{{}}}AR{{{}}}MA{{{}}}'.format(
            ','.join(map(str, self.ER_ks or [0])),
            ','.join(map(str, self.AR_ks or [0])),
            ','.join(map(str, self.MA_ks or [0])),
        )

#*****************************************************************************#
##    ╔═╗┌─┐┬ ┬┌─┐┌─┐┬┌─┐┌┐┌╔╦╗╦  ╔═╗╦═╗╦╔╦╗╔═╗
##    ║ ╦├─┤│ │└─┐└─┐│├─┤│││║║║║  ╠═╣╠╦╝║║║║╠═╣
##    ╚═╝┴ ┴└─┘└─┘└─┘┴┴ ┴┘└┘╩ ╩╩═╝╩ ╩╩╚═╩╩ ╩╩ ╩    
class GaussianMLARIMA(SpDataObject):

    def __init__(self, **kws):
        """
        Non-inherited parameters
        ------------------------
        verbose : bool, optional
            Displays indicators of processes' progression (True by default).
        opverbose : bool, optional
            Displays minimizer's messages/warnings (True by default).
           
        
        scipy's
        *******
        tolerance : float, optional
            Tolerance for termination. (1.e-7 by default)

        NB : check scipy's documentation for more details or do, e.g.
        >> from scipy.optimize import minimize; help(minimize)
            
        numdifftools's
        **************
        nd_order : int, optional
            Order of differentiation implied in the numerical approximation
            of hessian matrices' components. (3 by default)
        nd_step : float, optional
            Spacing used for differentiation.
        nd_method : str, optional
            Method used for approximation. ('central' by default)

        NB : check numdifftools's documentation for more details or do, e.g.
        >> from numdifftools import Hessian; help(Hessian)
        
        """
        super(GaussianMLARIMA, self).__init__(**kws)
        self._tolerance = kws.get('tolerance', 1e-12 if name_eq_main else 1e-7)
        self._nd_order  = kws.get('nd_order', 3)
        self._nd_step   = kws.get('nd_step', 1.e-6)
        self._nd_method = kws.get('nd_method', 'central')
        self.verbose   = kws.get('verbose', True)
        self.opverbose = kws.get('opverbose', True)
        self._thts_collection = {}
        
    to_save = [
        'par',
        'crt',
##        'cov', #[!!!] Too time-consuming.
##        'stt', #[!!!] stt needs covmat's diagonal.
    ]

    @property
    def default_thts(self):
        return {
            key:{'stack':getattr(self, 'thetas_%s'%key, self.thetas)}
            for key in self.to_save
        }
    _mid2pnames = {}
    def _thts_collector(self, **kws):
        key  = kws.get('key', 'hat')
        thts = kws.get('thts') or self.default_thts
        if key not in self._thts_collection:
            self._thts_collection[key] = {}
        self._thts_collection[key][self.model_id] = thts
        self._mid2pnames[self.model_id] = self.par_names

    @Cache._property_tmp
    def par_names(self):
        return [
            r'\beta_0'
        ] + [
            r'\beta_{%s}'%xn for xn in self.x_names
        ] + self.p_names + [
            r'\sigma^2_{ML}'
        ]

    @Cache._property
    def Idn(self):
        """ Stores (once for all) and returns an identity matrix frequently
        involved during calculations.
        """
        return np.identity(self.n)

    def _tempo_assigner(self, meth_name):
        """ Auxiliary method only defined to stay DRY.
        """
        ysri, xri, G, S, Ri, R = self._yNx_filterer(
            self.llik_maximizing_coeffs_as_list
        )
        lcls     = locals()
        unw_keys = [
            meth_name,
            'meth_name',
            'self',
        ]
        for key in lcls.keys():
            if key not in unw_keys:
                self._tempo[key] = lcls[key]
        return lcls[meth_name]

    @Cache._property_tmp
    def ysri(self):
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def xri(self):
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def G(self):
        ur"""
        \Gamma
        """
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def Gi(self):
        ur"""
        \Gamma^{-1}
        """
        return np.linalg.inv(self.G)
    @Cache._property_tmp
    def S(self):
        ur"""
        P
        """
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def Si(self):
        ur"""
        P^{-1}
        """
        return np.linalg.inv(self.S)
    @Cache._property_tmp
    def Ri(self):
        ur"""
        \Lambda^{-1}
        """
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def R(self):
        ur"""
        \Lambda
        """
        return self._tempo_assigner(
            sys._getframe().f_code.co_name
        )
    @Cache._property_tmp
    def RiSG(self):
        return self.dot(self.Ri, self.S, self.G)
    @Cache._property_tmp
    def jac(self):
        return np.linalg.det(self.RiSG)
    @Cache._property_tmp
    def RiSGi(self):
        return np.linalg.inv(self.RiSG)
    @property
    def ysritysri(self): return np.dot(self.ysri.T, self.ysri)
    @Cache._property_tmp
    def xrit(self): return self.xri.T    
    @Cache._property_tmp
    def xritxri(self): return np.dot(self.xrit, self.xri)
    @Cache._property_tmp
    def xritxrii(self): return np.linalg.inv(self.xritxri)
    @Cache._property_tmp
    def xritxriixrit(self): return np.linalg.solve(self.xritxri, self.xrit)
    
    @Cache._property_tmp
    def leverage(self):
        return np.dot(self.xri, self.xritxriixrit).diagonal()[:, None]
    @Cache._property_tmp
    def levscale(self):
        return np.power(1. - np.power(self.leverage, 2.), -0.5)
    @Cache._property_tmp
    def xritysri(self): return np.dot(self.xrit, self.ysri)    
    @Cache._property_tmp
    def betas(self): 
        # np.linalg.solve(self.xritxri, self.xritysri) [MAY CUMULATE TOO MANY ROUNDING TRUNCATIONS]
        return np.dot(self.xritxriixrit, self.ysri)
    @Cache._property_tmp
    def xribetas(self): return np.dot(self.xri, self.betas)
    @Cache._property_tmp
    def xbetas(self): return np.dot(self.x, self.betas)
    @Cache._property_tmp
    def usri(self): return self.ysri - self.xribetas
    @Cache._property_tmp
    def usritusri(self):
        return np.dot(self.usri.T, self.usri)
    @Cache._property_tmp
    def sig2(self):
        # (self.ysritysri - np.dot(self.xritysri.T, self.betas))/self.n [MAY CUMULATE TOO MANY ROUNDING TRUNCATIONS]
        return self.usritusri/self.n
    @Cache._property_tmp
    def betas_cov(self): return self.sig2*self.xritxrii
    @Cache._property_tmp
    def betas_se(self):return np.sqrt(self.betas_cov.diagonal())[:, None]
    @Cache._property_tmp
    def sig2n_k(self): return (self.sig2 * self.n/(self.n-self.k-self.AR_p))[0][0]
    @Cache._property_tmp
    def XACF_u(self): return self.usri
    @Cache._property_tmp
    def yhat(self):return self.xbetas
    @Cache._property_tmp
    def RiG(self):
        return np.dot(self.Ri, self.G)
    @Cache._property_tmp
    def RiGx(self):
        return self.dot(self.RiG, self.x)
    @Cache._property_tmp
    def yhat_e(self):
        return ps.spreg.utils.spdot(
            np.linalg.inv(self.RiSG),
            self.dot(self.RiG, self.xbetas)
        )
    @Cache._property_tmp
    def e(self):
        """ Prediction error."""
        return self.y - self.yhat_e

    @staticmethod
    def nan_interp(y):
        """ Returns arrays whose nan values have been replaced by
        retro-, inter- or extra-polation.

        Example
        -------
        >>> slib   = GaussianMLARIMA
        >>> nanarr = np.array([np.nan, np.nan, 1., np.nan, 3., np.nan])[:, None]
        >>> arr    = slib.nan_interp(nanarr)
        >>> np.hstack((nanarr, arr))
        array([[nan,  1.],
               [nan,  1.],
               [ 1.,  1.],
               [nan,  2.],
               [ 3.,  3.],
               [nan,  3.]])
        """
        y_       = copy.copy(y)
        nans, i  = np.isnan(y_), lambda n: n.nonzero()[0]
        y_[nans] = np.interp(i(nans), i(~nans), y_[~nans])
        return y_
    @staticmethod
    def hull_sides_computer(r, b, mus=None, O=1, o=0):
        r""" Returns `O`th-order local optima of `r`'s hull 

        Example
        -------
        >>> slib = GaussianMLARIMA
        >>> arr  = np.array([-5, 5, -10, 10, -8, 8, -12, 12])[:, None]
        >>> arr
        array([[ -5],
               [  5],
               [-10],
               [ 10],
               [ -8],
               [  8],
               [-12],
               [ 12]])

        Let's get `r`'s hull defined over each very-local optima.
        >>> ubnd1 = slib.hull_sides_computer(arr, 'up', O=1)
        >>> lbnd1 = slib.hull_sides_computer(arr, 'lo', O=1)
        >>> np.hstack((lbnd1, arr, ubnd1))
        array([[ -5. ,  -5. ,   5. ],
               [ -7.5,   5. ,   5. ],
               [-10. , -10. ,   7.5],
               [ -9. ,  10. ,  10. ],
               [ -8. ,  -8. ,   9. ],
               [-10. ,   8. ,   8. ],
               [-12. , -12. ,  10. ],
               [-12. ,  12. ,  12. ]])

        Let's do the same but focusing on the 2nd least local optima.
        >>> ubnd2 = slib.hull_sides_computer(arr, 'up', O=2)
        >>> lbnd2 = slib.hull_sides_computer(arr, 'lo', O=2)
        >>> np.hstack((lbnd2, arr, ubnd2))
        array([[-10. ,  -5. ,  10. ],
               [-10. ,   5. ,  10. ],
               [-10. , -10. ,  10. ],
               [-10. ,  10. ,  10. ],
               [-10. ,  -8. ,   9. ],
               [-10. ,   8. ,   9.5],
               [-12. , -12. ,  10. ],
               [-12. ,  12. ,  12. ]])

        And finally, focusing on global optima.
        >>> ubnd3 = slib.hull_sides_computer(arr, 'up', O=3)
        >>> lbnd3 = slib.hull_sides_computer(arr, 'lo', O=3)
        >>> np.hstack((lbnd3, arr, ubnd3))
        array([[-12.,  -5.,  12.],
               [-12.,   5.,  12.],
               [-12., -10.,  12.],
               [-12.,  10.,  12.],
               [-12.,  -8.,  12.],
               [-12.,   8.,  12.],
               [-12., -12.,  12.],
               [-12.,  12.,  12.]])
        """
        if mus is None:
            mus = (np.mean(r),)
        if o<O:
            o += 1
            if b == 'up':
                sp_posr       = np.full(r.shape, np.nan)
                pcnd          = r >=0            
                sp_posr[pcnd] = r[pcnd]
                posr          = GaussianMLARIMA.nan_interp(sp_posr)
                pmu           = np.nanmean(posr)
                return GaussianMLARIMA.hull_sides_computer(
                    posr - pmu, b, mus + (pmu,), O, o
                )
            else:            
                sp_negr       = np.full(r.shape, np.nan)
                ncnd          = r < 0
                sp_negr[ncnd] = r[ncnd]            
                negr          = GaussianMLARIMA.nan_interp(sp_negr)
                nmu           = np.nanmean(negr)
                return GaussianMLARIMA.hull_sides_computer(
                    negr - nmu, b, mus + (nmu,), O, o
                )
        return r + sum(mus)

    __fig_confs = {
        'bbox_inches': 'tight',
        'format'     : ['eps','png'][1],
        'dpi'        : 200,
        'figsize'    : (1, 1),
    }

    def __fig_saver(self, key, spec='', spec2='', nbs_dep=True):
        figconfs = self.__fig_confs
        fname  = '{}{}[{}]{}{}.{format}'.format(
            self.model_id,
            '(%d)'%self._nb_resamples*nbs_dep,
            key,
            spec and '[%s]'%spec,
            spec2 and '[%s]'%spec2,
            **figconfs
        )
        ffname = os.path.join(
            self.save_dir,
            fname.replace(
                '>', 'gt.'
            ).translate(None, '|\_')
        )
        plt.savefig(ffname, **figconfs)
        if self.verbose:
            print('saved in ',os.path.abspath(ffname))
        return ffname

    __hull_kwarger = lambda s,fc,O,u,c=None:{
        'x'          : s._up2n_line + 1,
        'y1'         : s.hull_sides_computer(s.XACF_u, 'lo', O=O).flatten(),
        'y2'         : s.hull_sides_computer(s.XACF_u, 'up', O=O).flatten(),
        'where'      : None,
        'color'      : c or fc,
        'facecolor'  : fc,
        'label'      : r'$%d^{\mathrm{%s}}\mathrm{{hull}}$'%(O, u),
        'interpolate': True,
    }

    __legend_kws = {
        'handlelength': None,
        'numpoints'   : 1,
        'frameon'     : True,
        'prop'        : {'size':10.},
        'loc'         : 'best',
    }
    def __set_legend(self, **kws):
        lkws = dict(self.__legend_kws.items() + kws.items())

        if kws.get('handles') is None:
            handles, labels = plt.gca().get_legend_handles_labels()
            labeled_handles = cl.OrderedDict(zip(labels, handles))
            plt.legend(
                labeled_handles.values(),
                labeled_handles.keys(),
                **lkws
            )
        else:
            plt.legend(**lkws)

    @property
    def u_hull_chart(self):
        self.hull_charter(
            self.XACF_u,
            save_fig=True
        )
        self._thts_collector()

    def u_hull_chart_of(self, **kws):
        self.from_scratch(**kws)
        self.u_hull_chart

    def hull_charter(self, u, **kws):
        """ Makes the hull chart of `u`. The idea behind this chart is about
        pinpointing graphically space-dependent trend and/or variance.
        """
        
        handles = []

        # -- 4th order hull
        _hull_kws = self.__hull_kwarger('snow', 4, 'th', 'seashell')
        plt.fill_between(**_hull_kws)
        handles.append(mpa.Patch(
            color=_hull_kws['facecolor'],
            label=_hull_kws['label']
        ))        

        # -- 3rd order hull
        _hull_kws = self.__hull_kwarger('aliceblue', 3, 'rd')
        plt.fill_between(**_hull_kws)
        handles.append(mpa.Patch(
            color=_hull_kws['facecolor'],
            label=_hull_kws['label']
        ))
        
        # -- 2nd order hull
        _hull_kws = self.__hull_kwarger('lavender', 2, 'nd')
        plt.fill_between(**_hull_kws)
        handles.append(mpa.Patch(
            color=_hull_kws['facecolor'],
            label=_hull_kws['label']
        ))
        
        # -- 1st order hull
        _hull_kws = self.__hull_kwarger('gainsboro', 1, 'st')
        plt.fill_between(**_hull_kws)
        handles.append(mpa.Patch(
            color=_hull_kws['facecolor'],
            label=_hull_kws['label']
        ))
        
        # -- red line of zeros
        plt.axhline(0., color='red', linestyle='-')

        # -- residuals
        plt.plot(
            self._k_domain + 1,
            u,
            color     = 'black',
            marker    = 'o' if self.n < 100 else '.',
            linestyle = 'dashed' if self.n < 100 else 'none'
        )
        
        # -- Upper and lower bounds
        max_ = 1.05*np.max(np.abs(u))
        plt.axis((1, self.n, -max_, max_))

        # -- Legend
        self.__set_legend(handles=handles)
        
        if kws.get('save_fig', True):
            self.__fig_saver('RESID', 'HULLS', nbs_dep=False)
        if kws.get('show_fig', False):
            plt.show()            
        plt.clf()
        plt.close()
        

    @Cache._property_tmp
    def llik_maximizing_coeffs_as_list(self):
        r""" \hat{\theta}_{\mathrm{ML}} """
        return self._maximized_conc_llik_object.x

    @property
    def gammas_rhos_lambdas(self):
        """ Alias for `self.llik_maximizing_coeffs_as_list`."""
        return self.llik_maximizing_coeffs_as_list

    @Cache._property_tmp
    def conc_llik(self):
        return self._conc_llik_computer(
            self.llik_maximizing_coeffs_as_list
        )

    @staticmethod
    def dot(*terms):
        """ Serial dot product computer. Defined to avoid
        code-golfed (and difficult to read) np.dot compositions.

        Example
        -------
        >>> _2I = 2*np.identity(3)
        >>> _3I = 3*np.identity(3)
        >>> _9I = 9*np.identity(3)
        >>> GaussianMLARIMA.dot(_2I, _3I, _9I)
        array([[54.,  0.,  0.],
               [ 0., 54.,  0.],
               [ 0.,  0., 54.]])
        """
        res = None
        for term in terms[::-1]:
            res = term if res is None else np.dot(term,res)
        return res

    _crt_names = [
        ('llik'                 , 'full_llik'),
        ('HQC'                  , 'hannan_quinn'),
        ('BIC'                  , 'schwarz'),
        ('AIC'                  , 'akaike'),
        ('AICg'                 , 'akaike_green'),
        ('pr^2'                 , 'pr2'),
        ('pr^2 (pred)'          , 'pr2_e'),
        ("Sh's W"               , 'shapiro'),
        ("Sh's Pr(>|W|)"        , None),
        ("Sh's W (pred)"        , 'shapiro_e'),
        ("Sh's Pr(>|W|) (pred)" , None),
        ("BP's B"               , 'breusch_pagan'),
        ("BP's Pr(>|B|)"        , None),
        ("KB's K"               , 'koenker_bassett'),
        ("KB's Pr(>|K|)"        , None),
    ]  
    @Cache._property_tmp
    def hannan_quinn(self):
        """ Hannan–Quinn information criterion (HQC). """
        return -2.*self.full_llik + 2.*np.log(np.log(self.n))*(self.k + self.p) 
    @Cache._property_tmp
    def schwarz(self):
        """ Bayesian information criterion (BIC) or Schwarz criterion (also SBC, SBIC). """
        return -2.*self.full_llik + np.log(self.n)*(self.k + self.p)
    @Cache._property_tmp
    def akaike(self):
        """ Akaike information criterion (AIC). """
        return -2.*self.full_llik + 2.*(self.k + self.p)
    @Cache._property_tmp
    def akaike_green(self):
        return self.akaike/float(self.n) - (1. + np.log(np.pi))
    @Cache._property_tmp
    def pr2(self):
        return pow(sc.stats.pearsonr(self.y, self.yhat)[0], 2.)[0]
    @Cache._property_tmp
    def pr2_e(self):
        return pow(sc.stats.pearsonr(self.y, self.yhat_e)[0], 2.)[0]
    @Cache._property_tmp
    def shapiro(self):
        return np.array(sc.stats.shapiro(self.XACF_u))[:, None]
    @Cache._property_tmp
    def shapiro_e(self):
        return np.array(sc.stats.shapiro(self.e))[:, None]

    _dict2arr_frmt   = lambda s,d,k:np.array([d[k], d['pvalue']])[:, None]
    @Cache._property_tmp
    def _ps_testable_inst(self):
        """ Type with pysal-requested attributes.
        """
        return type(
            'OnPurp',
            (object,),
            {
                'n'  :self.n,
                'k'  :self.k,
                'u'  :self.usri,
                'utu':self.usritusri,
                'x'  :self.xri,
                'y'  :self.ysri,
                'xtx':self.xritxri,
            },
        )
    @Cache._property_tmp
    def breusch_pagan(self):
        return self._dict2arr_frmt(ps.spreg.diagnostics.breusch_pagan(
            self._ps_testable_inst
        ), 'bp')
    @Cache._property_tmp
    def koenker_bassett(self):
        return self._dict2arr_frmt(ps.spreg.diagnostics.koenker_bassett(
            self._ps_testable_inst
        ), 'kb')
    @Cache._property_tmp
    def thetas_crt(self):
        lst = []
        for i, (_, k) in enumerate(self._crt_names):
            if k is not None:
                lst.append(getattr(self, k))
        return np.vstack(lst)

    @Cache._property_tmp
    def thetas(self):
        r""" In the present computing procedure, we assume that
        $$
        \hat{\theta}_{\mathrm{ML}}
            \stackrel{a}{\sim}
                \mathcal{N}\left(
                    \theta_{0},
                    \mathrm{Var}(\hat{\theta}_{\mathrm{ML}})
                \right)
        $$
        where $\theta_{0}$ denotes the true parameter value.
        """
        return np.vstack([
            self.betas,
            self.llik_maximizing_coeffs,
            self.sig2#n_k,
        ])

    @Cache._property_tmp
    def thetas_as_list(self):
        return self.thetas.flatten().tolist()

    def _full_llik_computer(self, _thetas):
        """ Returns (full) log-likelihood. Formulated so as to obtain second
        derivatives with respect to the parameters.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._AR_ks = [1, 2]
        >>> o._full_llik_computer([
        ...    43.06976402678653   , -0.9483216077864456, -0.2695103444402416,
        ...     0.19099696507708683,  0.2588788504534716, 92.61892002235702  ,
        ... ])
        array([[-179.96412919]])

        Note that just above `o._full_llik_computer` is actually evaluated
        at $\hat{\theta}_{\mathrm{ML}}$, code-namely `o.thetas`.
        >>> o._full_llik_computer(o.thetas_as_list)
        array([[-179.82717792]])
        """
        k                    = self.k
        p                    = self.p
        kp                   = k + p
        _betas               = _thetas[:k]
        _gammas_rhos_lambdas = _thetas[k:kp]
        _sig2                = _thetas[kp]
        _ysri, _xri, _G, _S, _Ri, _ = self._yNx_filterer(
            _gammas_rhos_lambdas
        )
        _r            = _ysri - np.dot(_xri, np.array(_betas)[:, None])
        _rss          = np.dot(_r.T, _r)
        _RiSG         = np.dot(_Ri, np.dot(_S, _G))
        _jac          = np.linalg.det(_RiSG)
        _fgll         = self._full_llik_core_computer(
            jacobian = _jac,
            variance = _sig2,
            rss      = _rss
        )
        return _fgll
    def _full_llik_core_computer(self, jacobian, variance, rss):
        """ Returns (full) log-likelihood.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> jac, sig2, rss = 0.136961999519103, 92.6189200224, 4075.23248098
        >>> o._full_llik_core_computer(jac, sig2, rss)
        -179.96412919101235
        
        Note that just above `o._full_llik_core_computer` is actually evaluated
        at quantities which derive from $\hat{\theta}_{\mathrm{ML}}$, in the
        AR{1,2} case.
        >>> o._AR_ks = [1, 2]
        >>> o.full_llik.item()
        -179.82717791831826
        """
        return np.log(jacobian) - .5*self.n * np.log(2.*np.pi*variance)\
               - rss/(2.*variance)

    @Cache._property_tmp
    def llik(self):
        """ Returns log-likelihoods.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o.llik
        array([[-187.37723881]])
        >>> o.full_llik
        array([[-187.37723881]])

        >>> o.from_scratch(ER_ks=[1])
        >>> o.llik
        array([[-184.53273963]])
        >>> o.full_llik
        array([[-184.53273963]])

        >>> o.from_scratch(
        ...     AR_ks=[1, 2],
        ...     MA_ks=[4],
        ...     ER_ks=[12],
        ... )
        >>> o.llik
        array([[-176.217913]])
        >>> o.full_llik
        array([[-176.217913]])
        """
        return self.conc_llik - .5*self.n * ( np.log(2.*np.pi) + 1.)

    @Cache._property_tmp
    def full_llik(self):
        """ Log-likelihood evaluated at $\hat{\theta}_{\mathrm{ML}}$.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._AR_ks = [12, 22]
        >>> o._full_llik_computer([
        ...    79.19795367735526   , -1.6822507035637728 ,  -0.26919340930707486,
        ...    -0.14094043293817296, -0.08509912061832947, 116.32195627123622   ,
        ... ])
        array([[-186.16744798]])

        Note that just above `o._full_llik_computer` is actually evaluated
        at $\hat{\theta}_{\mathrm{ML}}$, code-namely `o.thetas`.
        >>> o.full_llik
        array([[-186.16744798]])
        """
        return self._full_llik_computer(
            self.thetas_as_list
        )    

    @Cache._property_tmp
    def full_llik_hessian_computer(self):
        r"""
        Returns a callable capable of computing hessian matrices evaluated
        at some (hyper-) point. Remember that this hessian matrices' components
        consist of second derivatives of the likelihood function with respect to the
        parameters. The Hessian is defined as:
        $$
        \mathbf{H}(\theta)=
            \frac{\partial^{2}}{\partial\theta_{i}\partial\theta_{j}}
            l(\theta),
            ~~~~ 1\leq i, j\leq p
        $$

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._ER_ks, o._AR_ks, o._MA_ks = [1], [2], [4, 5]
        >>> hcomputer = o.full_llik_hessian_computer
        >>> type(hcomputer)
        <class 'numdifftools.core.Hessian'>

        We can now evaluate this function, say, at $\hat{\theta}_{\mathrm{ML}}$,
        >>> computed_h = hcomputer(o.thetas_as_list)
        >>> computed_h
        array([[  -0.21593187,   -3.16953753,   -9.23825195,   -0.05218509,   -6.53993419,   -0.19918455,    0.09379909,    0.        ],
               [  -3.16953753,  -79.717884  , -204.10758101,    3.57421133,  -70.51950825,  -13.65024788,   27.70386894,   -0.        ],
               [  -9.23825195, -204.10758101, -798.83993948,   -2.21191966, -231.20684196,   68.49722196,  127.95296062,    0.        ],
               [  -0.05218509,    3.57421133,   -2.21191966, -118.03070229,  -52.79686865,   -0.00619979,   -7.48157319,   -0.0422948 ],
               [  -6.53993419,  -70.51950825, -231.20684196,  -52.79686865, -339.86316692,   -5.56253156,  -20.47547473,   -0.03591995],
               [  -0.19918455,  -13.65024788,   68.49722196,   -0.00619979,   -5.56253156, -109.38337939,  -57.48321859,    0.10652965],
               [   0.09379909,   27.70386894,  127.95296062,   -7.48157319,  -20.47547473,  -57.48321859,  -77.2689966 ,    0.02243485],
               [   0.        ,   -0.        ,    0.        ,   -0.0422948 ,   -0.03591995,    0.10652965,    0.02243485,   -0.00332004]])

        Which, multiplied by -1, equals the so-called observed fisher matrix.
        >>> (-computed_h == o.obs_fisher_matrix).all()
        True
        """
        return nd.Hessian(
            self._full_llik_computer,
##            step = nd.MinStepGenerator(     P
##                base_step=self._nd_step,    R
##                step_ratio=None,            O
##                num_extrap=0                B
##            ),                              L
##            order  = self._nd_order,        E
##            method = self._nd_method        M
        )

    @Cache._property_tmp
    def obs_fisher_matrix(self):
        r"""
        The so-called observed information matrix, which consists of the
        Fisher information matrix, $\mathbf{I}(\theta)$, evaluated at the
        maximum likelihood estimates (MLE), $\hat{\theta}_{\mathrm{ML}}$,
        i.e.
        $$
        \mathbf{I}(\hat{\theta}_{\mathrm{ML}})=
            -\mathbf{H}(\hat{\theta}_{\mathrm{ML}})
        $$

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ... )
        >>> o.obs_fisher_matrix
        array([[  0.39917586,   5.73812859,  15.34281308,  -0.        ],
               [  5.73812859,  95.20485582, 241.13840591,  -0.        ],
               [ 15.34281308, 241.13840591, 723.05916578,  -0.        ],
               [ -0.        ,  -0.        ,  -0.        ,   0.00162593]])
        """
        return - self.full_llik_hessian_computer(
            self.thetas_as_list
        )

    @Cache._property_tmp
    def obs_cov_matrix(self):
        r"""
        Estimation of the asymptotic covariance matrix, which is the inverse
        of the Fisher information matrix, here evaluated at
        $\hat{\theta}_{\mathrm{ML}}$.
        $$
        \mathrm{Var}(\hat{\theta}_{\mathrm{ML}})=
            [\mathbf{I}(\hat{\theta}_{\mathrm{ML}})]^{-1}
        $$

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ... )
        >>> o.obs_cov_matrix
        array([[ 21.05188021,  -0.88465637,  -0.15167561,  -0.        ],
               [ -0.88465637,   0.10480806,  -0.01618143,   0.        ],
               [ -0.15167561,  -0.01618143,   0.00999793,  -0.        ],
               [ -0.        ,   0.        ,  -0.        , 615.03174058]])

        In accordance with its theoretically-grounded counterpart
        >>> o.betas_cov
        array([[21.05188021, -0.88465637, -0.15167561],
               [-0.88465637,  0.10480806, -0.01618143],
               [-0.15167561, -0.01618143,  0.00999793]])
        """
        return np.linalg.inv(
            self.obs_fisher_matrix
        )
    @property
    def thetas_cov(self):
        """ Alias for `self.obs_cov_matrix`."""
        return self.obs_cov_matrix
    def thetas_cov_of(self, **kws):
        self.from_scratch(**kws)
        return self.thetas_cov
    @Cache._property_tmp
    def thetas_se(self):
        """ Estimated standard error of the ML estimates, given by:
        $$
        \mathrm{SE}(\hat{\theta}_{\mathrm{ML}})=
            \sqrt{ \mathrm{D}[\mathrm{Var}(\hat{\theta}_{\mathrm{ML}})]  }
        $$
        where $\mathrm{D}[.]$ stands for an operator which takes diagonales.
        """
        return np.sqrt(self.thetas_cov.diagonal())[:, None]

    @staticmethod
    def t_tests_computer(stts, n, k, p):
        """ Returns a matrix whose second and third columns consist
        of probability-space images of a sample individual, respectively
        following a student and a standard normal distribution.

        Example
        -------
        >>> stts    = -2.04522961110887
        >>> n, k, p = 30, 1, 0
        >>> GaussianMLARIMA.t_tests_computer(stts, n, k, p)
        array([-2.04522961,  0.05      ,  0.04083223])
        """
        abs_stts = np.abs(stts)
        df = n - k - p
        ts = 2*sc.stats.t.sf(abs_stts, df)
        zs = 2*sc.stats.norm.sf(abs_stts)
        return np.hstack([
            stts,
            ts,
            zs,
        ])

    @Cache._property_tmp
    def thetas_tt(self):
        return self.t_tests_computer(
            self.thetas/self.thetas_se,
            self.n,
            self.k,
            self.p
        )

    @Cache._property_tmp
    def _stt_names(self):
        _100cl = 1e2*self.clevel
        return [
            'Estimate',
            'Std. Error',
            't|z value',
            'Pr(>|t|)',
            'Pr(>|z|)',
            '{:.1f}% CI.lo.'.format(_100cl),
            '{:.1f}% CI.up.'.format(_100cl),
        ]
    @Cache._property_tmp
    def thetas_loB(self):
        if self.n > 30:
            return self.thetas - self.z_bila*self.thetas_se*pow(self.n, -.5)
        return self.thetas - self.t_bila*self.thetas_se*pow(self.n, -.5)

    @Cache._property_tmp
    def thetas_upB(self):
        if self.n > 30:
            return self.thetas + self.z_bila*self.thetas_se*pow(self.n, -.5)
        return self.thetas + self.t_bila*self.thetas_se*pow(self.n, -.5)

    @Cache._property_tmp
    def thetas_stt(self):
        return np.hstack([
            self.thetas,
            self.thetas_se,
            self.thetas_tt,
            self.thetas_loB,
            self.thetas_upB,
        ])
    def thetas_stt_of(self, **kws):
        self.from_scratch(**kws)
        return self.thetas_stt

    @Cache._property_tmp
    def betas_tt(self):
        return self.t_tests_computer(
            self.betas/self.betas_se,
            self.n,
            self.k,
            self.p
        )            

    @Cache._property_tmp
    def llik_maximizing_coeffs(self):
        return np.array(self.llik_maximizing_coeffs_as_list)[:, None]

    @Cache._property_tmp
    def _maximized_conc_llik_object(self):
        """ Object of type `<class 'scipy.optimize.optimize.OptimizeResult'>`
        whose one of attributes is, hopefully, the concentrated underlier of
        $\hat{\theta}_{\mathrm{ML}}$, the spatial parameter(s).
        NB : when the concentrated log-likelihood is evaluated with no*
        spatial parameters, the optimizer warns that it has found no
        solutions. This is simply because the (empty) initial guess IS
        the solution: when concentrated with respect to nothing, the
        log-likelihood is actually evaluated at its analytical solutions.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._AR_ks = [1]
        >>> optzd = o._maximized_conc_llik_object
        >>> optzd
         final_simplex: (array([[0.32015173],
               [0.32015173]]), array([112.66576367, 112.66576367]))
                   fun: 112.66576366678524
               message: 'Optimization terminated successfully.'
                  nfev: 106
                   nit: 48
                status: 0
               success: True
                     x: array([0.32015173])

        The spatial autoregressive parameter is
        >>> optzd.x
        array([0.32015173])

        The value of the concentrated log-likelihood at `optzd.x` is
        >>> - optzd.fun
        -112.66576366678524

        which is stored in `o.conc_llik`
        >>> o.conc_llik
        array([[-112.66576367]])
        
        """
        x0 = self.p*[.0]
        try:
            optzd = sc.optimize.minimize(
                x0      = x0,
                fun     = lambda xN: -self._conc_llik_computer(xN),
                tol     = self._tolerance,
                method  = 'Nelder-mead',
                options = {
                    'disp'   : self.p and self.opverbose,
                    'maxiter': self.p*200,
                    'maxfev' : self.p*200,
                },
            )          
            return optzd                
        except Exception as exc:
            print(exc.message, end="\n")
            return type(
                'optimerr', (object,), {
                    'success':None, 'x':x0
                }
            )

    def _conc_llik_computer(self, _gammas_rhos_lambdas):
        """ Returns concentrated log-likelihood.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._AR_ks = [1, 2]
        >>> o._conc_llik_computer([
        ...    0.19099696507708683, 0.2588788504534716,
        ... ])
        array([[-110.29918979]])

        Note that just above `o._conc_llik_computer` is actually evaluated
        evaluated at $\hat{\theta}_{\mathrm{ML}}$'s spatial-determinants,
        namely, `o.gammas_rhos_lambdas`.
        >>> o._conc_llik_computer(o.gammas_rhos_lambdas)
        array([[-110.29918979]])
        """
        _ysri, _xri, _G, _S, _Ri, _ = self._yNx_filterer(
            _gammas_rhos_lambdas
        )
        n             = self.n
        xrit          = _xri.T
        _xritxri      = np.dot(xrit, _xri)
        _xritxriixrit = np.linalg.solve(_xritxri, xrit)        
        _betas        = np.dot(_xritxriixrit, _ysri)
        _xribetas     = np.dot(_xri, _betas)
        _r            = _ysri - _xribetas
        _rss          = np.dot(_r.T, _r)
        _sig2         = _rss/n
        _RiSG         = np.dot(_Ri, np.dot(_S, _G))
        _jac          = np.linalg.det(_RiSG)
        _cgll         = self._conc_llik_core_computer(
            jacobian = _jac,
            variance = _sig2
        )
        return _cgll

    def _conc_llik_core_computer(self, jacobian, variance):
        """ Returns concentrated log-likelihood.

        Example
        -------
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> jac, sig2 = 0.136961999519103, 92.6189200224
        >>> o._conc_llik_core_computer(jac, sig2)
        -112.93614106401363
        
        Note that just above `o._conc_llik_core_computer` is actually evaluated
        at quantities which derive from $\hat{\theta}_{\mathrm{ML}}$ in the
        AR{1,2} case.
        >>> o._AR_ks = [1, 2]
        >>> o.full_llik.item()
        -179.82717791831826
        """
        return np.log(jacobian) - .5*self.n*np.log(variance)

    def _yNx_filterer(self, gammas_rhos_lambdas):
        r""" Returns `gammas_rhos_lambdas`-determined spatial filters
        and filtered model's variables.
        NB: this methid is defined to comply with the D.R.Y. principle.
        Its core is involved in self._conc_llik_core_computer as well
        as in self._full_llik_core_computer.

        Chk
        ---
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     opverbose = False,
        ... )
        >>> o._AR_ks = [1, 2]
        >>> h = o.hash_(o._yNx_filterer([
        ...    0.19099696507708683, 0.2588788504534716,
        ... ]))
        >>> yNx_filterer_as_expected = 'jmRSJwC/AqELU/fumF6U2A==\n' == h
        >>> yNx_filterer_as_expected
        True
        """
        ER_p   = self.ER_p
        AR_p   = self.AR_p
        ERAR_p = ER_p + AR_p

        rs_ER_w_collection = self.ER_w_collection
        rs_AR_w_collection = self.AR_w_collection
        rs_MA_w_collection = self.MA_w_collection

        Idn = self.Idn
        y0, y = copy.copy(self.y), copy.copy(self.y)
        x0, x = copy.copy(self.x), copy.copy(self.x)

        _gammas  = gammas_rhos_lambdas[:ER_p]
        _rhos    = gammas_rhos_lambdas[ER_p:ERAR_p]
        _lambdas = gammas_rhos_lambdas[ERAR_p:]

        _G = copy.copy(Idn)
        for i, _w_i in enumerate(rs_ER_w_collection):
            _gamma_i  = _gammas[i]
            _G       -= _gamma_i*_w_i
            y        -= _gamma_i*np.dot(_w_i, y0)
            x        -= _gamma_i*np.dot(_w_i, x0)

        _S    = copy.copy(Idn)
        ys_rs = copy.copy(y)
        for i, _w_i in enumerate(rs_AR_w_collection):
            _rho_i    = _rhos[i]
            _S       -= _rho_i*_w_i
            ys_rs    -= _rho_i*np.dot(_w_i, y)

        _R = copy.copy(Idn)
        for i,_w_i in enumerate(rs_MA_w_collection):
            _lambda_i = _lambdas[i]
            _R       += _lambda_i*_w_i
        if len(rs_MA_w_collection):
            _Ri       = np.linalg.inv(_R)
            _ysri     = np.dot(_Ri, ys_rs)
            _xri      = np.dot(_Ri, x)
        else:
            _Ri       = copy.copy(Idn)
            _ysri     = ys_rs
            _xri      = x        
        
        return _ysri, _xri, _G, _S, _Ri, _R

#*****************************************************************************#
##    ═╗ ╦╔═╗╔═╗╔═╗╔╦╗┌─┐┬┌─┌─┐┬─┐
##    ╔╩╦╝╠═╣║  ╠╣ ║║║├─┤├┴┐├┤ ├┬┘
##    ╩ ╚═╩ ╩╚═╝╚  ╩ ╩┴ ┴┴ ┴└─┘┴└─
class XACFMaker(GaussianMLARIMA):

    type_I_err = .05
    @Cache._dep_property('type_I_err')
    def clevel(self):return 1. - self.type_I_err
    @Cache._dep_property('type_I_err')
    def z_unil(self): return sc.stats.norm.ppf(self.clevel, loc=0, scale=1)
    @Cache._dep_property('type_I_err')
    def l_bound(self): return self.type_I_err/2.    
    @Cache._dep_property('type_I_err')
    def r_bound(self): return 1. - self.l_bound     
    @Cache._dep_property('type_I_err')
    def z_bila(self): return sc.stats.norm.ppf(self.r_bound, loc=0, scale=1)
    @Cache._dep_property('type_I_err')
    def t_bila(self): return sc.stats.t.ppf(self.r_bound, self.n - self.k - self.p)

    __AK = 'ACF'
    __PK = 'PACF'
    __RK = 'Rs'
    __MK = 'Ms'
    __GK = 'Gs'
    __BK = 'Bs'
    __QK = 'Qs'
    
    __stts_tmpl = {
        __AK:{
            __RK: None,
            __MK: None,
            __GK: None,
            __BK: None,
        },
        __PK:{
            __RK: None,
            __QK: None,
        }
    }

    __BnQ_figs_kwarger = lambda s,n:{
        'i'     : 3,
        'meth'  : 'plot',
        'color' : 'red',
        'label' : r"${}\ {:1.1f}\%$".format(
            "\mathrm{%s's}"%n,
            round(1e2*s.type_I_err)
        ),
        'marker' : '_',
        'linestyle': 'None',
    }
    
    @property
    def __figs_kwargs(self):
        """
        Check existence
        of relevant keys
        ----------------
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> tmpl = o._XACFMaker__stts_tmpl
        >>> tstr = lambda K: getattr(o, '_XACFMaker__%sK'%K) in tmpl
        >>> AP_in_stts_tmpl = all(map(tstr, ['A', 'P']))
        >>> AP_in_stts_tmpl
        True

        >>> tmpl = o._XACFMaker__stts_tmpl[o._XACFMaker__AK]
        >>> tstr = lambda K: getattr(o, '_XACFMaker__%sK'%K) in tmpl
        >>> RMGB_in_stts_tmpl_A = all(map(tstr, ['R', 'M', 'G', 'B']))
        >>> RMGB_in_stts_tmpl_A
        True

        >>> tmpl = o._XACFMaker__stts_tmpl[o._XACFMaker__PK]
        >>> tstr = lambda K: getattr(o, '_XACFMaker__%sK'%K) in tmpl
        >>> RQ_in_stts_tmpl_P = all(map(tstr, ['R', 'Q']))
        >>> RQ_in_stts_tmpl_P
        True

        >>> tstr = lambda K: hasattr(o, '_XACFMaker__%sK'%K)
        >>> RMGBQ_as_attr = all(map(tstr, ['R', 'M', 'G', 'B', 'Q']))
        >>> RMGBQ_as_attr
        True

        >>> fkws = o._XACFMaker__figs_kwargs
        >>> tstr = lambda K: getattr(o, '_XACFMaker__%sK'%K) in fkws
        >>> RMGBQ_in_fkws = all(map(tstr, ['R', 'M', 'G', 'B', 'Q']))
        >>> RMGBQ_in_fkws
        True
        """
        return {
            self.__RK:{
                'i'     : 0,
                'meth'  : 'bar',
                'color' : 'black',
                'label' : None,
                'width' : .05,
            },
            self.__MK:{
                'i'     : 1,
                'meth'  : 'plot',
                'color' : 'green',
                'label'  : r"$\mathrm{Moran's \ I-E[I]}$",
                'marker'  : '.',
                'linestyle': 'None',
                'markersize': 10,
            },
            self.__GK:{
                'i'     : 2,
                'meth' : 'plot',
                'color' : 'orange',
                'label'  : "$\mathrm{Geary's \ E[C]-C}$",
                'marker'  : 'x',
                'linestyle': 'None',
                'markersize': 5,
            },
            self.__BK:self.__BnQ_figs_kwarger('Bartlett'),
            self.__QK:self.__BnQ_figs_kwarger('Quenouille'),
        }
    
    def __init__(self, **kws):
        super(XACFMaker, self).__init__(**kws)
        self._k_domain = self._up2n_line[:, None]
        self._k_domain_aslist  = self._up2n_line.tolist()
        self._empty_arr = np.zeros_like(self._k_domain).astype(np.float64)

    @Cache._method
    def Bartlett(self, Rs):
        r""" Bartlett's significance threshold [3]_.

        Chk
        ---
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> o.type_I_err = .05
        >>> np.random.seed(0)
        >>> Rs = np.random.normal(0, 1, size=o.n)[:, None]
        >>> v  = o.Bartlett(Rs)
        >>> hash_ref = 'ec6fYBPx1EXxT/HuaaM1TA==\n'
        >>> Bartlett_as_expected = o.hash_(v) ==  hash_ref
        >>> Bartlett_as_expected
        True
        """
        v = self.z_unil*np.array([
            pow((np.sum(
                    pow(np.array(Rs[:k+1]), 2)
                 )*2 + 1.)
            /self.n, .5)
            for k,r in enumerate(Rs)
        ])[:, None]
        return np.hstack([v, -v])

    @Cache._property
    def Quenouille(self):
        r""" Quenouille's significance threshold [2]_.

        Chk
        ---
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> o.type_I_err = .05
        >>> v  = o.Quenouille
        >>> hash_ref = 'l7gWSnn7Vhohx+PzGIOaiA==\n'
        >>> Quenouille_as_expected = o.hash_(v) == hash_ref
        >>> Quenouille_as_expected
        True
        """
        v = self.ones*self.z_unil/pow(self.n, 0.5)
        return np.hstack([v, -v])

    _m_stdzer = lambda s,m: (m.I - m.EI).item()
    _g_stdzer = lambda s,g: (g.EC - g.C).item()
    @Cache._method
    def XACF_computer(self, u):
        r""" Returns some (P)ACF-related statistics.

        Example
        -------
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ... )
        >>> xacf_stts = o.XACF_computer(u=o.XACF_u)

        Chk 
        ---
        >>> o.type_I_err = .05
        >>> hashs_ref = {
        ...     'u': 'VuzlpLO+nRQJVPTEDUDv6w==\n',
        ...     'PACF':(
        ...         ('Qs', 'l7gWSnn7Vhohx+PzGIOaiA==\n'),
        ...         ('Rs', '6+FSrlcQXQLSx1iO5nLnDQ==\n')
        ...     ),
        ...     'ACF': (
        ...         ('Bs', 'RGYusgaf/BYAJu1r8O15yA==\n'),
        ...         ('Gs', 'nyjG/8cgOc4X89DzsTHHfQ==\n'),
        ...         ('Ms', '2vmYqYdhxxGOQmzS1pmR3A==\n'),
        ...         ('Rs', '5K9ljWV22t9jUYf/4FdcQA==\n')
        ...     )
        ... }
        >>> hashs_ref_as_set = set(tuple(hashs_ref.items()))
        >>> hashs_chk =  {'u':o.hash_(o.XACF_u)}
        >>> for xacf, stts in xacf_stts.items():
        ...     hashs_chk[xacf] = tuple((k, o.hash_(v)) for k,v in sorted(stts.items()))
        >>> hashs_chk # doctest: +SKIP
        >>> _XACF_as_expected = set(tuple(hashs_chk.items())) == hashs_ref_as_set
        >>> _XACF_as_expected
        True
        """
        
        stats = copy.deepcopy(self.__stts_tmpl)
        for dict_ in stats.values():
            for key in dict_.keys():
                dict_[key] = self._empty_arr.copy()

        acf_stats = stats[self.__AK]
        acf_r     = acf_stats[self.__RK]
        acf_m     = acf_stats[self.__MK]
        acf_g     = acf_stats[self.__GK]
        
        pacf_stats = stats[self.__PK]
        pacf_r     = pacf_stats[self.__RK]
        pacf_x     = [self.ones]
        
        for k in self._k_domain_aslist:
            if k == 0:
                continue

            # --- (P)ACF related
            W  = self.oknn_computer(k=k, as_psw=True)
            Wu = ps.lag_spatial(W, u)

            # --- ACF related          
            acf_r[k,:] = ps.spreg.ols.BaseOLS(
                y=u, x=np.hstack([self.ones, Wu])
            ).betas[-1,-1]
            acf_m[k,:] = self._m_stdzer(
                ps.Moran(y=u, w=W, two_tailed=True, permutations=None)
            )
            acf_g[k,:] = self._g_stdzer(
                ps.Geary(y=u, w=W, permutations=None)
            )            

            # --- PACF related
            pacf_x.append(Wu)           
            pacf_r[k,:] = ps.spreg.ols.BaseOLS(
                    y=u, x=np.hstack(pacf_x)
            ).betas[-1,-1]

        acf_stats[self.__BK] = B = self.Bartlett(
            acf_stats[self.__RK]
        )
        pacf_stats[self.__QK] = Q = self.Quenouille

        ##self._k_domain[np.abs(acf_r)-B[:,:1]>0]
        ##self._k_domain[np.abs(pacf_r)-Q[:,:1]>0]
        
        # 0 to nan
        for key, arr in acf_stats.items():
            arr[arr==0] = np.nan
        return stats

    @property
    def u_XACF_chart(self):
        self.XACF_charter(
            u = self.XACF_u,
            save_fig = True
        )
        self._thts_collector()

    def u_XACF_chart_of(self, **kws):
        self.from_scratch(**kws)
        self.u_XACF_chart

    def XACF_charter(self, **kws):
        """ Plots both ACFs and PACFs following formats defined
        via self._figs_kwarger.      
        """
        if 'u' in kws:
            stats = self.XACF_computer(kws['u'])
        elif 'stats' in kws:
            stats = kws['stats']
        else:
            raise TypeError(
                "XACF_charter() takes at least either "
                "stats (XACF_computer's output) "
                "or u (XACF_computer's input) as keyword argument."
            )
            
        xacf_keys  = stats.keys()
        saved_figs = dict.fromkeys(xacf_keys, ())
        
        x_values = self._k_domain
        figconfs = self._GaussianMLARIMA__fig_confs

        for xacf_key, sbp in zip(xacf_keys, [212, 211]):

            plt.subplot(sbp)
            plt.grid()
            plt.axis((
                x_values.min()-1,
                x_values.max(),
                -1.2,
                1.2
            ))
            
            figs_kwargs = self.__figs_kwargs      
            sorted_items = sorted(
                stats[xacf_key].items(),
                key=lambda(k, _) :figs_kwargs[k].pop('i'),
                reverse=True
            )        
            for stat, y_values in sorted_items:
                y_kwargs = figs_kwargs[stat]
                obj = getattr(plt, y_kwargs.pop('meth'))(
                    x_values, y_values, **y_kwargs
                )

            self._GaussianMLARIMA__set_legend(
                handlelength=1
            )
        
        if kws.get('save_fig', True):
            self._GaussianMLARIMA__fig_saver('RESID', '(P)ACF', nbs_dep=False)
        if kws.get('show_fig', False):
            plt.show()

        plt.clf()
        plt.close()

#*****************************************************************************#
##    ╔═╗┌─┐┌┬┐┌─┐┬  ┌─┐┬─┐
##    ╚═╗├─┤│││├─┘│  ├┤ ├┬┘
##    ╚═╝┴ ┴┴ ┴┴  ┴─┘└─┘┴└─
class Sampler(XACFMaker):
    def __init__(self, **kws):
        super(Sampler, self).__init__(**kws)
        self.Idn0     = copy.copy(self.Idn)
        self.y0       = copy.copy(self.y)
        self.x0       = copy.copy(self.x)
        self.n0       = copy.copy(self.n)
        self._jk_i = None
        self._bt_s = None

    @Cache._property
    def Idn_jk(self):
        """ Stores (once for all) and returns identity matrices involved
        during the Jackknife procedure.
        """
        return np.identity(self.n-1)

    def get_jk_w_collector_and_ixs(self):
        """ Returns (i) the callable to use in order to build oknn matrices that
        do not take the jackknife's removed individual -- the `_jk_i`th -- into
        account and (ii) the indexes of individuals that are still considered.
        """
        ixs       = np.delete(self._up2n_line, self._jk_i)
        dmat_jk_i = copy.copy(self.full_dmat)[ixs].T[ixs].T
        smat_jk_i = self.smat_computer(dmat_jk_i)
        return lambda ks: self._w_collector(
            lambda k: self._oknn_computer(
                smat_jk_i,
                k,
                as_psw=False
            ),
            ks
        ), ixs        

    def _set_hat_env(self, **kws):
        self.from_scratch(**kws)
        _c        = self._cache
        _c['Idn'] = self.Idn0
        _c['y']   = self.y0
        _c['x']   = self.x0
        _c['n']   = self.n0

    def _set_jk_i_env(self, **kws):
        self.from_scratch(**kws)
        _c, _t = self._cache, self._tempo  
        _w_cllctr, ixs = self.get_jk_w_collector_and_ixs()
        _c['Idn'] = self.Idn_jk
        _c['y']   = self.y0[ixs]
        _c['x']   = self.x0[ixs]
        _c['n']   = self.n0 - 1
        _t['ER_w_collection'] = _w_cllctr(self.ER_ks)
        _t['AR_w_collection'] = _w_cllctr(self.AR_ks)
        _t['MA_w_collection'] = _w_cllctr(self.MA_ks)

    def _set_bt_s_env(self, **kws):
        self.from_scratch(**kws)
        _c      = self._cache
        ixs     = self.get_bt_ixs()
        _c['y'] = np.dot(
            self.RiSGi0,
            self.ysrihat0 + self._r_rs_standardizer(
                self.usri0[ixs], self.levscale0
            )
        )

    @Cache._property
    def grandi_vector(self):
        """ Stores (once for all) and returns an exponentiated vector of -1,
        as done in Grandi's series.
        """
        return np.power(-self.ones, self._up2n_line[:, None])

    def _r_rs_standardizer(self, r_rs, lsca):         
        n          = self.n
        _unbiaser  = np.sqrt(n/(n - 1))
        _av_scaler = -np.sum(r_rs*lsca)/n
        return _unbiaser*(r_rs*lsca + _av_scaler)*self.grandi_vector

    def get_bt_ixs(self):
        """ Returns the resampled list of individuals' index that are kept
        so as to compute the bootstrap estimates.
        """
        np.random.seed(self._bt_s)
        return np.floor(
            np.random.rand(self.n)*self.n
        ).astype(int)


#*****************************************************************************#
##    ╔╦╗┌─┐┌┬┐┬─┐┬┌─┐┬┌─┐┌┐┌
##    ║║║├┤  │ ├┬┘││  │├─┤│││
##    ╩ ╩└─┘ ┴ ┴└─┴└─┘┴┴ ┴┘└┘
class Metrician(Sampler):
    
    __JK = 'JK'
    __BT = 'BT'
    __HT = 'HAT'
    
    __axhliner = lambda c,l,ls: lambda y,larg:{
        'meth'     : 'axhline',
        'y'        : y,
        'color'    : c,
        'label'    : l%larg,
        'linestyle': ls,
    }
    
    __figs_kwarger = {
        __HT: __axhliner(
            'red', '$\mathrm{\widehat{%s}}$', 'solid'
        ),
        __BT: __axhliner(
            'blue', '$[\mathrm{\overline{%s}}]_{bt}$', 'dashed'
        ),
        __JK: __axhliner(
            'green', '$[\mathrm{\overline{%s}}]_{jk}$', '-.'
        ),
    }

    def __figs_plter(self, key, val, larg):
        kws = self.__figs_kwarger[key](
            val, larg
        )
        getattr(plt, kws.pop('meth'))(
            **kws
        )

    def __init__(self, **kws):
        super(Metrician, self).__init__(**kws)
        self._nb_resamples = int(kws.get('nbsamples', 2000))

    @Cache._property_tmp
    def _key2rnames(self):
        return {
            'par':self.par_names,
            'crt':map(lambda (n,k):n, self._crt_names),
            'cov':self.par_names,
            'stt':self.par_names,
        }

    def _run(self): self.thetas

    @Cache._property_tmp
    def _thts_tmpl(self):
        """ `self.to_save`-dependent template for saving
        results.
        """
        return {
            key:{'stack': None}
            for key in self.to_save
        }
    def _i_results_computer(self, keyed_thts, obj):        
        if keyed_thts['stack'] is None:
            stack     = obj
            mean_conv = obj
            std_conv  = np.full_like(obj, np.nan)
        else:
            stack = np.dstack((
                keyed_thts['stack'], obj
            ))
            mean_conv = np.dstack((
                keyed_thts['mean_conv'],
                np.nanmean(
                    stack, axis=2, keepdims=True
                )
            ))
            std_conv = np.dstack((
                keyed_thts['std_conv'],
                np.nanstd(
                    stack, axis=2, keepdims=True, ddof=1
                )
            ))
        return stack, mean_conv, std_conv
        
    def _save_i_results(self, thts):        
        other_objs_to_save = [
            (key, getattr(self, 'thetas_%s'%key, self.thetas))
            for key in self.to_save         
        ]
        for key, obj in other_objs_to_save:
            st, mc, sc = self._i_results_computer(thts[key], obj)
            thts[key]['stack']     = st
            thts[key]['mean_conv'] = mc
            thts[key]['std_conv']  = sc

    def _save_results(self, thts):
        for key, obj in thts.items():
            stack = obj['stack']
            obj['mean'] = np.nanmean(stack,
                axis=2, keepdims=False
            )
            obj['std'] = np.nanstd(stack,
                axis=2, keepdims=False, ddof=1
            )

    def hat_run(self, **kws):
        """ "Public" wrapper of self._hat_run.

        Example
        -------
        >>> o = Metrician(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )

        Let's first make an example with no spatial parameters.
        >>> run_kwargs = {
        ...     'ER_ks'    : [],
        ...     'AR_ks'    : [],
        ...     'MA_ks'    : [],
        ... }
        >>> hat_results = o.hat_run(**run_kwargs)
        >>> o.thetas_cov
        array([[ 21.05188021,  -0.88465637,  -0.15167561,  -0.        ],
               [ -0.88465637,   0.10480806,  -0.01618143,   0.        ],
               [ -0.15167561,  -0.01618143,   0.00999793,  -0.        ],
               [ -0.        ,   0.        ,  -0.        , 615.03174058]])

        Then an example with multiple spatial parameters, e.g. involving (i) a
        1st neighbor-order-based (partial) differencing, (ii) a 2nd neighbor-
        order-based autoregressive process in the (diff.) regressand and (iii)
        (iii) a 3rd neighbor-order-based moving average process in the (diff.)
        residuals.
        >>> run_kwargs = {
        ...     'ER_ks'    : [1],
        ...     'AR_ks'    : [2],
        ...     'MA_ks'    : [3],
        ... }
        >>> hat_results = o.hat_run(**run_kwargs)
        >>> o.thetas_cov
        array([[ 51.28218262,  -1.72251913,   0.005793  ,   0.13816817,  -0.77809545,   0.58460032,  22.53934157],
               [ -1.72251913,   0.11802467,  -0.01781421,   0.00617914,   0.02257574,  -0.02903433,  -0.70596416],
               [  0.005793  ,  -0.01781421,   0.00927433,   0.00031505,  -0.00357443,   0.01018458,   0.09167906],
               [  0.13816817,   0.00617914,   0.00031505,   0.01972134,  -0.00727362,   0.00163422,   0.08249111],
               [ -0.77809545,   0.02257574,  -0.00357443,  -0.00727362,   0.01866495,  -0.01850644,  -0.50307512],
               [  0.58460032,  -0.02903433,   0.01018458,   0.00163422,  -0.01850644,   0.05185993,   0.47629102],
               [ 22.53934157,  -0.70596416,   0.09167906,   0.08249111,  -0.50307512,   0.47629102, 340.95796717]])

        NB: Covariance matrices (components) are not subject to bootstrapping
         for performance reasons: this implies computing `self.obs_fisher_matrix`
         `self._nb_resamples` times. However, this can easily be turned on by
         uncommenting the elements `'cov'` and `'stt'` of the list-attribute
         `self.to_save`. However, doing so has side effects on self.summary()
         since tables related to covmats and t-tests are nonsensically
         concatenated.
        """
        self._set_hat_env(**kws)
        thts = self._hat_run(cid=self.model_id, **kws)
        self._thts_collector(
            key='hat', thts=thts
        )
        return thts
    @Cache._method
    def _hat_run(self, **kws):
        thts = copy.deepcopy(self._thts_tmpl)
        if self.verbose:
            print(u'[HAT~proc]', end="\n" if self.opverbose else "\r")
        self._run()
        self._save_i_results(thts)
        return thts

    def jk_run(self, **kws):
        """ Caching/serializing wrapper of self._jk_run.

        Example
        -------
        >>> o = Metrician(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> run_kwargs = {
        ...     'ER_ks'    : [],
        ...     'AR_ks'    : [],
        ...     'MA_ks'    : [],
        ... }
        >>> jk_results = o.jk_run(**run_kwargs)
        >>> jk_results['par']['mean']
        array([[ 68.63591892],
               [ -1.59876117],
               [ -0.27383054],
               [122.47669909]])

        Let's see what are the "hat" results for comparison purposes
        >>> ht_results = o.hat_run(**run_kwargs)
        >>> ht_results['par']['stack']
        array([[ 68.6189611 ],
               [ -1.59731083],
               [ -0.27393148],
               [122.75291298]])
        """
        self._set_hat_env(**kws)
        _key_ = '_{}-JK'.format(self.model_id)  
        if _key_ not in self._tempo:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self._jk_run(**kws),
                self._tempo
            )
        self._thts_collector(
            key='jk', thts=self._tempo[_key_]
        )
        return self._tempo[_key_]        
    def _jk_run(self, **kws):
        thts = copy.deepcopy(self._thts_tmpl)
        for i in self._up2n_line:
            self._jk_i = i
            if self.verbose:
                print(u'[JK~proc] removed individual {} n° {} over {}'.format(
                    self.geoids[i],
                    i+1,
                    self.n+1
                ), end="\n" if self.opverbose else "\r")
            self._set_jk_i_env(**kws)
            self._run()
            self._save_i_results(thts)
        self._save_results(thts)
        return thts

    def bt_run(self, **kws): 
        r""" Caching/serializing wrapper of self._bt_run.

        Example
        -------
        >>> o = Metrician(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = True,
        ... )
        >>> run_kwargs = {
        ...     'nbsamples': 10000,
        ...     'plot_conv': False,
        ...     'ER_ks'    : [],
        ...     'AR_ks'    : [],
        ...     'MA_ks'    : [],
        ... }

        >>> bt_results = o.bt_run(**run_kwargs)

        Note that if `plot_conv` is set to True (as well as `verbose` -
        set to  True by default -), running `o.bt_run(**run_kwargs)` leads
        the plotting process to be displayed as it goes. I.e:
        >> bt_results = o.bt_run(**run_kwargs)
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][beta0][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][beta{inc}][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][beta{hoval}][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][sigma^2][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][beta0][stdconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[par][beta{inc}][stdconv].png
        < etc...>
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][llik][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][hqc][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][bic][meanconv].png
        < etc...>
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][sh's w][meanconv].png
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][sh's pr(gt.w)][meanconv].png
        < etc...>
        data/columbus.out/er{0}ar{0}ma{0}(10000)[crt][llik][stdconv].png
        < etc...>

        We can now check results related to parameters
        >>> bt_results['par']['std']
        array([[ 4.53939228],
               [ 0.32716959],
               [ 0.10166388],
               [27.77199447]])
        >>> bt_results['par']['mean']
        array([[ 68.64830927],
               [ -1.60017103],
               [ -0.27412991],
               [113.65767514]])

        As well as those related to model-selection criteria
        >>> bt_results['crt']['std']
        array([[ 6.0350291 ],
               [12.07005819],
               [12.07005819],
               [12.07005819],
               [ 0.24632772],
               [ 0.08459525],
               [ 0.08459525],
               [ 0.01696112],
               [ 0.256684  ],
               [ 0.01696112],
               [ 0.256684  ],
               [ 2.82947687],
               [ 0.28237586],
               [ 2.33566169],
               [ 0.27324022]])
        >>> bt_results['crt']['mean']
        array([[-184.75829442],
               [ 377.66985078],
               [ 381.19204973],
               [ 375.51658884],
               [   5.51887397],
               [   0.57746949],
               [   0.57746949],
               [   0.96413782],
               [   0.27278186],
               [   0.96413782],
               [   0.27278186],
               [   2.20490559],
               [   0.48933157],
               [   1.94012218],
               [   0.51922623]])

        NB : the child class Presenter has methods which displays results
        in a more intelligible fashion.
        """
        self._nb_resamples = kws.pop('nbsamples', self._nb_resamples)
        self._set_hat_env(**kws)
        _key_ = '_{}-BT({})'.format(self.model_id, self._nb_resamples)
        if _key_ not in self._tempo:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self._bt_run(**kws),
                self._tempo
            )

        # -- Convergence plots
        bt_ = self._tempo[_key_]
        self._thts_collector(key='bt', thts=bt_)
        if kws.get('plot_conv', True):
            jk_ = self.jk_run(**kws)
            ht_ = self.hat_run(**kws)
            for key, btd in sorted(bt_.items()):
                jkd = jk_[key]
                htd = ht_[key]
                self.conv_charter(key, btd, jkd, htd, m='mean')
                self.conv_charter(key, btd, jkd, m='std')
        return bt_
        
    def _bt_run(self, **kws):
        self.RiSGi0    = copy.copy(self.RiSGi)
        self.ysrihat0  = copy.copy(self.xribetas)
        self.usri0     = copy.copy(self.usri)
        self.levscale0 = copy.copy(self.levscale)
        thts = copy.deepcopy(self._thts_tmpl)
        s    = 0
        nbs  = self._nb_resamples
        nnd  = not self.opverbose
        while s<nbs:
            self._bt_s = s
            if self.verbose:
                if s%10 == 0:
                    print(
                        u'[BT~proc] resampling/seed '
                        u'n° {} over {}({})[{:.3f}%]'.format(
                            s, nbs, nbs - self._nb_resamples,
                            100.*s/nbs
                        ),
                        end="\n" if self.opverbose else "\r"
                    )
            self._set_bt_s_env(**kws)
            self._run()
            if self._maximized_conc_llik_object.success is None:
                nbs += 1        
            else:
                self._save_i_results(thts)            
            s += 1
        self._save_results(thts)         
        return thts

    _rname_stdzer = lambda s,n:(' '.join(n.split()), n.replace(' ',r'\ '))
    def conv_charter(self, key, btd, jkd, htd=None, m='mean', **kws):
        """ 
        """
        bt_  = btd['%s_conv'%m]
        nvar = bt_.shape[0]
        rnames = self._key2rnames[key]        
        for i in range(nvar):

            plt.plot(bt_[i,0,:], color='cornflowerblue')
            
            rname, rname_tx = self._rname_stdzer(rnames[i])

            if m=='std':
                rname_tx = '([{{{}}}]_i - \mu)^2'.format(rname_tx)
            
            self.__figs_plter(self.__BT, btd[m][i,:], rname_tx)
            self.__figs_plter(self.__JK, jkd[m][i,:], rname_tx)
            if htd is not None:
                self.__figs_plter(self.__HT, htd['stack'][i,:], rname_tx)
                
    
            plt.grid()
            x0, x1, y0, y1 = plt.axis()
            plt.axis((
                x0,
                self._nb_resamples,
                y0,
                y1
            ))
            
            self._GaussianMLARIMA__set_legend()
            
            if kws.get('save_fig', True):
                self._GaussianMLARIMA__fig_saver(key, rname, '%s_conv'%m)
            if kws.get('show_fig', False):
                plt.show()
            plt.clf()
        plt.close()

#*****************************************************************************#
##    ╔═╗╦┌┐┌┌┬┐┌─┐┬─┐┬  ┬┌─┐┬  ┌─┐┬─┐
##    ╠═╝║│││ │ ├┤ ├┬┘└┐┌┘├─┤│  ├┤ ├┬┘
##    ╩  ╩┘└┘ ┴ └─┘┴└─ └┘ ┴ ┴┴─┘└─┘┴└─
class PIntervaler(Metrician):

    __PI = 'PI'
    __BC = 'BCa'
    _M_  = Metrician
    __BT = _M_._Metrician__BT
    __HT = _M_._Metrician__HT

    __hist_kwarger = lambda s,x,c:{
        'x'          : x,
        'bins'       : s._nb_resamples/100,
        'histtype'   : 'stepfilled',
        'facecolor'  : c,
        'edgecolor'  : 'none',
        'alpha'      : 0.5,
        'orientation': 'vertical',
    }
    __axvliner = lambda c,l,ls: lambda x,larg:{
        'meth'     : 'axvline',
        'x'        : x,
        'color'    : c,
        'label'    : l%larg,
        'linestyle': ls,
    }
    
    __figs_kwarger = {
        __PI: __axvliner(
            'orange', '$\mathrm{PI \ %s.B}$', 'dotted'
        ),
        __BC: __axvliner(
            'red', '$\mathrm{PI^{BCa} \ %s.B}$', 'dashdot'
        ),
        __BT: __axvliner(
            'black', '$\mathrm{\overline{%s}}$', 'dashed'
        ),
        __HT: __axvliner(
            'black', '$\mathrm{\widehat{%s}}$', 'solid'
        )
    }

    def __figs_plter(self, key, val, larg):
        kws = self.__figs_kwarger[key](
            val, larg
        )
        getattr(plt, kws.pop('meth'))(
            **kws
        )        

    def __init__(self, **kws):
        super(PIntervaler, self).__init__(**kws)

    @staticmethod
    def Z_computer(bt_, ht_, nb_resamples):
        """ Returns the bias-correction value that corrects for median bias
        in the distribution of `ht_` on the standard normal scale.

        Example
        -------
        >>> nb_resamples = 10000
        >>> np.random.seed(0)
        >>> _3d_arr = np.random.random((1, 1, nb_resamples))
        >>> PIntervaler.Z_computer(_3d_arr, .5, nb_resamples)
        (array([[0.01604311]]), array([[0.5064]]))
        """
        trues_abs = np.sum(
            bt_ < ht_, axis=2, dtype=np.float
        )
        trues_rel = trues_abs/nb_resamples
        return sc.stats.norm.ppf(
            trues_rel,
            loc=0, scale=1
        ), trues_rel

    @staticmethod
    def A_computer(jk_dv):
        """ Returns the acceleration constant that corrects for the dependence
        on the parameter of the variance of the tranformed value of `ht_`.
        It is proportional to the skewness of the bootstrap distribution. The
        (simple) method employed below relies the jackknife method.

        Example
        -------
        >>> sample_size = 10000
        >>> np.random.seed(0)
        >>> _3d_arr = np.random.random((1, 1, sample_size))
        >>> PIntervaler.A_computer(_3d_arr)
        array([[0.00217299]])
        """
        return (
            np.sum(pow(jk_dv, 3), axis=2)/6\
          /pow(np.sum(pow(jk_dv, 2.), axis=2) ,1.5)
        )

    @staticmethod
    def A12_computer(Z, A, Z_z):
        """ Returns the values used to locate the end points of the
        corrected percentile confidence interval.

        Example
        -------
        >>> Z = A = 0
        >>> z = 1.959963984540054
        >>> PIntervaler.A12_computer(Z, A, -z)
        0.025
        >>> PIntervaler.A12_computer(Z, A, z)
        0.975

        Example 2
        ---------
        >>> Z = -0.01
        >>> A = -0.05
        >>> z = 1.959963984540054
        >>> PIntervaler.A12_computer(Z, A, Z - z)
        0.014074537684125843
        >>> PIntervaler.A12_computer(Z, A, Z + z)
        0.9613637303516626
        """
        return sc.stats.norm.cdf(
            Z + Z_z/(1. - A*Z_z)
        )

    @staticmethod
    def bounds_computer(bt_, ctile):
        """ Wrapper of numpy's percentile function preset with
        contingented arguments.

        Example
        -------
        >>> np.random.seed(0)
        >>> _3d_arr = np.random.random((1, 1, 5))
        >>> _3d_arr
        array([[[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ]]])

        >>> median_ctile = .5
        >>> PIntervaler.bounds_computer(_3d_arr, median_ctile)
        array([[0.5488135]])
        
        """
        return np.percentile(
            bt_, 1.e2*ctile, axis=2
        )

    def PIs_computer(self, **run_kws):
        """ See Efron and Tibshirani (1993, chap. 14) for BCa.

        Example
        -------
        >>> o = PIntervaler(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     nbsamples = 10000,
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> o.type_I_err = .05
        >>> run_kwargs = {
        ...     'plot_hist': False,
        ...     'plot_conv': False,
        ... }
        >>> results = o.PIs_computer(**run_kwargs)
        >>> parameters_related_results = results['par']
        >>> parameters_related_results['hat']
        array([[ 68.6189611 ],
               [ -1.59731083],
               [ -0.27393148],
               [122.75291298]])
        >>> parameters_related_results['boot_mean']
        array([[ 68.64830927],
               [ -1.60017103],
               [ -0.27412991],
               [113.65767514]])
        >>> parameters_related_results['BCa_lo']
        array([[59.97711055],
               [-2.35708693],
               [-0.4471698 ],
               [76.50712145]])
        >>> parameters_related_results['BCa_up']
        array([[ 77.92911911],
               [ -1.04484449],
               [ -0.02301598],
               [186.09021696]])

        Let's check for distributions' symmetry
        >>> parameters_related_results['Z']
        array([[0.00476261],
               [0.01453896],
               [0.01253347],
               [0.40102733]])

        The last component of the above vector is related to
        the variance of the residuals. As expected, it is far
        from being symmetrical. Actually, it reflects the fact
        that residuals' variance follows a chi-square distrib.
        (as the theory predicts) and thus necessitates a fairly
        large correction factor, i.e. 0.40320149 >> 0.

        What about skewnesses? 
        >>> parameters_related_results['A']
        array([[ 0.01817213],
               [-0.08415481],
               [ 0.07601707],
               [-0.0981961 ]])

        Let's see what are the end points of the corrected
        percentile confidence interval corresponding to a 5%
        type I error. Remember that in a normal-approximation
        case, those are 2.5% and 97.5%.
        >>> parameters_related_results['A1']
        array([[0.02982623],
               [0.0103961 ],
               [0.04611224],
               [0.07497771]])
        >>> parameters_related_results['A2']
        array([[0.97943582],
               [0.95615396],
               [0.99017499],
               [0.98976623]])

        Let's see what are the BCa-underlying standard deviations that
        would prevail in the symmetrical-distribution case.
        >>> parameters_related_results['std']
        array([[ 4.57967817],
               [ 0.33476188],
               [ 0.10820449],
               [27.95538499]])

        while that of the bootstrap-sample are
        >>> parameters_related_results['boot_std']
        array([[ 4.53939228],
               [ 0.32716959],
               [ 0.10166388],
               [27.77199447]])
        """

        plt_h = run_kws.pop('plot_hist', True)

        ht = self.hat_run(**run_kws)
        jk = self.jk_run(**run_kws)
        bt = self.bt_run(**run_kws)

        z  = self.z_bila

        for key in self.to_save:
            htd = ht[key]
            ht_ = htd['hat'] = htd.pop('stack')
            btd = bt[key]
            bt_ = btd['stack']
            htd['Z'], htd['T'] = Z, T = self.Z_computer(
                bt_, ht_[..., np.newaxis],
                self._nb_resamples
            )
            jkd = jk[key]
            htd['A'] = A = self.A_computer(
                jkd['stack'] - jkd['mean'][..., np.newaxis]
            )
            htd['A1'] = A1 = self.A12_computer(Z, A, Z - z)
            htd['A2'] = A2 = self.A12_computer(Z, A, Z + z)

            htd['%s_lo'%self.__BC] = bcloB = self.bounds_computer(
                bt_, A1.flatten()
            ).diagonal().T
            htd['%s_up'%self.__BC] = bcupB = self.bounds_computer(
                bt_, A2.flatten()
            ).diagonal().T

            htd['%s_lo'%self.__PI] = piloB = self.bounds_computer(
                bt_, self.l_bound
            )
            htd['%s_up'%self.__PI] = piupB = self.bounds_computer(
                bt_, self.r_bound
            )

            if plt_h:
                self.hist_charter(
                    key, btd, htd,
                )

            # -- save other stats for ease of reporting
            htd['jack_mean'] = jkd['mean']
            htd['jack_std']  = jkd['std']
            htd['boot_mean'] = btd['mean']
            htd['boot_std']  = btd['std']

            # -- just for free since not grounded theoretically.
            htd['std'] = (
                .5*(ht_ - bcloB) + .5*(bcupB - ht_)
            )/z

            # -- augmentation of the parameters' statistic table.
            if key == 'par':
                _100cl = 1e2*self.clevel
                self._tempo['_stt_names'] = self._stt_names + [
                    '{:.1f}% PI.lo.'.format(_100cl),
                    '{:.1f}% PI.up.'.format(_100cl),  
                    '{:.1f}% BCa.lo.'.format(_100cl),
                    '{:.1f}% BCa.up.'.format(_100cl),                     
                ]
                self._tempo['thetas_stt'] = np.hstack((
                    self.thetas_stt,
                    piloB, piupB,
                    bcloB, bcupB,
                ))
        return ht        

    def hist_charter(self, key, btd, htd, **kws):
        """ Charts bootstrap distribution.
        """
        bt_  = btd['stack']
        nvar = bt_.shape[0]
        cmap = plt.cm.get_cmap(kws.get('colormap', 'autumn') , nvar + 1)
        ht_  = htd['hat']
        bt_mu = btd['mean']
        rnames = self._key2rnames[key]
        
        for i in range(nvar):
            rname, rname_tx = self._rname_stdzer(rnames[i])
            
            # -- histogram
            plt.hist(**self.__hist_kwarger(bt_[i,0,:], cmap(i)))

            # -- Normal-theory based bootstrap interval
            for b in ['lo', 'up']:
                self.__figs_plter(
                    key  = self.__PI,
                    val  = htd['%s_%s'%(self.__PI, b)][i],
                    larg = b
                )
    
            # -- Bias-corrected and accelerated bootstrap interval
            for b in ['lo', 'up']:  
                self.__figs_plter(
                    key  = self.__BC,
                    val  = htd['%s_%s'%(self.__BC, b)][i],
                    larg = b
                )
                
            # -- Bootstrap sample mean
            self.__figs_plter(self.__BT, bt_mu[i], rname_tx) 

            # -- Hat-estimated parameter
            self.__figs_plter(self.__HT, ht_[i], rname_tx) 

            self._GaussianMLARIMA__set_legend()
            # -- save or show
            if kws.get('save_fig', True):
                self._GaussianMLARIMA__fig_saver(key, rname, 'dist')
            if kws.get('show_fig', False):
                plt.show()
            plt.clf()
        plt.close()           

#*****************************************************************************#
##    ╔═╗┬─┐┌─┐┌─┐┌─┐┌┐┌┌┬┐┌─┐┬─┐
##    ╠═╝├┬┘├┤ └─┐├┤ │││ │ ├┤ ├┬┘
##    ╩  ┴└─└─┘└─┘└─┘┘└┘ ┴ └─┘┴└─
class Presenter(PIntervaler):
    def __init__(self, **kws):
        super(Presenter, self).__init__(**kws)

    @staticmethod
    def _dim_renamer(df, **kws):
        r""" Renames dimensions of dataframe-like objects

        Example
        -------
        >>> np.random.seed(0)
        >>> df = pd.DataFrame(np.random.randint(0, 100, size=(2, 4)))
        >>> df
            0   1   2   3
        0  44  47  64  67
        1  67   9  83  21

        >>> slib = Presenter
        >>> slib._dim_renamer(
        ...     df,
        ...     x='VDIM',
        ...     y='HDIM',
        ... )
        HDIM   0   1   2   3
        VDIM                
        0     44  47  64  67
        1     67   9  83  21

        >>> slib._dim_renamer(
        ...     df,
        ...     y='HDIM',
        ... )
        HDIM   0   1   2   3
        0     44  47  64  67
        1     67   9  83  21
        """
        if isinstance(df, pd.core.frame.DataFrame):
            df.index.name   = kws.get('x')
            df.columns.name = kws.get('y')
        else:
            raise NotImplemented
        return df

    @staticmethod
    def _labeler(ylab, arr, rnames, cnames=None):
        r""" Renames dimensions and coordinates of dataframe-like objects

        Example
        -------
        >>> np.random.seed(0)
        >>> arr = np.random.randint(0, 100, size=(2, 4))
        >>> arr
        array([[44, 47, 64, 67],
               [67,  9, 83, 21]])

        >>> slib = Presenter
        >>> slib._labeler(
        ...     'TABLE-NAME', arr,
        ...     rnames = ['r1', 'r2'],
        ...     cnames = ['c1', 'c2', 'c3', 'c4']
        ... )
        \\\\ TABLE-NAME ////  c1  c2  c3  c4
        r1                    44  47  64  67
        r2                    67   9  83  21
        """
        return Presenter._dim_renamer(
            pd.DataFrame(
                arr,
                index   = rnames,
                columns = cnames or rnames
            ),
            y = r'\\\\ {} ////'.format(ylab),
        )

    @Cache._property
    def labelers(self):
        r""" Returns a dict-aggregate of callables that are needed
        to label arrays given their nature.

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> hat_results = o.hat_run(AR_ks=[1], MA_ks=[4, 6])

        >>> labeled_pars = o.labelers['par'](
        ...     'HAT', hat_results
        ... )
        >>> labeled_pars
        \\\\ HAT ////  ER{0}AR{1}MA{4,6}
        \beta_0                56.512740
        \beta_{INC}            -1.627661
        \beta_{HOVAL}          -0.166921
        \rho_{1}                0.207512
        \lambda_{4}             0.498545
        \lambda_{6}            -0.254709
        \sigma^2_{ML}          78.122191

        >>> labeled_criteria = o.labelers['crt'](
        ...     'HAT', hat_results
        ... )
        >>> labeled_criteria
        \\\\ HAT ////         ER{0}AR{1}MA{4,6}
        llik                        -175.498634
        HQC                          367.303791
        BIC                          374.348189
        AIC                          362.997267
        AICg                           5.263378
        pr^2                           0.544721
        pr^2 (pred)                    0.577244
        Sh's W                         0.968770
        Sh's Pr(>|W|)                  0.216237
        Sh's W (pred)                  0.945628
        Sh's Pr(>|W|) (pred)           0.024649
        BP's B                         3.225397
        BP's Pr(>|B|)                  0.199349
        KB's K                         1.892857
        KB's Pr(>|K|)                  0.388125
        """
        key_getter = lambda thts, key:thts[key].get(key, thts[key]['stack'])\
                     if isinstance(thts, dict) else thts
        return {
            'par':lambda ylab, thts, key='stack':self._labeler(
                ylab, key_getter(thts, 'par'),
                self.par_names,
                [self.model_id]
            ),
            'crt':lambda ylab, thts, key='stack':self._labeler(
                ylab, key_getter(thts, 'crt'),
                map(lambda (n,k):n, self._crt_names),
                [self.model_id]
            ),
            'cov':lambda ylab, thts, key='stack':self._labeler(
                ylab, key_getter(thts, 'cov'),
                self.par_names
            ),
            'stt':lambda ylab, thts, key='stack':self._labeler(
                ylab, key_getter(thts, 'stt'),
                self.par_names, self._stt_names
            )
        }

    @Cache._property_tmp
    def table_test(self):
        r""" Returns labeled statistical tables outcoming from
        the currently specified model.

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> _ = o.hat_run(AR_ks=[1], MA_ks=[4, 6])
        >>> o.table_test
        \\\\ STT ////   Estimate  Std. Error  t|z value      Pr(>|t|)      Pr(>|z|)  95.0% CI.lo.  95.0% CI.up.
        \beta_0        56.512740    4.717540  11.979281  2.732522e-15  4.562623e-33     55.191853     57.833627
        \beta_{INC}    -1.627661    0.254482  -6.395976  9.700321e-08  1.595251e-10     -1.698915     -1.556408
        \beta_{HOVAL}  -0.166921    0.069070  -2.416678  1.997937e-02  1.566285e-02     -0.186260     -0.147582
        \rho_{1}        0.207512    0.080001   2.593863  1.292346e-02  9.490420e-03      0.185112      0.229912
        \lambda_{4}     0.498545    0.110704   4.503422  5.051773e-05  6.686776e-06      0.467548      0.529541
        \lambda_{6}    -0.254709    0.123000  -2.070811  4.441309e-02  3.837645e-02     -0.289148     -0.220270
        \sigma^2_{ML}  78.122191   16.160158   4.834247  1.738740e-05  1.336507e-06     73.597430     82.646952
        
        """
        return self.labelers['stt'](
            'STT', self.thetas_stt
        )

    def table_test_of(self, **kws):
        r""" Returns labeled statistical tables outcoming from
        the specified model.

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> o.table_test_of(AR_ks=[1], MA_ks=[4, 6])
        \\\\ STT ////   Estimate  Std. Error  t|z value      Pr(>|t|)      Pr(>|z|)  95.0% CI.lo.  95.0% CI.up.
        \beta_0        56.512740    4.717540  11.979281  2.732522e-15  4.562623e-33     55.191853     57.833627
        \beta_{INC}    -1.627661    0.254482  -6.395976  9.700321e-08  1.595251e-10     -1.698915     -1.556408
        \beta_{HOVAL}  -0.166921    0.069070  -2.416678  1.997937e-02  1.566285e-02     -0.186260     -0.147582
        \rho_{1}        0.207512    0.080001   2.593863  1.292346e-02  9.490420e-03      0.185112      0.229912
        \lambda_{4}     0.498545    0.110704   4.503422  5.051773e-05  6.686776e-06      0.467548      0.529541
        \lambda_{6}    -0.254709    0.123000  -2.070811  4.441309e-02  3.837645e-02     -0.289148     -0.220270
        \sigma^2_{ML}  78.122191   16.160158   4.834247  1.738740e-05  1.336507e-06     73.597430     82.646952
        """
        return self.labelers['stt'](
            'STT', self.thetas_stt_of(**kws)
        )

    @Cache._property_tmp
    def covmat(self):
        r""" Returns labeled parameters-covariance matrices outcoming
        from the currently specified model.

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> _ = o.hat_run(AR_ks=[1], MA_ks=[4, 6])
        >>> o.covmat
        \\\\ COV ////    \beta_0  \beta_{INC}  \beta_{HOVAL}  \rho_{1}  \lambda_{4}  \lambda_{6}  \sigma^2_{ML}
        \beta_0        22.255185    -0.715986       0.019400 -0.284952     0.194901    -0.114581       9.055682
        \beta_{INC}    -0.715986     0.064761      -0.010798  0.006958    -0.013901     0.000230      -0.453305
        \beta_{HOVAL}   0.019400    -0.010798       0.004771 -0.001465     0.003658     0.001065       0.119453
        \rho_{1}       -0.284952     0.006958      -0.001465  0.006400    -0.003668     0.001329      -0.194041
        \lambda_{4}     0.194901    -0.013901       0.003658 -0.003668     0.012255    -0.001044       0.350230
        \lambda_{6}    -0.114581     0.000230       0.001065  0.001329    -0.001044     0.015129       0.042212
        \sigma^2_{ML}   9.055682    -0.453305       0.119453 -0.194041     0.350230     0.042212     261.150703
        
        """
        return self.labelers['cov'](
            'COV', self.thetas_cov
        )

    def covmat_of(self, **kws):
        r""" Returns labeled parameters-covariance matrices outcoming
        from the specified model.

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> o.covmat_of(AR_ks=[1], MA_ks=[4, 6])
        \\\\ COV ////    \beta_0  \beta_{INC}  \beta_{HOVAL}  \rho_{1}  \lambda_{4}  \lambda_{6}  \sigma^2_{ML}
        \beta_0        22.255185    -0.715986       0.019400 -0.284952     0.194901    -0.114581       9.055682
        \beta_{INC}    -0.715986     0.064761      -0.010798  0.006958    -0.013901     0.000230      -0.453305
        \beta_{HOVAL}   0.019400    -0.010798       0.004771 -0.001465     0.003658     0.001065       0.119453
        \rho_{1}       -0.284952     0.006958      -0.001465  0.006400    -0.003668     0.001329      -0.194041
        \lambda_{4}     0.194901    -0.013901       0.003658 -0.003668     0.012255    -0.001044       0.350230
        \lambda_{6}    -0.114581     0.000230       0.001065  0.001329    -0.001044     0.015129       0.042212
        \sigma^2_{ML}   9.055682    -0.453305       0.119453 -0.194041     0.350230     0.042212     261.150703
        """
        return self.labelers['cov'](
            'COV', self.thetas_cov_of(**kws)
        )

    def summary(self, moment='mean'):
        r""" Displays labeled arrays

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = False,
        ... )
        >>> _ = o.hat_run()
        >>> _ = o.hat_run(AR_ks=[1])
        >>> _ = o.hat_run(AR_ks=[1], MA_ks=[4, 6])
        >>> o.summary(moment='mean')
        ================================= PARS
        \\\\ HAT ////  ER{0}AR{0}MA{0}  ER{0}AR{1}MA{0}  ER{0}AR{1}MA{4,6}
        \beta_0              68.618961        49.374308          56.512740
        \beta_{HOVAL}        -0.273931        -0.270035          -0.166921
        \beta_{INC}          -1.597311        -1.159673          -1.627661
        \lambda_{4}                NaN              NaN           0.498545
        \lambda_{6}                NaN              NaN          -0.254709
        \rho_{1}                   NaN         0.320152           0.207512
        \sigma^2_{ML}       122.752913        93.805430          78.122191
        ================================= CRTS
        \\\\ HAT ////         ER{0}AR{0}MA{0}  ER{0}AR{1}MA{0}  ER{0}AR{1}MA{4,6}
        llik                      -187.377239      -182.193752        -175.498634
        HQC                        382.907740       375.258520         367.303791
        BIC                        386.429939       379.954785         374.348189
        AIC                        380.754478       372.387504         362.997267
        AICg                         5.625770         5.455015           5.263378
        pr^2                         0.552404         0.548625           0.544721
        pr^2 (pred)                  0.552404         0.588923           0.577244
        Sh's W                       0.977076         0.966407           0.968770
        Sh's Pr(>|W|)                0.449724         0.173524           0.216237
        Sh's W (pred)                0.977076         0.978610           0.945628
        Sh's Pr(>|W|) (pred)         0.449724         0.508454           0.024649
        BP's B                       7.900442        15.168577           3.225397
        BP's Pr(>|B|)                0.019250         0.000508           0.199349
        KB's K                       5.694088         8.368048           1.892857
        KB's Pr(>|K|)                0.058016         0.015237           0.388125
        """
        for obj in self.to_save:
            print(33*"=", '%sS'%obj.upper())

            for key, ided_models in self._thts_collection.items():
                labeled_objs = []

                ided_models_items = sorted(
                    ided_models.items(),
                    key=lambda item:item[0]
                )

                for model_id, thts in ided_models_items:
                    self._tempo['model_id']  = model_id
                    self._tempo['par_names'] = self._mid2pnames[model_id]
                    labeled_objs.append(self.labelers[obj](
                        key.upper(), thts, moment
                    ))

                print(pd.concat(labeled_objs, axis=1))

if name_eq_main:
    npopts(8, True, 5e4)
    pdopts(5e4)
    import doctest
    doctest.testmod(verbose=False)
