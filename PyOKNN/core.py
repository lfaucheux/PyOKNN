#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import print_function

__authors__ = [
    "laurent.faucheux@hotmail.fr",
]

__version__ = '0.1.6'

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

npopts = lambda p,b,w:np.set_printoptions(
    precision=p, suppress=b, linewidth=w,
)
pdopts = lambda w:pd.set_option(
    'display.width', int(w)
)

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
    def _args_hash(args, kwargs):
        hh = ''
        for a in args:
            h = UniHasher.hash_(a)
            hh += h
        for kw, a in sorted(kwargs.items()):
            hh += UniHasher.hash_(a)
        return _hash_(hh)


#*****************************************************************************#
##    ┌─┐┌─┐┌─┐┬ ┬┌─┐
##    │  ├─┤│  ├─┤├┤ 
##    └─┘┴ ┴└─┘┴ ┴└─┘
class Cache(UniHasher):
    def __init__(self):
        """
        Homemade cache class which aims at being inherited

        Example
        -------
        >>> import random
        >>> class_ = type(
        ...     'class_', 
        ...     (Cache, ), 
        ...     {'attr':Cache._property(lambda _ :random.random())}, 
        ... )
        >>> instance = class_()
        >>> instance.attr == instance.attr
        True
        """
        self._cache = {}
        self._tempo = {} 

    @classmethod
    def _property(cls, meth):
        @property
        @ft.wraps(meth)
        def __property(cls):
            meth_name = meth.__name__
            if meth_name not in cls._cache:
                cls._cache[meth_name] = meth(cls)
            return cls._cache[meth_name]
        return __property

    @classmethod
    def _property_tmp(cls, meth):
        @property
        @ft.wraps(meth)
        def __property(cls):
            meth_name = meth.__name__
            if meth_name not in cls._tempo:
                cls._tempo[meth_name] = meth(cls)
            return cls._tempo[meth_name]
        return __property

    @classmethod
    def _method(cls, meth):
        @ft.wraps(meth)
        def __method(cls, *args, **kwargs):
            meth_name = '{}_{}'.format(
                meth.__name__,
                cls._args_hash(args, kwargs)
            )
            if meth_name not in cls._cache:
                cls._cache[meth_name] = meth(cls, *args, **kwargs)
            return cls._cache[meth_name]
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
	
    def __init__(self, **kwargs):
        super(Serializer, self).__init__()
        self._modes = {
            'hickle':{'r':'r', 'w':'w'}, 
            'pickle':{'r':'rb', 'w':'wb'}, 
        }[ckl.__name__]
        self._ext  = kwargs.get('ext', 'ckl')
        self._sdir = kwargs.get('sdir')
        self._upd  = kwargs.get('update') or False

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
               if (self._ckle_exist(_key_) and _key_ not in [])\
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
    def __init__(self, **kwargs): self.ext = kwargs.get('ext')

    def _file_not_found(self, name):
        message = ['',
            ' {}.%s does not exist in the working directory.'%self.ext,
            ' Either the file has been renamed or removed.',
            ' This may concern other files as well.'
        ]
        raise IOError(errno.ENOENT, '\n\t'.join(message).format(name))
    def _file_not_unique(self, dirs, **kwargs):
        enum_dirs = map(lambda(i,d):'{0} - {1}'.format(i+1,d), enumerate(dirs))
        message = ['',
            '{name}.%s is not unique.'%self.ext,
            '{len_} files found.',
            '\n\t\t'.join(['']+enum_dirs)
        ]
        print(IOError('%s2'%self.ext, '\n\t'.join(message).format(**kwargs)))
        ix = raw_input('\tType the index of the one your want to work with:')
        return int(ix) - 1
    def _unknown_error(self, **kwargs):        
        message = ['',
            " Something unexpected happened with {name}.%s"%self.ext,
            ' "{exc}"',         
        ]
        raise IOError('%s3'%self.ext, '\n\t'.join(message).format(**kwargs))

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

    def _get_dir_of(self, name, ext, **kwargs):
        """ Methode which gets the targeted file by its name
        and extension.    
        """
        dirs = getattr(self, '_{}_files'.format(ext)).get(name, [])
        len_ = len(dirs)
        file_ix = self._from_id_to_index(dirs, kwargs.get('file_id'))
        if not len_:
            FileError(ext=ext)._file_not_found(name)            
        elif len_ > 1 and kwargs.get('file_id') is None:
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
    def _get_dbf_dir(self, name, **kwargs):
        return self._get_dir_of(
            name, 'dbf', **kwargs
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
    def _get_shp_dir(self, name, **kwargs):
        return self._get_dir_of(
            name, 'shp', **kwargs
        )
        

#*****************************************************************************#
##    ╔╦╗┌─┐┌┬┐┌─┐╔═╗┌┐  ┬┌─┐┌─┐┌┬┐
##     ║║├─┤ │ ├─┤║ ║├┴┐ │├┤ │   │ 
##    ═╩╝┴ ┴ ┴ ┴ ┴╚═╝└─┘└┘└─┘└─┘ ┴ 
class DataObject(DataGetter):

    def __init__(self, data_name, y_name, x_names, **kwargs):
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
        self.id_name  = kwargs.get('id_name')
        self.srid     = kwargs.get('srid')

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
    def points_array(self):
        """ Data-object rendered from a PySAL-opened shp-file source."""
        return ps.weights.util.get_points_array_from_shapefile(
            self.shp_fname
        )

    @Cache._property
    def geoids(self):
        """ Names of spatial units."""
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
        
    def __init__(self, **kwargs):
        super(SpDataObject, self).__init__(**kwargs)
        self._up2n_line = np.arange(self.n)
        self.__ER_ks = kwargs.get('ER_ks', [])  
        self.__AR_ks = kwargs.get('AR_ks', []) 
        self.__MA_ks = kwargs.get('MA_ks', [])

    def from_scratch(self, **kws):
        """ Cleans cache from all objects that are not permanent
        relatively to a given dataset. One consequence of this is that
        re-executing any instance's methode from scrath returns ols
        like results.
        """
        self._tempo.clear()
        self.__ER_ks = kws.get('ER_ks', [])  
        self.__AR_ks = kws.get('AR_ks', []) 
        self.__MA_ks = kws.get('MA_ks', [])
        

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
    def psw_from_array(arr, **kwargs):
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
        tmp_ids   = kwargs.get('tmp_ids', np.arange(n))
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
        if 'ids' in kwargs:
            psw.remap_ids(kwargs['ids'])
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
        >>> hash_ref = 'bc3JZoBk99nnYkC43dgn0Q==\n'
        >>> hash_chk = o.hash_(
        ...     _w_collection
        ... )
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
        >>> hash_ref = 'bc3JZoBk99nnYkC43dgn0Q==\n'
        >>> hash_chk = o.hash_(
        ...     w_collection
        ... )
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
        return filter(lambda k:k>0, sorted(self.__ER_ks))
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
        return filter(lambda k:k>0, sorted(self.__AR_ks))
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
        return filter(lambda k:k>0, sorted(self.__MA_ks))
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

    def __init__(self, **kwargs):
        super(GaussianMLARIMA, self).__init__(**kwargs)
        self._tolerance = kwargs.get('tolerance', 1.e-8)
        self._nd_order  = kwargs.get('nd_order', 3.)
        self._nd_step   = kwargs.get('nd_step', 1.e-6)
        self._nd_method = kwargs.get('nd_method', 'central')
        self._verbose   = kwargs.get('verbose', True)
        self._opverbose = kwargs.get('opverbose', True)
        self._thts_collection = {}

    @property
    def default_thts(self):
        return {
            'par':{'stack':self.thetas},
            'crt':{'stack':self.thetas_crt},
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
            r'\sigma^2'
        ]
        

    @property
    def initial_guesses_as_list(self): return self.p*[0.]

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
    def betas(self): return np.dot(self.xritxrii, self.xritysri)
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
        """ (self.ysritysri - np.dot(self.xritysri.T, self.betas))/self.n """
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
        if self._verbose:
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

    def hull_charter(self, u, save_fig=True):
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
        
        if save_fig:
            self.__fig_saver('RESID', 'HULLS', nbs_dep=False)
        else:
            plt.show()            
        plt.clf()
        plt.close()
        

    @Cache._property_tmp
    def llik_maximizing_coeffs_as_list(self):
        r""" \hat{\theta}_{\mathrm{ML}} """
        return self._maximized_conc_llik_object.x

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
        ('llik'                 , 'llik'),
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
##        ("Wh's w"               , 'white'),
##        ("Wh's Pr(>|w|)"        , None),
    ]   
    @Cache._property_tmp
    def llik(self):
        """ Shorcut for `self.full_llik`.

        Chk
        ---
        >>> o = GaussianMLARIMA(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ... )
        >>> o.llik
        array([[-187.37723881]])
        >>> o.full_llik
        array([[-187.4251219]])
        """
        return self.conc_llik - .5*self.n * ( np.log(2.*np.pi) + 1.)
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
##    @Cache._property_tmp
##    def white(self):
##        return self._dict2arr_frmt(ps.spreg.diagnostics.white(
##            self._ps_testable_inst
##        ), 'wh')

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
            self.sig2n_k,
        ])

    @Cache._property_tmp
    def thetas_as_list(self):
        return self.thetas.flatten().tolist()

    def _full_llik_computer(self, _thetas):
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
        return np.log(jacobian)\
               - .5*self.n*np.log(2.*np.pi*variance)\
               - rss/(2.*variance)

    @Cache._property_tmp
    def full_llik(self):
        return self._full_llik_computer(
            self.thetas_as_list
        )    

    @Cache._property_tmp
    def full_llik_hessian_computer(self):
        r"""
        Callable returning hessian matrices, which consist of second
        derivatives of the likelihood function with respect to the
        parameters. The Hessian is defined as:
        $$
        \mathbf{H}(\theta)=
            \frac{\partial^{2}}{\partial\theta_{i}\partial\theta_{j}}
            l(\theta),
            ~~~~ 1\leq i, j\leq p
        $$
        """
        return nd.Hessian(
            self._full_llik_computer,
##            step = nd.MinStepGenerator(
##                base_step=self._nd_step,
##                step_ratio=None,
##                num_extrap=0
##            ),
##            order  = self._nd_order,
##            method = self._nd_method
        )

    @Cache._property_tmp
    def obs_fisher_matrix(self):
        r"""
        The so-called observed information matrix, which consists of the
        Fisher information matrix, $\mathbf{I}(\theta)$, evaluated at the
        maximum likelihood estimates (MLE), i.e.
        $$
        \mathbf{I}(\hat{\theta}_{\mathrm{ML}})=
            -\mathbf{H}(\hat{\theta}_{\mathrm{ML}})
        $$
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
        """
        return np.linalg.inv(
            self.obs_fisher_matrix
        )
    @property
    def thetas_cov(self):
        """ Alias for self.obs_cov_matrix."""
        return self.obs_cov_matrix
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

    _stt_names = [
        'Estimate',
        'Std. Error',
        't|z value',
        'Pr(>|t|)',
        'Pr(>|z|)'
    ]
    @Cache._property_tmp
    def thetas_stt(self):
        return np.hstack([
            self.thetas,
            self.thetas_se,
            self.thetas_tt,
        ])

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
        x0 = self.initial_guesses_as_list
        try:
            return sc.optimize.minimize(
                x0      = x0,
                fun     = lambda xN : -self._conc_llik_computer(xN),
                #tol     = self._tolerance,
                method  = 'Nelder-mead',
                options = {
                    'disp' : self.p and self._opverbose,
                },
            )
        except Exception as exc:
            print(exc.message)
            return type(
                'optimerr',
                (object,),
                {'success':False, 'x':x0}
            )

    def _conc_llik_computer(self, _gammas_rhos_lambdas):
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
        _sig2         = np.dot(_r.T, _r)/n
        _RiSG         = np.dot(_Ri, np.dot(_S, _G))
        _jac          = np.linalg.det(_RiSG)
        _cgll         = self._conc_llik_core_computer(
            jacobian = _jac,
            variance = _sig2
        )
        return _cgll

    def _conc_llik_core_computer(self, jacobian, variance):
        return np.log(jacobian)\
               - .5*self.n*np.log(variance)

    def _yNx_filterer(self, gammas_rhos_lambdas):
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
## https://en.wikipedia.org/wiki/Correlogram#Statistical_inference_with_correlograms
##    ═╗ ╦╔═╗╔═╗╔═╗╔╦╗┌─┐┬┌─┌─┐┬─┐
##    ╔╩╦╝╠═╣║  ╠╣ ║║║├─┤├┴┐├┤ ├┬┘
##    ╩ ╚═╩ ╩╚═╝╚  ╩ ╩┴ ┴┴ ┴└─┘┴└─
class XACFMaker(GaussianMLARIMA):

    type_I_err = .05    
    clevel = 1 - type_I_err
    
    z_unil  = sc.stats.norm.ppf(clevel, loc=0, scale=1)    
    l_bound = type_I_err/2.
    r_bound = 1. - l_bound
    z_bila  = sc.stats.norm.ppf(r_bound, loc=0, scale=1)

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
    
    def __init__(self, **kwargs):
        super(XACFMaker, self).__init__(**kwargs)
        self._k_domain = self._up2n_line[:, None]
        self._k_domain_aslist  = self._up2n_line.tolist()
        self._empty_arr = np.zeros_like(self._k_domain).astype(np.float64)

    @Cache._method
    def Bartlett(self, Rs):
        r""" Bartlett's significance threshold [3]_.

        Chk
        ---
        >>> hash_ref = 'ec6fYBPx1EXxT/HuaaM1TA==\n'
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> o.z_unil = 1.6448536269514722
        >>> np.random.seed(0)
        >>> Rs = np.random.normal(0, 1, size=o.n)[:, None]
        >>> v  = o.Bartlett(Rs)
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
        >>> hash_ref = 'l7gWSnn7Vhohx+PzGIOaiA==\n'
        >>> o = XACFMaker(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = [],
        ... )
        >>> o.z_unil = 1.6448536269514722
        >>> v  = o.Quenouille
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
        >>> hashs_ref = {
        ...     'u': 'u2mcxvWwqQN52zLVemvoqA==\n',
        ...     'PACF': (
        ...         ('Qs', 'l7gWSnn7Vhohx+PzGIOaiA==\n'),
        ...         ('Rs', 'Ney8wPS2VpYjGc80KXS2rQ==\n')
        ...     ),
        ...     'ACF': (
        ...         ('Bs', 'GUx0lAGhCu2BVBNr+BjAsg==\n'),
        ...         ('Gs', 'Txc5idb4KyG7KSk0HjTb7w==\n'),
        ...         ('Ms', 'LKuiqU62uEXCOdXKvwtNMg==\n'),
        ...         ('Rs', '72cHWJadnMEbcQClKWHWdA==\n')
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

    def XACF_charter(self, save_fig=False, **kwargs):
        """ Plots both ACFs and PACFs following formats defined
        via self._figs_kwarger.      
        """
        if 'u' in kwargs:
            stats = self.XACF_computer(kwargs['u'])
        elif 'stats' in kwargs:
            stats = kwargs['stats']
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
        
        if save_fig:
            self._GaussianMLARIMA__fig_saver('RESID', '(P)ACF', nbs_dep=False)
        else:
            plt.show()

        plt.clf()
        plt.close()

#*****************************************************************************#
##    ╔═╗┌─┐┌┬┐┌─┐┬  ┌─┐┬─┐
##    ╚═╗├─┤│││├─┘│  ├┤ ├┬┘
##    ╚═╝┴ ┴┴ ┴┴  ┴─┘└─┘┴└─
class Sampler(XACFMaker):
    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)  
        self.Idn0     = copy.copy(self.Idn)
        self.y0       = copy.copy(self.y)
        self.x0       = copy.copy(self.x)
        self.n0       = copy.copy(self.n)
        self.RiSGi0   = copy.copy(self.RiSGi)
        self.ysrihat0 = copy.copy(self.xribetas)
        self.usri0    = copy.copy(self.usri)
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

    def _set_jk_env(self, **kws):
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

    def _set_bt_env(self, **kws):
        self.from_scratch(**kws)
        _c, _t  = self._cache, self._tempo  
        ixs     = self.get_bt_ixs()
        _c['y'] = np.dot(
            self.RiSGi0,
            self.ysrihat0 + self._r_rs_standardizer(
                self.usri0[ixs],
            )
        )

    @Cache._property
    def grandi_vector(self):
        """ Stores (once for all) and returns an exponentiated vector of -1,
        as done in Grandi's series.
        """
        return np.power(-self.ones, self._up2n_line[:, None])

    def _r_rs_standardizer(self, r_rs):         
        n          = self.n
        _unbiaser  = np.sqrt(n/(n - 1))
        _av_scaler = -np.sum(r_rs*self.levscale)/n
        return _unbiaser*(r_rs*self.levscale + _av_scaler)*self.grandi_vector

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
        kwargs = self.__figs_kwarger[key](
            val, larg
        )
        getattr(plt, kwargs.pop('meth'))(
            **kwargs
        )

    def __init__(self, **kwargs):
        super(Metrician, self).__init__(**kwargs)
        self._nb_resamples = int(kwargs.get('nbsamples', 2000))
        self.to_save = [
            'par', 'crt',
##          'cov', 'stt', #[!!!] Too time-consuming. stt needs cov's diagonal.
        ]

    @Cache._property_tmp
    def _key2rnames(self):
        return {
            'par':self.par_names,
            'crt':map(lambda (n,k):n, self._crt_names),
            'cov':self.par_names,
            'stt':self.par_names,
        }

    def _run(self):
        self.thetas

    @Cache._property_tmp
    def _thts_tmpl(self):
        """ `self.to_save`-dependent template for saving
        results.
        """
        return {
            key:{'stack': None}
            for key in self.to_save
        }
    def _save_i_results(self, thts):
        objs_to_save = [
            (key, getattr(self, 'thetas_%s'%key, self.thetas))
            for key in self.to_save            
        ]
        for key, obj in objs_to_save:
            if thts[key]['stack'] is None:
                thts[key]['stack']     = obj
                thts[key]['mean_conv'] = obj
                thts[key]['std_conv']  = np.full_like(obj, np.nan)
            else:
                stack = thts[key]['stack'] = np.dstack((
                    thts[key]['stack'], obj
                ))
                thts[key]['mean_conv'] = np.dstack((
                    thts[key]['mean_conv'],
                    np.nanmean(
                        stack, axis=2, keepdims=True
                    )
                ))
                thts[key]['std_conv'] = np.dstack((
                    thts[key]['std_conv'],
                    np.nanstd(
                        stack, axis=2, keepdims=True, ddof=1
                    )
                ))
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
        """ Caching/serializing wrapper of self._hat_run.

        Example
        -------
        >>> o = Metrician(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
        ...     verbose   = False,
        ...     opverbose = True,
        ... )

        Let's first make an example with no spatial parameters.
        >>> run_kwargs = {
        ...     'ER_ks'    : [],
        ...     'AR_ks'    : [],
        ...     'MA_ks'    : [],
        ... }
        >>> hat_results = o.hat_run(**run_kwargs)
        >>> o.thetas_cov
        array([[ 22.42482892,  -0.94235135,  -0.16156749,   0.        ],
               [ -0.94235135,   0.11164337,  -0.01723674,   0.        ],
               [ -0.16156749,  -0.01723674,   0.01064997,  -0.        ],
               [  0.        ,   0.        ,  -0.        , 795.24628791]])

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
        Optimization terminated successfully.
                 Current function value: 111.656163
                 Iterations: 138
                 Function evaluations: 254
        array([[ 54.4809296 ,  -1.84502206,   0.00342826,   0.12955141,  -0.81519065,   0.62122773,  28.33119842],
               [ -1.84502206,   0.12774818,  -0.01942022,   0.00683215,   0.0240067 ,  -0.03171117,  -0.89739894],
               [  0.00342826,  -0.01942022,   0.01014639,   0.0002697 ,  -0.00385608,   0.01133015,   0.11799748],
               [  0.12955141,   0.00683215,   0.0002697 ,   0.02038757,  -0.00723236,   0.0013995 ,   0.09174917],
               [ -0.81519065,   0.0240067 ,  -0.00385608,  -0.00723236,   0.0195459 ,  -0.01995175,  -0.63160178],
               [  0.62122773,  -0.03171117,   0.01133015,   0.0013995 ,  -0.01995175,   0.05767519,   0.61264511],
               [ 28.33119842,  -0.89739894,   0.11799748,   0.09174917,  -0.63160178,   0.61264511, 484.41122657]])

        NB: Covariance matrices (components) are not subject to bootstrapping
         for performance reasons. This implies computing `self.obs_fisher_matrix`
         `self._nb_resamples` times. However, this can easily be turned on by
         uncommenting the elements `'cov'` and `'stt'` of the list-attribute
         `self.to_save`.
        """
        self._set_hat_env(**kws)
        _key_ = '_{}-HAT'.format(self.model_id)  
        if _key_ not in self._tempo:
            self._szer._may_do_and_dump(
                _key_,
                lambda:self._hat_run(**kws),
                self._tempo
            )
        self._thts_collector(
            key='hat', thts=self._tempo[_key_]
        )
        return self._tempo[_key_]
    def _hat_run(self, **kws):
        thts = copy.deepcopy(self._thts_tmpl)
        if self._verbose:
            print(u'[HAT~proc]')
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
        ... )
        >>> run_kwargs = {
        ...     'verbose'  : False,
        ...     'ER_ks'    : [],
        ...     'AR_ks'    : [],
        ...     'MA_ks'    : [],
        ... }
        >>> jk_results = o.jk_run(**run_kwargs)
        >>> jk_results['par']['mean']
        array([[ 68.63591892],
               [ -1.59876117],
               [ -0.27383054],
               [130.64181237]])

        Let's see what are the "hat" results for comparison purposes
        >>> ht_results = o.hat_run(**run_kwargs)
        >>> ht_results['par']['stack']
        array([[ 68.6189611 ],
               [ -1.59731083],
               [ -0.27393148],
               [130.75853773]])
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
            if self._verbose:
                print(u'[JK~proc] removed individual {} n° {} over {}'.format(
                    self.geoids[i],
                    i+1,
                    self.n+1
                ), end="\r")
            self._set_jk_env(**kws)
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

        We can now check the parameters related results
        >>> bt_results['par']['std']
        array([[ 4.53918732],
               [ 0.32716949],
               [ 0.10166229],
               [29.58291632]])
        >>> bt_results['par']['mean']
        array([[ 68.64873766],
               [ -1.60016822],
               [ -0.27413558],
               [121.07273943]])

        As well as those related to model-selection crteria
        >>> bt_results['crt']['std']
        array([[ 6.0350291 ],
               [13.97593616],
               [13.97593616],
               [13.97593616],
               [ 0.28522319],
               [ 0.0869042 ],
               [ 0.0869042 ],
               [ 0.49132822],
               [ 0.50000291],
               [ 0.48874503],
               [ 0.50002499],
               [ 2.83002401],
               [ 0.28241245],
               [ 2.33592783],
               [ 0.27327164]])
        >>> bt_results['crt']['mean']
        array([[-184.75829442],
               [ 387.18638126],
               [ 390.7085802 ],
               [ 385.03311931],
               [   5.71308888],
               [   0.55807297],
               [   0.55807297],
               [   0.50435712],
               [   0.4953    ],
               [   0.51187232],
               [   0.5001    ],
               [   2.20559647],
               [   0.48927259],
               [   1.94062635],
               [   0.51915986]])

        NB : the forthcoming Presenter class has methods which displays results
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

        # -- Plot convergences
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
        thts = copy.deepcopy(self._thts_tmpl)
        s    = 0
        nbs  = self._nb_resamples
        while s<nbs:
            self._bt_s = s
            if kws.get('verbose', False):
                if self._verbose:
                    if s%10 == 0:
                        print(
                            u'[BT~proc] resampling/seed '
                            u'n° {} over {}({})'.format(
                                s, self._nb_resamples, nbs
                            ),
                            end="\r"
                        )
            self._set_bt_env(**kws)
            if not self._maximized_conc_llik_object.success:
                nbs += 1    
            else:
                self._save_i_results(thts)            
            s += 1
        self._save_results(thts)            
        return thts

    _rname_stdzer = lambda s,n:(' '.join(n.split()), n.replace(' ',r'\ '))
    def conv_charter(self, key, btd, jkd, htd=None, m='mean', save_fig=True):
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
            
            if save_fig:
                self._GaussianMLARIMA__fig_saver(key, rname, '%s_conv'%m)
            else:
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
        kwargs = self.__figs_kwarger[key](
            val, larg
        )
        getattr(plt, kwargs.pop('meth'))(
            **kwargs
        )        

    def __init__(self, **kwargs):
        super(PIntervaler, self).__init__(**kwargs)

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
        on the parameter of the variance of the tranformed value of `ht_`. It
        relates to the skewness of the sampling distribution of the estimator
        of the quantity of interest. The (simple) method employed below relies
        the jackknife method.

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
        ... )
        >>> o.type_I_err = .05
        >>> run_kwargs = {
        ...     'plot_hist': False,
        ...     'plot_conv': False,
        ...     'verbose'  : True,
        ... }
        >>> results = o.PIs_computer(**run_kwargs)
        >>> parameters_related_results = results['par']
        >>> parameters_related_results['hat']
        array([[ 68.6189611 ],
               [ -1.59731083],
               [ -0.27393148],
               [130.75853773]])
        >>> parameters_related_results['boot_mean']
        array([[ 68.64873766],
               [ -1.60016822],
               [ -0.27413558],
               [121.07273943]])
        >>> parameters_related_results['BCa_lo']
        array([[59.97660869],
               [-2.35734682],
               [-0.4471698 ],
               [81.4898608 ]])
        >>> parameters_related_results['BCa_up']
        array([[ 77.92756627],
               [ -1.04499783],
               [ -0.02301598],
               [198.21210783]])

        Let's check for distributions' symmetry
        >>> parameters_related_results['Z']
        array([[0.00451195],
               [0.01428827],
               [0.01253347],
               [0.4007557 ]])

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
        array([[0.02979344],
               [0.01037932],
               [0.04611224],
               [0.07488572]])
        >>> parameters_related_results['A2']
        array([[0.97941001],
               [0.95611356],
               [0.99017499],
               [0.98975397]])

        Let's see what are the BCa-underlying standard deviations that
        would prevail in the symmetrical-distribution case.
        >>> parameters_related_results['std']
        array([[ 4.57941006],
               [ 0.33478906],
               [ 0.10820449],
               [29.77663058]])

        while that of the bootstrap-sample are
        >>> parameters_related_results['boot_std']
        array([[ 4.53918732],
               [ 0.32716949],
               [ 0.10166229],
               [29.58291632]])
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

            htd['%s_lo'%self.__BC] = loB = self.bounds_computer(
                bt_, A1.flatten()
            ).diagonal().T
            htd['%s_up'%self.__BC] = upB = self.bounds_computer(
                bt_, A2.flatten()
            ).diagonal().T

            htd['%s_lo'%self.__PI] = self.bounds_computer(
                bt_, self.l_bound
            )
            htd['%s_up'%self.__PI] = self.bounds_computer(
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
                .5*(ht_ - loB) + .5*(upB - ht_)
            )/z
        return ht        

    def hist_charter(self, key, btd, htd, save_fig=True, **kws):
        """ 
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
            if save_fig:
                self._GaussianMLARIMA__fig_saver(key, rname, 'dist')
            else:
                plt.show()
            plt.clf()
        plt.close()           

#*****************************************************************************#
##    ╔═╗┬─┐┌─┐┌─┐┌─┐┌┐┌┌┬┐┌─┐┬─┐
##    ╠═╝├┬┘├┤ └─┐├┤ │││ │ ├┤ ├┬┘
##    ╩  ┴└─└─┘└─┘└─┘┘└┘ ┴ └─┘┴└─
class Presenter(PIntervaler):
    def __init__(self, **kwargs):
        super(Presenter, self).__init__(**kwargs)

    @staticmethod
    def _dim_renamer(o, **kws):
        if isinstance(o, pd.core.frame.DataFrame):
            o.index.name   = kws.get('x')
            o.columns.name = r'\\\\ {y} ////'.format(**kws)
        else:
            raise NotImplemented
        return o

    @staticmethod
    def __labeler(ylab, arr, rnames, cnames=None):
        return Presenter._dim_renamer(
            o = pd.DataFrame(
                arr,
                index   = rnames,
                columns = cnames or rnames
            ),
            y = ylab,
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
        \sigma^2               85.066386

        >>> labeled_criteria = o.labelers['crt'](
        ...     'HAT', hat_results
        ... )
        >>> labeled_criteria
        \\\\ HAT ////         ER{0}AR{1}MA{4,6}
        llik                      -1.754986e+02
        HQC                        3.674765e+02
        BIC                        3.745209e+02
        AIC                        3.631700e+02
        AICg                       5.266903e+00
        pr^2                       5.447215e-01
        pr^2 (pred)                5.772440e-01
        Sh's W                     1.430368e-02
        Sh's Pr(>|W|)              2.010151e-16
        Sh's W (pred)              1.000000e+00
        Sh's Pr(>|W|) (pred)       1.000000e+00
        BP's B                     3.225397e+00
        BP's Pr(>|B|)              1.993489e-01
        KB's K                     1.892857e+00
        KB's Pr(>|K|)              3.881247e-01
        """
        return {
            'par':lambda ylab, thts, key='stack':self.__labeler(
                ylab, thts['par'].get(key, thts['par']['stack']),
                self.par_names,
                [self.model_id]
            ),
            'crt':lambda ylab, thts, key='stack':self.__labeler(
                ylab, thts['crt'].get(key, thts['crt']['stack']),
                map(lambda (n,k):n, self._crt_names),
                [self.model_id]
            ),
            'cov':lambda ylab, thts, key='stack':self.__labeler(
                ylab, thts['cov'].get(key, thts['cov']['stack']),
                self.par_names
            ),
            'stt':lambda ylab, thts, key='stack':self.__labeler(
                ylab, thts['stt'].get(key, thts['stt']['stack']),
                self.par_names, self._stt_names
            )
        }

    def summary(self, moment='mean'):
        r""" Displays labeled arrays

        Example
        -------
        >>> o = Presenter(
        ...     data_name = 'columbus',
        ...     y_name    = 'CRIME',
        ...     x_names   = ['INC', 'HOVAL'],
        ...     id_name   = 'POLYID',
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
        \sigma^2            130.758538       102.143690          85.066386
        ================================= CRTS
        \\\\ HAT ////         ER{0}AR{0}MA{0}  ER{0}AR{1}MA{0}  ER{0}AR{1}MA{4,6}
        llik                      -187.377239      -182.193752      -1.754986e+02
        HQC                        383.003506       375.431252       3.674765e+02
        BIC                        386.525705       380.127517       3.745209e+02
        AIC                        380.850244       372.560236       3.631700e+02
        AICg                         5.627724         5.458540       5.266903e+00
        pr^2                         0.552404         0.548625       5.447215e-01
        pr^2 (pred)                  0.552404         0.588923       5.772440e-01
        Sh's W                       1.000000         1.000000       1.430368e-02
        Sh's Pr(>|W|)                1.000000         1.000000       2.010151e-16
        Sh's W (pred)                1.000000         1.000000       1.000000e+00
        Sh's Pr(>|W|) (pred)         1.000000         1.000000       1.000000e+00
        BP's B                       7.900442        15.168577       3.225397e+00
        BP's Pr(>|B|)                0.019250         0.000508       1.993489e-01
        KB's K                       5.694088         8.368048       1.892857e+00
        KB's Pr(>|K|)                0.058016         0.015237       3.881247e-01
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

if __name__ == '__main__':
    npopts(8, True, 5e4)
    pdopts(5e4)
    import doctest
    doctest.testmod(verbose=False)
