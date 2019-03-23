# -*- coding: utf8 -*-
from __future__ import absolute_import
from setuptools import setup
import pathlib as pa
import codecs as cd

package_version = '0.1.65'
package_name    = 'PyOKNN'

with pa.Path('requirements.txt').open() as requirements:
    requires = [l.strip() for l in requirements]

with cd.open('README.md', encoding='utf-8') as readme_f:
    readme = readme_f.read()

setup(
    license      = 'MIT',
    name         = package_name,
    version      = package_version,
    packages     = [package_name],
    package_data = {
        package_name: [
            'examples/*',
            'data/tests/*',
            'data/tests/subfolder/*',
            'data/COLUMBUS/*',
            'data/COLUMBUS.out/.szd/*',
            'data/COLUMBUS.out/.szd/*',
        ]
    },
    description =(
        "A spatial lag operator proposal implemented in Python21: "
        "only the k-nearest neighbor (oknn), %s."%package_name
    ),
    long_description              = readme,
    long_description_content_type = 'text/markdown',
    author       = 'Laurent Faucheux',
    author_email = "laurent.faucheux@hotmail.fr",
    url          = 'https://github.com/lfaucheux/%s'%package_name,
    download_url = 'https://github.com/lfaucheux/{}/archive/{}.tar.gz'.format(package_name, package_version),
    classifiers  = [
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    keywords = [
        'spatial econometrics',
        'oknn specification',
        'weight matrix',
    ],
    install_requires = requires,
)
