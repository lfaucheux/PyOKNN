from distutils.core import setup
from PyOKNN import __version__ as v

setup(
    name='PyOKNN',
    packages = ['PyOKNN'],
    package_data={
        'PyOKNN': [
            'examples/*',
            'data/tests/*',
            'data/tests/subfolder/*',
            'data/COLUMBUS/*',
            'data/COLUMBUS.out/.szd/*',
            'data/COLUMBUS.out/.szd/*',
        ]
    },
    version=v,
    description="A spatial lag operator proposal implemented in Python: only the k-nearest neighbor (oknn).",
    author='Laurent Faucheux',
    author_email="laurent.faucheux@hotmail.fr",
    url='https://github.com/lfaucheux/PyOKNN',
    download_url = 'https://github.com/lfaucheux/PyOKNN/archive/{}.tar.gz'.format(v),
    keywords = ['spatial econometrics', 'weight matrix', 'oknn specification'],
    classifiers=[],
)
