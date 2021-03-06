from setuptools import setup, find_packages


def read(filename):
    with open(filename, 'r') as file_handler:
        data = file_handler.read()

    return data

setup(
    name='pds4_tools',
    version='0.8',

    description='Package to read and display NASA PDS4 data',
    long_description=read('README'),

    author='Lev Nagdimunov',
    author_email='lnagdi1@astro.umd.edu',

    url='http://sbndev.astro.umd.edu/wiki/Python_PDS4_Tools',
    license='BSD',
    keywords=['pds4_viewer', 'pds4', 'pds'],

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: BSD License',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    packages=find_packages(exclude=['contrib', 'doc', 'tests*']),
    package_data={'': ['viewer/logo/*']},

    zip_safe=False,

    install_requires=[
        'numpy',
    ],

    extras_require={
        'viewer': ['Tkinter', 'matplotlib'],
    }
)
