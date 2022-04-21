from setuptools import setup

setuptools.setup(
    name = 'coastal_data',
    version = '0.1',
    author = 'Susann Aschenneller',
    author_email = 's.aschenneller@utwente.nl',
    description = 'Functionalities to process observational datasets describing a coast',
    install_requires=['numpy','xarray','json','datetime'],
    classifiers=["Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha"]
)