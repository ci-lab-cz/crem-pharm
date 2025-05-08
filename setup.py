import setuptools
from os import path
import crem

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="crempharm",
    version=crem.__version__,
    author="Pavel Polishchuk",
    author_email="pavel_polishchuk@ukr.net",
    description="CReM-pharm: enumeration of structures based on 3D pharmacophores by means of CReM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ci-lab-cz/crem-pharm",
    packages=['crempharm'],
    include_package_data=True,
    package_data={'crem': ['scripts/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    python_requires='>=3.6',
    install_requires=['pmapper>=1.1.3'],
    extras_require={
        'rdkit': ['rdkit>=2017.09'],
    },
    entry_points={'console_scripts':
                      ['crempharm = crempharm.pgrow:entry_point',
                       'crempharm_add_pmapper = crempharm.scripts.crem_db_labeling:entry_point']}
)
