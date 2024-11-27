from setuptools import setup, find_packages 
from pathlib import Path

setup( 
    name='pyEpiAneufinder', 
    version='0.1', 
    description='Python version of epiAneufinder for calling CNVs from scATACseq', 
    author='Aikaterini Symeonidi, Katharina Schmid', 
    author_email='asymeonidi@bmc.med.lmu.de, katharina.schmid@bmc.med.lmu.de',
    license='BSD',
    python_requires='>=3.10',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    packages=find_packages() 
) 