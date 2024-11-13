from setuptools import setup 
  
setup( 
    name='pyEpiAneufinder', 
    version='0.1', 
    description='Python version of epiAneufinder for calling CNVs from scATACseq', 
    author='Aikaterini Symeonidi, Katharina Schmid', 
    author_email='asymeonidi@bmc.med.lmu.de, katharina.schmid@bmc.med.lmu.de', 
    #packages=[], 
    install_requires=[ 
        'numpy', 
        'pandas', 
    ], 
) 