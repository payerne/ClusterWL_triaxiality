# ClusterWL_triaxiality
**Author:** Constantin Payerne
**Contact:** constantin.payerne@gmail.com

This repository provides tools to generate mock shear and convergence maps for elliptical clusters. It provides also the tools to estimate the cluster lensing shear multipoles from sheared background galaxies, as well as modelling the lensing multipoles. 

- These data analysis and modeling codes are used in the context of my work within The Three Hundred (The300) cluster simulation project ([Cui et al. 2018](http://ui.adsabs.harvard.edu/abs/2018MNRAS.480.2898C/abstract)). The official website of the The300 project can be found [here](https://weiguangcui.github.io/the300/).
- This project on inferring the triaxiality of The300 simulater cluster from lensing shear multipoles has been presented at the conference "[Observing the Universe at millimetre wavelengths](https://lpsc-indico.in2p3.fr/event/2859/contributions/6402/)" and in the Proceeding [Payerne et al. (2023)](https://www.epj-conferences.org/articles/epjconf/abs/2024/03/epjconf_mmUniverse2023_00039/epjconf_mmUniverse2023_00039.html). More details can be found in my [PhD thesis](https://theses.hal.science/tel-04405434) (Chapter 6).

This repository uses the following dependencies:
- [NumPy](https://www.numpy.org/)
- [SciPy](https://scipy.org/)
- [Astropy](https://www.astropy.org/)
- [CLMM](https://github.com/LSSTDESC/CLMM) (LSST-DESC Cluster Lensing Mass Modeling)
- [lenspack](https://github.com/CosmoStat/lenspack) (implementation of the Kaiser & Squires (1993) inversion formula)
- [Matplotlib](https://matplotlib.org/) 