# DNN-PhaseMatrix-Deep-Learning-for-Non-spherical-Particle-Scattering

Deep learning models for fast and accurate prediction of scattering matrix elements for non-spherical particles.

## Model Description

- Training on the database calculated by Invariant Imbedding T-matrix method (IITM) and the Improved Geometric Optics Method  (IGOM) of super-spheroids
- Predicts scattering matrix elements (P11, P12/P11, P43/P11, P22/P11, P33/P11, P44/P11)
- Parameter ranges:
  - Size parameter (x): 0.1-1000
  - Real part of refractive index (mr): 1.30-1.80
  - Imaginary part of refractive index (mi): 0.0000001-0.1
  - Aspect ratio (asp): 0.5-2.0
  - Roundness (n): 1.2~3.0
  - Scattering angle (theta): 0-180°
- High accuracy compared with IGOM calculations in medium size parameters.
- Significantly faster than traditional methods

## Superspheroid Model details

-  Geometry: super-spheroid  (x/a)^(2/n)+(y/a)^(2/n)+(z/c)^(2/n)=1 
-  Shape parameters:
    Aspect ratio (asp) = a/c
    Roundness n
-  Physical parameters:
    Size parameter: x = 2πr/λ (r = max(a,c))
    Complex refractive index: m = mr + mi*i

## Installation
    -python
    tensorflow==2.4.0
    numpy==1.19.5
    pandas==1.3.4

## Model Prediction
    - Single prediction example
    - Batch prediction example
    - Input parameters: mr, log(mi), asp, n, log(xszie), theta
    - Output parameters: log(P11) or Pij/P11

## Directory Structure

main/
├── models             # Pre-trained models
├── predict.py             # Example code
└── README.md            # README file

## Notes
mi and size parameters need log10 transformation before input

## References
[1] Xi, Y., Bi, L., & Lin, W. (2025). Application of deep learning to enhance the computation of phase matrices of non-spherical atmospheric particles across all size parameters. Manuscript submitted to *Journal of Geophysical Research - Machine Learning and Computation*.
[2] Bi, L., Xi, Y., Han, W., & Du, Z. (2024). How machine learning approaches are useful in computing the optical properties of non-spherical particles across a broad range of size parameters? Journal of Quantitative Spectroscopy and Radiative Transfer, 323, 109057. https://doi.org/10.1016/j.jqsrt.2024.109057
