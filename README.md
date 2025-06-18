# diffEcosys_Acclimation

![framework](Framework/diffEcosys_framework.jpg) 
Differentiable Ecosystem Model (δ<sub>psn</sub>) Framework. The δpsn framework integrates a neural network (NN)
component with a differentiable process‐based model in this case the photosynthesis module from FATES (The Functionally Assembled Terrestial Ecosystem Simulator). The NN
component includes three networks—NN<sub>B,i</sub>, NN<sub>V</sub>, and NN<sub>⍺</sub>—each trained on different sets of predictors, such as soil
attributes, plant functional type (PFT), leaf nitrogen content (LNC), and environmental conditions. These networks learn
distinct parameters that feed into the photosynthesis module along with meteorological forcings (F) and constant terrain
attributes (θ<sub>c</sub>). The photosynthesis module then simulates A<sub>N</sub> (net photosynthesis) and g<sub>S</sub> (stomatal conductance). The
outputs, including V<sub>c,max25</sub> (maximum carboxylation rate at 25°C, a key parameter representing plant photosynthetic
capacity), are compared against multivariate observations to compute a loss function, which guides the backpropagation
process to update the NN parameters.

Aboelyazeed, D., Xu, C., Gu, L., Luo, X.,
Liu, J., Lawson, K., & Shen, C. (2025).
Inferring plant acclimation and improving
model generalizability with differentiable
physics‐informed machine learning of
photosynthesis. Journal of Geophysical
Research: Biogeosciences, 130,
e2024JG008552. https://doi.org/10.1029/
