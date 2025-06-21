# diffEcosys_Acclimation

![framework](Framework/diffEcosys_framework.jpg) 
**Description**: Differentiable ecosystem model (δ<sub>psn</sub>) ramework. The δpsn framework integrates a neural network (NN) component with a differentiable process‐based model in this case the photosynthesis module from FATES. The NN component includes three networks—NN<sub>B</sub>, NN<sub>V</sub>, and NN<sub>⍺</sub>—each trained on different sets of predictors, such as soil attributes, plant functional type (PFT), leaf nitrogen content (LNC), and environmental conditions. These networks learn distinct parameters (including V<sub>c,max25</sub>—the maximum carboxylation rate at 25°C, a key parameter representing plant photosynthetic capacity—and soil water constraint parameter B<sub>i</sub>) that feed into the photosynthesis module along with meteorological forcings (F) and constant terrain attributes (θ<sub>c</sub>). NN<sub>⍺</sub> captures parameter acclimation to environmental factors, and predicts a perturbation term (⍺<sub>p</sub>) which is multiplied by each parameter (p) learned by NN<sub>V</sub>. The photosynthesis module then simulates A<sub>N</sub> (net photosynthesis) and g<sub>S</sub> (stomatal conductance). The outputs, including V<sub>c,max25</sub>, are compared against multivariate observations to compute a loss function, which guides the backpropagation process to update the NN parameters.

**Citation**: Please cite this paper if the code is helpful to you.

Aboelyazeed, D., Xu, C., Gu, L., Luo, X.,
Liu, J., Lawson, K., & Shen, C. (2025).
Inferring plant acclimation and improving
model generalizability with differentiable
physics‐informed machine learning of
photosynthesis. Journal of Geophysical
Research: Biogeosciences, 130,
e2024JG008552. https://doi.org/10.1029/2024JG008552
