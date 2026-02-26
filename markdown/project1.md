# Project 1 · Medium-Term Yield Prediction using UAV-LiDAR

## The Core Objective

> *Can high-density drone data replace resource-intensive field measurements for future yield forecasting?*

This project establishes a proof-of-concept for using a single UAV-LiDAR acquisition to predict the individual tree stem volume of *Pinus taeda* (L.) four years into the future (from age 8 to age 12). By relying exclusively on LiDAR-derived **Individual Tree Crown (ITC)** metrics and spatially explicit **Competition Indices (CIs)**, we investigated whether machine learning can accurately capture the complex dynamics of tree growth without traditional field mensuration.

#### The article was published in **Forest Ecology and Management**. [Read the article here](https://doi.org/10.1016/j.foreco.2025.122977)

### Methodology & Data

The study leveraged data from two distinct physiographic regions: the Piedmont of Virginia (Reynolds Homestead) and the Coastal Plain of North Carolina (Bladen Lakes). 

To thoroughly test the robustness of our models, the data encompassed a wide range of silvicultural treatments and initial planting densities, including 618, 1,236, and 1,853 trees per hectare (TPH), as well as variable-density Nelder trials. High-density UAV-LiDAR (312 to 498 pulses/m²) was utilized to extract highly detailed, three-dimensional structural features for each tree, including:
*   **Individual Tree Crown (ITC) Metrics:** Tree top height, height to live crown, 3D convex hull volumes, surface areas, and Leaf Area Index (LAI).
*   **Competition Indices (CIs):** Distance-dependent metrics (such as the Hegyi and SILVA indices) that quantify the competitive stress exerted by a focal tree's immediate neighbors.

### The Modeling Approach

We compared the predictive capabilities of two non-parametric Machine Learning algorithms—**Random Forest (RF)** and **Support Vector Regression (SVR)**—against a traditional parametric baseline, **Multiple Linear Regression (MLR)**. 

Furthermore, to test the operational efficiency of these models, we applied Permutation Feature Importance (PFI) to identify the most critical drivers of yield. We then trained "Reduced" models using only the top seven most influential predictors to determine if we could reduce computational complexity without sacrificing predictive power.

---

### Key Findings & Evaluation

The validation on our independent test dataset revealed several critical insights into the integration of LiDAR and machine learning for forest biometrics:

*   **Machine Learning Outperforms MLR:** The traditional MLR model failed to meet the assumptions of homoscedasticity, struggling with the complex, non-linear realities of stand development. Both RF and SVM successfully captured these dynamics.
*   **High Individual and Stand-Level Accuracy:** The SVM model achieved the highest accuracy at the individual tree level (nRMSE: 9.59%). When individual predictions were aggregated to the stand level, the RF model excelled, deviating from the true field volume by only 1.53%.
*   **The Power of Parsimony:** The Reduced models maintained excellent accuracy. The Reduced SVM model underpredicted stand volume by only 0.90%, proving that highly dimensional LiDAR data can be efficiently streamlined for operational use.
*   **The Density Constraint:** A clear inverse relationship was observed between initial planting density and model accuracy. As stands became denser (>1,235 TPH), intensified canopy interlocking introduced segmentation noise, slightly constraining the predictive accuracy of the models.

*Explore the interactive plots below to visualize the model performance, error distributions, and the structural variables driving these predictions.*

---
