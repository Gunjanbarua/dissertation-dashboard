# Project 2 · Temporal Transferability of Yield Prediction

## Predicting Multi-Year Growth from a Single Drone Flight

> *Can a single, early-age drone flight accurately predict individual tree yield up to seven years into the future? And how do the drivers of tree growth change as the forest matures?*

In operational forestry, acquiring high-density LiDAR data annually is often cost-prohibitive. A common practice is to use a single LiDAR snapshot to predict future yield, implicitly assuming that the relationship between a tree's early structure and its future growth remains static. However, as young, open stands transition into dense, closed-canopy forests, the ecological drivers of growth fundamentally change.

### The Temporal Challenge & Objectives

This project investigates the **temporal transferability** of machine learning models. Using a single UAV-LiDAR acquisition obtained at age 8, we forecasted the annual individual tree yield of *Pinus taeda* (L.) from age 9 through 15. 

We sought to answer:
1. **Predictive Robustness:** Can non-parametric ML models (Random Forest, SVR) outperform traditional Linear Mixed-Effects (LME) models over an extended forecasting horizon?
2. **Dynamic Stand Behavior:** Does the predictive importance of variables systematically shift from individual tree metrics to neighborhood competition indices as canopy closure intensifies?
3. **The Density Effect:** How do initial planting densities (618, 1,235, and 1,853 TPH) constrain multi-year predictive accuracy?

### Key Findings & Biological Insights

The longitudinal analysis revealed critical insights into both algorithmic stability and forest ecology:

* **Random Forest Excels over Time:** Both ML models significantly outperformed the LME baseline. However, Random Forest demonstrated superior robustness across the entire 7-year prediction horizon (R² ≥ 0.83), whereas SVR struggled with noise in the early years and high-density plots.
* **A Fundamental Biological Shift:** Permutation Feature Importance (PFI) analysis beautifully confirmed our ecological hypothesis. In the early forecasting years (ages 9–11), yield was driven primarily by individual photosynthetic capacity (e.g., top height, crown volume). By age 13, as the canopy closed, a distance-dependent competition index (CI_Z) emerged as the single most dominant predictor. 
* **The High-Density Bottleneck:** A significant three-way interaction between planting density, stand age, and model type was discovered. In the highest density stands (1,853 TPH), intense canopy interlocking degraded individual tree crown segmentation, significantly reducing the accuracy of the SVR model compared to the more robust, ensemble-based RF model.

*Use the interactive slider in the plots below to animate tree growth from age 9 to 15, and observe how prediction accuracy and feature importance dynamically shift as the stand matures.*

---

#### What to Look For in the Animated Scatter Plot

Use the **age slider** (or press Play) to step through stand ages chronologically. Notice how:

- The **scatter cloud tightens** as age increases — predictions become more precise as trees
  differentiate in size
- **High variance in young stands** reflects the stochastic nature of early establishment
- The **1:1 reference line** alignment improves with age, demonstrating model convergence toward
  the true growth trajectory

---
