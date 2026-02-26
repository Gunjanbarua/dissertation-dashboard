# Project 3 · Multi-Sensor Fusion & Machine Learning

## Predicting Loblolly Pine Volume from Orbit

*Can open-source satellite time-series replace labor-intensive field inventories for plantation management?*

The operational bottleneck in forest management is the high cost and low temporal resolution of traditional ground-based inventories. While field plots offer high local accuracy, they are resource-intensive and often conducted only every five or more years. This study presents a scalable alternative: integrating 24 months of Sentinel-1 (SAR) and Sentinel-2 (Optical) data to predict plot-level timber volume ($V_{ob}$) across 258 sites in the southeastern United States.

---

## The Remote Sensing Feature Set

Instead of 3D point cloud metrics, this framework leverages the synergy between spectral reflectance and radar backscatter, temporally aggregated into seasonal means to capture dominant phenological signals:

- **Sentinel-1 Radar Indices:** Includes VH and VV backscatter, Radar Vegetation Index (RVI), and terrain-normalized Gamma Nought ($\gamma^0$) to capture structural information.

- **Sentinel-2 Spectral Bands:** 13 multispectral bands ranging from visible to Short-Wave Infrared (SWIR).

- **Vegetation Indices:** A suite of 21 indicators, including NDVI, EVI, and the Normalized Difference Lignin Index (NDLI), used to track canopy vigor and moisture stress.

- **Dimensionality Reduction:** Predictive power remained stable ($R^2 = 0.49$) even when the feature space was compressed from 48 variables down to the top 10 most influential drivers.

---

## Seven Competing Architectures

We utilized Bayesian optimization to tune seven supervised learning models. The results revealed that deep learning sequence models (RNNs) offer a robust and potentially superior alternative to traditional ensemble methods for processing dense satellite time series.

| Model Category | Key Characteristic in This Study |
|---|---|
| **GRU (Deep Learning)** | Top Performer ($RMSE = 60.38\text{ m}^3/\text{ha}$); gating mechanisms effectively filtered noise in backscatter and phenological signals. |
| **LSTM (Deep Learning)** | Achieved comparable performance ($R^2 = 0.49$); explicitly designed to handle temporal dependencies in biomass accumulation. |
| **Random Forest (RF)** | Strong ensemble baseline ($R^2 = 0.49$); relied heavily on SWIR bands (B11) which correlate to canopy water content and density. |
| **XGBoost** | High-performance gradient boosting; achieved moderate accuracy with an $R^2$ of 0.43. |
| **LightGBM** | Optimized for efficiency; showed intermediate error levels ($RMSE = 64.85\text{ m}^3/\text{ha}$). |
| **SVR** | Kernel-based regression baseline; achieved an $R^2$ of 0.40. |
| **GBM** | Traditional gradient boosting; yielded the lowest accuracy among all tested algorithms ($RMSE = 66.72\text{ m}^3/\text{ha}$). |

---

## The Thinning & Density Effect

Model accuracy is highly sensitive to silvicultural conditions. We identified a fundamental limitation in retrieving volume from current satellite sensors: a consistent "U-shaped" error trend across stand densities.

- **Optimal Density Range:** All models achieved their highest predictive accuracy in medium-density stands (618–1236 TPH), where the canopy generates a coherent signal without inducing complete saturation.

- **High-Density Saturation:** Substantial increases in error (up to $99.36\text{ m}^3/\text{ha}$) occurred in high-density stands (>1237 TPH), consistent with signal saturation in C-band SAR and optical indices.

- **Thinning Effect:** Every model demonstrated higher accuracy on thinned plots. For the GRU model, RMSE was reduced from $67.01\text{ m}^3/\text{ha}$ in unthinned stands to $48.97\text{ m}^3/\text{ha}$ in thinned plots. Thinning may reduce canopy complexity and delay the onset of signal saturation.

---

### Selecting a Model

Use the **model selector** above the chart to switch between all seven algorithm predictions.
Compare where each model under- or over-predicts, and which stand conditions drive the
largest errors.

---
