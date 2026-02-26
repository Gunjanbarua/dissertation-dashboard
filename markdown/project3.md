# Project 3 · Remote Sensing Integration

## Predicting Forest Volume from the Sky

> *Can airborne LiDAR and multispectral remote sensing replace intensive ground-based surveys
> for stand-level yield prediction — and which model architecture best exploits the 3D
> structural information captured by ALS?*

The operational bottleneck in large-scale forest inventory is the **cost of field data
collection**. Measuring thousands of individual trees across vast plantation landscapes is
labour-intensive, temporally sparse, and logistically constrained. Airborne Laser Scanning
(ALS) offers a scalable alternative: a single survey flight captures the complete
three-dimensional canopy structure of an entire estate in hours.

### The Remote Sensing Feature Set

From each ALS point cloud, a suite of **height distribution statistics** and **canopy density
metrics** were extracted at the plot level:

- **Height percentiles** (H10 · H25 · H50 · H75 · H90 · H95) — describing the vertical profile
- **Canopy Cover (CC)** — fraction of first returns above a threshold height
- **Rumple Index** — surface complexity of the canopy (proxy for structural heterogeneity)
- **Density strata** — proportion of returns within defined vertical layers
- **Spectral indices** (NDVI, EVI) — where multispectral imagery was co-acquired

### Seven Competing Architectures

| Model | Category | Key Characteristic |
|-------|----------|--------------------|
| **GRU** | Deep Learning (RNN) | Gated unit captures ordered height-profile structure |
| **LSTM** | Deep Learning (RNN) | Long short-term memory for complex sequential patterns |
| **Random Forest** | Ensemble (Bagging) | Robust to outliers; interpretable via permutation importance |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art accuracy; regularised gradient descent |
| **LightGBM** | Ensemble (Boosting) | Memory-efficient; excels on high-dimensional feature sets |
| **SVR** | Kernel Method | Strong baseline in mid-dimensional regression tasks |
| **GBM** | Ensemble (Boosting) | Classical gradient boosting reference |

---

### The Thinning Effect

A critical forest management variable is **thinning status**. Thinned stands have undergone
selective tree removal to reduce competition, producing structurally distinct canopies with
higher light penetration and more variable vertical profiles. This confounds remote sensing
algorithms that were calibrated on uniform, unthinned stands.

Use the **thinning status colour coding** in the scatter plot below to assess whether each
model handles the structural contrast between thinned and unthinned stands equally well —
or whether systematic bias emerges for one management type.

---

### Selecting a Model

Use the **model selector** above the chart to switch between all seven algorithm predictions.
Compare where each model under- or over-predicts, and which stand conditions drive the
largest errors.

---
