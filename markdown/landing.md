<div style="text-align: center; padding: 40px 20px 20px 20px;">
  <p style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 3px; color: #7A9E7E; margin-bottom: 8px;">
    PhD Dissertation Research
  </p>
  <h1 style="font-size: 2.5rem; color: #3D5A5A; font-family: Georgia, serif; line-height: 1.35; margin-bottom: 16px;">
    A Multi-Scalar Framework for <em>Pinus taeda</em> (L.) Yield Prediction
  </h1>
  <p style="font-size: 1.1rem; color: #1A1A1A; max-width: 680px; margin: 0 auto 20px auto; line-height: 1.75;">
    Integrating Advanced Biometrics and Remote Sensing to model forest yields
    across individual trees, stands, and landscapes.
  </p>

  <div style="display: inline-block; text-align: center; background: #F5F0E8; border: 1px solid #C5B89A; border-radius: 10px; padding: 20px 32px; margin: 0 auto 12px auto; max-width: 640px;">
    <p style="font-size: 1.05rem; font-family: Georgia, serif; color: #3D5A5A; font-weight: bold; margin: 0 0 4px 0;">
      Gunjan Barua
    </p>
    <p style="font-size: 0.92rem; color: #1A1A1A; margin: 0 0 14px 0;">
      PhD Candidate · Forest Resources and Environmental Conservation · Virginia Tech
    </p>
    <p style="font-size: 0.88rem; color: #5B7B7A; font-weight: bold; margin: 0 0 3px 0; text-transform: uppercase; letter-spacing: 1.5px;">
      Advisors
    </p>
    <p style="font-size: 0.9rem; color: #1A1A1A; margin: 0 0 3px 0;">
      Dr. Valerie Thomas · Forest Resources and Environmental Conservation · Virginia Tech
    </p>
    <p style="font-size: 0.9rem; color: #1A1A1A; margin: 0 0 16px 0;">
      Dr. David Carter · Department of Forestry · Michigan State University
    </p>
    <p style="font-size: 0.9rem; color: #1A1A1A; margin: 0; line-height: 2;">
      ✉ <a href="mailto:gunjanb@vt.edu" style="color: #5B7B7A; text-decoration: none;">gunjanb@vt.edu</a>
      &nbsp;·&nbsp;
      <a href="https://www.linkedin.com/in/gunjanbarua/" target="_blank" style="color: #5B7B7A; text-decoration: none;">LinkedIn</a>
      &nbsp;·&nbsp;
      <a href="https://gunjanbarua.github.io/" target="_blank" style="color: #5B7B7A; text-decoration: none;">Personal Website</a>
    </p>
  </div>
</div>

---

## The Research Challenge

*Pinus taeda* (L.), commonly known as loblolly pine, is the cornerstone of modern plantation forestry in the United States. Occupying over 14 million hectares, it accounts for approximately 60% of the total US wood supply and serves as a massive carbon sink. Accurate yield estimation in these forests is an absolute necessity for optimizing economic value and meeting global greenhouse gas reduction goals. 

However, traditional forest monitoring is struggling to keep pace. Conventional field inventories are labor-intensive, expensive, and typically capture data from less than 1% of the standing timber. This sparse sampling fails to capture the true spatial heterogeneity of complex forest structures, resulting in significant uncertainty and "yield gaps" where actual production deviates from traditional models.

While remote sensing technologies, such as UAV-LiDAR and multi-spectral satellites offer a multi-scalar solution to these spatial limitations, operationalizing this data is incredibly complex. Current frameworks treat yield estimation as a static problem, struggling to account for how tree competition evolves over time, and frequently suffer from sensor "saturation" in dense, high-biomass stands. 

**This dissertation addresses these critical gaps.** By integrating high-resolution, single-date LiDAR with continuous multi-sensor satellite time series and state-of-the-art Machine Learning (ML) and Deep Learning algorithms, this research transitions forest yield modeling from static inventory snapshots to dynamic, data-driven forecasting systems.

---

## Three Interconnected Projects

To achieve this overarching goal, the research is structured across three progressive spatial and temporal scales:

#### 1. Precision at the Individual Tree Level (Medium-Term Yield)
Can machine learning models predict future tree volume using only a single drone flight? This project establishes a proof-of-concept for using UAV-LiDAR to forecast individual tree yield over a four-year interval. By comparing Random Forest (RF) and Support Vector Machines (SVM) against traditional linear regression, the study proves that combining Individual Tree Crown (ITC) metrics with distance-dependent Competition Indices (CIs) provides highly accurate, scalable yield predictions.

#### 2. Predicting Through Time (Longitudinal Stand Dynamics)
Forests are dynamic, but can our static models keep up? Building on the first project, this study investigates the "temporal transferability" of a single early-age LiDAR acquisition (Age 8) to forecast annual yield up to 7 years into the future. It uncovers a fascinating biological shift: early-year growth is driven by individual tree size and vigor, but as the canopy closes over time, neighborhood competition becomes the dominant predictor of yield.

#### 3. Scaling to the Landscape (SAR-Optical Fusion & Deep Learning)
How do we monitor regional timber volume without the high cost of continuous LiDAR flights? The final project scales from the individual tree to the plot level using 24 months of open-source satellite data (Sentinel-1 SAR and Sentinel-2 optical). By employing advanced Recurrent Neural Networks (LSTM and GRU), this study demonstrates how deep learning sequence models can filter out noise and overcome optical signal saturation, offering a highly accurate, cost-effective tool for continuous, large-scale plantation monitoring.

---

## How to Explore This Dashboard

**Start with The Big Picture** — a visual, jargon-free summary of the
entire dissertation. Then dive deeper into any project that interests you.

Use the **sidebar** to navigate between pages. Each project page follows a
**scrollytelling** format:

1. Read the narrative context
2. Interact with the data visualisation
3. Explore the feature importance analysis

<br>
