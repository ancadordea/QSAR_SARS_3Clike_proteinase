# QSAR Modeling of SARS-CoV-2 3C-like Protease (Mpro) Inhibitors

This repository contains a full pipeline for QSAR modeling of small molecule inhibitors targeting the SARS-CoV-2 3C-like protease (Mpro), a key enzyme in the viral life cycle, using bioactivity data from [ChEMBL Target CHEMBL3927](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL3927/). The analysis includes preprocessing, chemical space analysis (EDA), fingerprint generation, regression modeling with Random Forests and key structure–activity patterns analysis.

---

## **Project Purpose**

The main objective of this project is to build an accurate and generalisable machine learning model to predict the biological activity of chemical compounds, specifically focusing on their pIC50 values against a particular target (e.g., a kinase or GPCR). Predicting pIC50—a logarithmic measure of inhibitory potency—is critical for early-stage drug discovery, allowing researchers to identify promising leads before proceeding to expensive and time-consuming in vitro or in vivo testing.

By using structure-based molecular representations like Morgan fingerprints, this project aims to leverage the power of chemical informatics and machine learning to streamline virtual screening pipelines and accelerate hit-to-lead optimization in medicinal chemistry.

---

## Overview

* **Target:** SARS-CoV-2 3C-like protease (Mpro)
* **Bioactivity Metric:** IC₅₀ values (converted to pIC₅₀)
* **Data Source:** ChEMBL (Target ID: CHEMBL3927)
* **Goal:** Build a QSAR model to predict compound potency (pIC₅₀) and analyse structure–activity relationships.

---

##  Project Structure

```
.
├── data/                     # Raw and processed data
├── Results/                  # All saved plots and results
├── preprocessing_and_EDA.py # Preprocessing, EDA, fingerprinting
├── analysis.py              # ML modeling, evaluation, feature interpretation
├── fingerprints.npy         # Numpy array of fingerprints
├── metadata.pkl             # Bioactivity and descriptors
├── mols_and_bitinfo.pkl     # RDKit Mol objects and bit info for interpretation
├── README.md
```

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Workflow Summary

### 1. **Data Preprocessing & Chemical Space Analysis(EDA)**

Script: `preprocessing_and_EDA.py`

* Load and clean ChEMBL data
* Filter only IC₅₀ values and assign **bioactivity classes**:

  * `active`: IC₅₀ ≤ 1000 nM
  * `inactive`: IC₅₀ ≥ 10000 nM
  * `intermediate`: 1000 < IC₅₀ < 10000 nM (removed before modeling) --> Removing "intermediate" samples helped to create a **cleaner separation** between potent and weak inhibitors for classification and regression tasks.

This stratification reflects typical thresholds used in medicinal chemistry ([Tropsha, A. et al. J. Chem. Inf. Model., 2010](https://onlinelibrary.wiley.com/doi/10.1002/minf.201000061))

* Convert IC₅₀ → pIC₅₀ -->  pIC₅₀ improves interpretability (higher = more potent) and reduces data skewness.
* Compute **Lipinski descriptors** --> These descriptors are interpretable and reflect key ADME properties relevant to drug-likeness.
* Generate **visualizations**: histograms, violin plots, scatterplots, boxplots to inspect class balance and potency spread.
* Statistical tests: **Kruskal–Wallis** and **Mann–Whitney U**
* Generate **Morgan fingerprints (ECFP4)** for machine learning --> Generated **Extended Connectivity Fingerprints (ECFP)** via Morgan algorithm with:
    * **Radius = 2**, which encodes circular substructures (ECFP4 standard)
    * **Length = 1024 bits**, binary vectors representing molecular substructures
  -> ECFP4 is widely used in QSAR for its balance between sensitivity and generalisation ([Rensi, S et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5942586/?utm_source=chatgpt.com))

  Preprocessing included:
  * Salt removal
  * Retaining the largest organic fragment (to avoid counterions/solvents)
  
Outputs:

* Cleaned datasets (`bioactivity_data.csv`, `fingerprints.csv`)
* Saved plots in `Results/`
* Feature and fingerprint arrays for modeling

---

### 2. **Model Training & Evaluation**

Script: `analysis.py`

* Load fingerprints and bioactivity labels
* Split data (stratified) --> (80/20) using bioactivity class to preserve class distribution in both train and test sets
* Train **Random Forest Regressor** on pIC₅₀
* Evaluate using:

  * R²
  * RMSE
  * Prediction plots
* Optional: **Hyperparameter tuning** with `GridSearchCV`
* Optional: **Feature selection**:

  * Remove low-variance bits
  * Use top-ranked bits by importance
* Visualise model predictions

---

## 3. Substructure Analysis

### Fingerprint Bit Mapping

* For top N important fingerprint bits, substructures were extracted using:

  * RDKit's `bitInfo` map
  * Atom environment extraction at triggered bit positions

* Enables **chemical interpretation** of model predictions by linking key fingerprint bits to molecular substructures.
* Helps identify **privileged scaffolds** or **undesirable motifs**.

---

## **Usefulness of the Project**

1. **Drug Discovery Acceleration**: Predicting bioactivity in silico reduces experimental burden.
2. **Cost-Efficiency**: Early triaging of compounds avoids unnecessary synthesis and biological testing.
3. **Generalisation**: Using structure-based fingerprints makes the model applicable across multiple chemical scaffolds and targets.
4. **Open Science Contribution**: This project provides a reproducible pipeline that can be reused or extended by researchers in cheminformatics, pharmacology, or computational biology.

---

## References

* ChEMBL Database: [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
* RDKit: [http://www.rdkit.org](http://www.rdkit.org)
* Lipinski Rule of Five: [https://doi.org/10.1016/S0169-409X(96)00423-1](https://doi.org/10.1016/S0169-409X%2896%2900423-1)
