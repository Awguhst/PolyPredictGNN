# Polymer Property and Solubility Prediction Demo

This app provides a simple interface to demonstrate two models for predicting **polymer properties** and **solubility** using **SMILES representations**. The app showcases the following models:

- **HybridGNN** - Predicts **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** for polymers.

- **SolubilityGNN** - Estimates **polymer solubility** in various solvents.

Both models utilize **Graph Neural Networks (GNNs)** to analyze molecular structures as graphs, combining them with physicochemical descriptors to improve accuracy.

---

## Motivation

Accurately predicting **polymer properties** is essential for material design across industries such as **packaging**, **electronics**, and **aerospace**. Traditional experimental approaches are often **resource-intensive** and **time-consuming**. This app offers a **data-driven, scalable alternative**, giving you real-time predictions for **thermal** and **solubility** properties.

---

## 🎥 **Live Demo**

![Streamlit app GIF](media/demo.gif)  
>*Explore the interactive Streamlit web app for predicting polymer properties.*

**Experience it in real time by clicking here: [Launch the App](https://polypredictgnn-wvus6kfzglaxlxlwl8aawy.streamlit.app/)**

---

## Streamlit App Interface

- **User Input:**
   - The user inputs a **polymer SMILES** string and an optional **solvent SMILES** string.

- **Prediction Process:**
   - The app checks if the **SMILES** strings are valid.
   - If valid, it predicts the **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** of the polymer.

- **Polymer and Solvent Visualization:**
   - 2D molecular images of the **polymer** and **solvent** (if provided) are displayed.

- **Solubility Prediction:**
   - If a solvent is provided, the app predicts the **solubility** of the polymer in the solvent as a percentage.

- **Similar Monomers, Solvents, and Results Display:**
   - The app recommends **top similar monomers** and **top alternative solvents** based on predicted solubility.
   - It also shows the predicted **thermal properties (Tg, Tm)** and **solubility results** alongside the 2D images of the polymer and solvent.

---

## Model Architecture

### HybridGNN (Polymer Property Prediction)
**HybridGNN** predicts **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** using molecular graph and physicochemical descriptors.

- **GIN**: Extracts structural features from the polymer graph.
- **GAT**: Focuses on important features with attention weights.
- **GraphConv**: Captures higher-level structural features.
- **PCA**: Reduces descriptor dimensionality.
- **Fully connected layers**: Combines graph features and descriptors with **GELU** activations and dropout.

**Multi-target Regression**: Simultaneously predicts **Tg** and **Tm**.

### SolubilityGNN (Polymer Solubility Prediction)
**SolubilityGNN** predicts polymer solubility using dual SMILES inputs (polymer and solvent).

- **TransformerConv & GINConv**: Model molecular interactions and patterns.
- **Fully connected layers**: Similar to HybridGNN, combining graph features with descriptors.

Both models use **GNN architectures** optimized for thermal properties and solubility prediction.

---

## Datasets

### Polymer Property Dataset
The **Polymer Property Dataset** contains **1,564** samples, each consisting of a **monomer SMILES** strings along with corresponding **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** values for the polymer. This dataset is used to train the **HybridGNN** model for predicting the thermal properties of polymers based on their molecular structure.

### Polymer Solubility Dataset
The **Polymer Solubility Dataset** includes **1,819** pairs of **monomer SMILES** and **solvent SMILES** strings. Each entry is labeled with a binary column indicating whether the polymer is **soluble** in the corresponding solvent. This dataset is utilized to train the **SolubilityGNN** model for predicting polymer-solvent solubility interactions.

Both datasets were sourced from **peer-reviewed papers**.

---

## Performance Metrics

### HybridGNN:
- **R² score for Tg**: 0.8 ± 0.02
- **R² score for Tm**: 0.7 ± 0.02

### SolubilityGNN:
- **Accuracy**: 82% ± 1.99%
- **AUC (Area Under the Curve)**: 0.88 ± 0.02

> All metrics were calculated using 5-fold cross-validation.
---

## References

- **Stubbs, C. D., et al.** (2025). Predicting homopolymer and copolymer solubility through machine learning. *Dalton Transactions*.  
   [https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c)

- **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.  
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y.** (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)

- **Xu, K., Hu, W., Leskovec, J., & Jegelka, S.** (2019). How Powerful are Graph Neural Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1810.00826](https://doi.org/10.48550/arXiv.1810.00826)

- **Feinberg, E. N., et al.** (2018). PotentialNet for Molecular Property Prediction. *ACS Central Science*, 4(11), 1520–1530.  
   [https://doi.org/10.1021/acscentsci.8b00507](https://doi.org/10.1021/acscentsci.8b00507)

- **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.** (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*.  
   [https://doi.org/10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)

- **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
   [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

- **RDKit: Open-source cheminformatics software.** (2006).  
   Available at: [http://www.rdkit.org](http://www.rdkit.org)
