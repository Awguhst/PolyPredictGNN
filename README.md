# Polymer Property and Solubility Prediction Demo

This app provides a simple interface to demonstrate two advanced models for predicting **polymer properties** and **solubility** using **SMILES representations**. The app showcases the following models:

- **HybridGNN** – Predicts **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** for polymers.

- **SolubilityGNN** – Estimates **polymer solubility** in various solvents.

Both models utilize **Graph Neural Networks (GNNs)** to analyze molecular structures as graphs, combining them with physicochemical descriptors to improve prediction accuracy.

---

## Motivation

The ability to predict polymer properties and solubility is crucial for material design in various industries, including **packaging**, **electronics**, and **aerospace**. Traditional experimental methods are often resource intensive and time consuming. This framework offers a **data-driven, scalable alternative**, providing real-time predictions for:

- **Thermal Properties**: Glass Transition Temperature (Tg) and Melting Temperature (Tm).
- **Solubility**: Polymer solubility in different solvents, helping with solvent selection for specific applications.

By leveraging modern **graph-based deep learning** techniques, the framework bridges the gap between molecular structure and practical material design.

---

## 🎥 **Demo**
![Streamlit app GIF](media/demo.gif)

> *Visualization of the interactive Streamlit web app for polymer property prediction.*

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
**HybridGNN** predicts the **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** of polymers using both molecular graph representation and physicochemical descriptors through **multi-target regression**.

- **GIN**: Extracts structural features from the polymer graph, capturing atom and bond interactions.
- **GAT**: Assigns attention weights to graph nodes, focusing on important features for complex molecular structures.
- **GraphConv**: Processes molecular graph data, capturing higher-level structural features.
- **PCA**: Reduces dimensionality of descriptors, improving model efficiency while retaining key chemical information.
- **Fully connected layers**: Combine graph embeddings and descriptor vectors for enhanced prediction.

**Multi-target Regression**: Predicts both **Tg** and **Tm** simultaneously, optimizing the process for both properties.

### SolubilityGNN (Polymer Solubility Prediction)
**SolubilityGNN** predicts polymer solubility in a given solvent, considering polymer-solvent interactions with **dual SMILES inputs** (polymer and solvent).

- **TransformerConv**: Models molecular interactions by considering both node and edge features.
- **GINConv**: Extracts higher-level molecular patterns from polymer and solvent graphs.
- **Edge Features**: Uses both atom (node) and bond (edge) features to better capture solubility dynamics.
- **Fully connected layers**: Combines graph features with molecular descriptors, utilizing **GELU activations** and dropout to prevent overfitting.

**Dual SMILES Inputs**: Takes separate SMILES strings for the polymer and solvent, predicting the polymer's solubility in the solvent.

Both models use **GNN-based architectures** with enhancements tailored to predict thermal properties (Tg, Tm) and solubility, combining graph embeddings with molecular descriptors for accurate predictions.

---

## Datasets

### Polymer Property Dataset
The **Polymer Property Dataset** contains **1,564** samples, each consisting of a **monomer SMILES** string along with corresponding **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)** values for the polymer. This dataset is used to train the **HybridGNN** model for predicting the thermal properties of polymers based on their molecular structure.

### Polymer Solubility Dataset
The **Polymer Solubility Dataset** includes **1,819** pairs of **monomer SMILES** and **solvent SMILES** strings. Each entry is labeled with a binary column indicating whether the polymer is **soluble** in the corresponding solvent. This dataset is utilized to train the **SolubilityGNN** model for predicting polymer-solvent solubility interactions.

Both datasets were sourced from **peer-reviewed papers**.

---

## Performance Metrics

### HybridGNN:
- **R² score for Tg**: 0.80 (5-fold cross-validation)
- **R² score for Tm**: 0.70 (5-fold cross-validation)

### SolubilityGNN:
- **Accuracy**: 82% (5-fold cross-validation)
- **AUC (Area Under the Curve)**: 0.88 (5-fold cross-validation)

---

## References

- **Feinberg, E. N., et al.** (2018). PotentialNet for Molecular Property Prediction. *ACS Central Science*, 4(11), 1520–1530.  
   [https://doi.org/10.1021/acscentsci.8b00507](https://doi.org/10.1021/acscentsci.8b00507)

- **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.** (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*.  
   [https://doi.org/10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)

- **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
   [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

- **RDKit: Open-source cheminformatics software.** (2006).  
   Available at: [http://www.rdkit.org](http://www.rdkit.org)

- **Stubbs, C. D., et al.** (2025). Predicting homopolymer and copolymer solubility through machine learning. *Dalton Transactions*.  
   [https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c)

- **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.  
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y.** (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)

- **Xu, K., Hu, W., Leskovec, J., & Jegelka, S.** (2019). How Powerful are Graph Neural Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1810.00826](https://doi.org/10.48550/arXiv.1810.00826)

