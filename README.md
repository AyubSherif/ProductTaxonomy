# 🔌 Product Taxonomy Classifier

> **🔒 Disclaimer**
> This is a simplified public version and does not reflect the full internal deployment used at Loeb Electric.

At **Loeb Electric**, we rely on third-party product content syndication services to enrich and categorize our product data. However, some products — particularly wires and related electrical components — return with no matches. This tool addresses that gap.

---

## 📌 Problem Description

- Incomplete product classification from syndication services
- Manual categorization is slow, inconsistent, and labor-intensive
- Internal analytics and e-commerce workflows require accurate product taxonomy

---

## 🎯 Project Objective

Build a supervised machine learning system to classify electrical products into a **4-level hierarchy**, using structured and unstructured text fields like descriptions, product family, keywords, etc.

### Category Levels
| Level | Description    | Example                       |
|-------|----------------|-------------------------------|
| 1     | Type           | Wire, Cable, Conduit          |
| 2     | Use Case       | Building, Welding, Control    |
| 3     | Attributes     | THHN, XHHW, Armored, Tray     |
| 4     | Specs          | 12 AWG, Copper, 500ft, Spool  |

---

## 🧱 Modular Pipeline Overview

The project is structured into clearly defined modules:

- `text_pipeline.py` — Load and preprocess text fields
- `model_training.py` — Train logistic regression models using different field weights
- `analysis_text_weight.py` — Analyze and visualize how description weighting affects performance
- `analysis_hierarchy.py` — Evaluate and visualize prediction accuracy across hierarchy levels
- `export_results.py` — Export final predictions with confidence and true labels
- `main.py` — Pipeline entry point: coordinates everything

---

## 📊 Visual Outputs

The project includes:
- Metric comparison by weight (accuracy, precision, recall, F1)
- Training vs validation class distribution
- Scatter plot of predicted vs actual class indices
- Hierarchy-level confidence analysis and accuracy heatmaps

---

## 📂 Key Input Files

- `data/training_set.csv` — Labeled training samples
- `data/validation_set.csv` — Labeled validation set
- `data/taxonomy.csv` — Mapping of class labels to 4-level hierarchy

---

## 🚀 How to Run

1. Clone the repo and set up a virtual environment
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Run the main pipeline:
   ```bash
   python main.py
   ```

Outputs will be saved to `data/prediction_set.csv` and visualizations will display inline.

---

## ✅ Example Output Columns

| PRODUCT_ID | description              | COMMODITY_LEAF | COMMODITY_LEAF_PREDICTED | PREDICTION_CONFIDENCE |
|------------|---------------------------|----------------|---------------------------|------------------------|
| 12345      | 12 AWG THHN Copper Spool  | Wire_THHN_12    | Wire_THHN_12              | 0.93                   |

---

## 📬 Contact
For internal access or full production model support, contact the data team at Loeb Electric.