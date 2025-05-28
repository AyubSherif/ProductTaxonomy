# 🔌 Product Taxonomy Classifier

> **🔒 Disclaimer**
> This is just a sample version of the tool and does not reflect the full functionality or scale of the internal version deployed at Loeb Electric.

At **Loeb Electric**, we rely on a 3rd party product content syndication service to enrich and categorize our product data. However, **some products consistently return with no matches** from that service — often due to inconsistent naming conventions, lack of industry-standard identifiers, or missing metadata.

To address this gap, I developed a custom **Product taxonomy classification tool** that uses a simple a supervised machine learning model to automatically categorize products into a 4 level taxonomy. This sample version showcases the core functionality — the full production-grade implementation is hosted on the company’s private GitHub.

---

## 📌 Problem Description

- Many products lack enriched metadata from our syndication partner.
- Manual categorization is labor-intensive.
- We needed a scalable and automated solution to support internal analytics and e-commerce workflows.

---

## 🎯 Project Objective

Build a supervised machine learning model that can classify products into **4 consistent hierarchical levels**, using only the product name or basic attributes:

### Category Levels
| Level | Description    | Example                       |
|-------|----------------|-------------------------------|
| 1     | Type           | Wire, Cable, Conduit          |
| 2     | Use Case       | Building, Welding, Control    |
| 3     | Attributes     | THHN, XHHW, Armored, Tray     |
| 4     | Specs          | 12 AWG, Copper, 500ft, Spool  |

---

## 📂 Files in This Sample

- `training_set.csv` — Labeled training data with known classifications
- `unclassified_set.csv` — Unclassified product names to be categorized
- `main.py` — Python script to train and predict taxonomy levels
- `classified_set.csv` — Output with predicted categories
- `validation_set.csv` — Labeled data with known classifications to validate predicted classification

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn
