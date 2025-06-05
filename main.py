import pandas as pd
import re
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load Data ===
train_df = pd.read_csv('data/training_set.csv')
validat_df = pd.read_csv('data/validation_set.csv')
taxonomy_df = pd.read_csv('data/taxonomy.csv')
predict_df = validat_df.drop(columns=['COMMODITY_LEAF']).copy()

# === 2. Combine Fields with Field Weighting ===
def combine_fields(df, n=1):
    df = df.fillna('')
    return (
        df['description'].astype(str) * n + ' ' +
        df['buy_line_id'].astype(str) + ' ' +
        df['price_line_id'].astype(str) + ' ' +
        df['keywords'].astype(str) + ' ' +
        df['product_family_id'].astype(str) + ' ' +
        df['product_family_description'].astype(str)
    )

# === 3. Clean Text ===
def clean_text(text):
    noise_phrases = ['webready', 'n/a', 'none']
    text = str(text).lower()
    for phrase in noise_phrases:
        text = text.replace(phrase, '')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['combined_text'] = combine_fields(train_df).apply(clean_text)
predict_df['combined_text'] = combine_fields(predict_df).apply(clean_text)
validat_df['combined_text'] = combine_fields(validat_df).apply(clean_text)

# === 4. Train or Load Model ===
X_train = train_df['combined_text']
y_train = train_df['COMMODITY_LEAF']
model_path = 'data/training_result.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# === 5. Predict and Evaluate ===
X_val = validat_df['combined_text']
y_val = validat_df['COMMODITY_LEAF']
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val).max(axis=1)

# === 6. Metrics ===
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')

# Convert to numeric class indices for regression metrics
y_val_num = pd.factorize(y_val)[0]
y_pred_num = pd.factorize(y_pred)[0]
mae = mean_absolute_error(y_val_num, y_pred_num)
mse = mean_squared_error(y_val_num, y_pred_num)
rmse = np.sqrt(mse)

# === 7. Plot Combined ===
plt.figure(figsize=(12, 8))

# Classification metrics
plt.subplot(2, 1, 1)
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [accuracy, precision, recall, f1], color='skyblue')
plt.title('Classification Metrics\nHigher is better (Range: 0 to 1). Common benchmarks: Accuracy > 0.85, F1 > 0.80')
plt.ylim(0, 1.05)
for i, val in enumerate([accuracy, precision, recall, f1]):
    plt.text(i, val + 0.02, f"{val:.2f}", ha='center')

# Scatter plot of predicted vs actual indices
plt.subplot(2, 1, 2)
plt.scatter(y_val_num, y_pred_num, alpha=0.6, c='mediumseagreen')
plt.plot([min(y_val_num), max(y_val_num)], [min(y_val_num), max(y_val_num)], color='gray', linestyle='--')
plt.title(f'Scatter Plot: True vs Predicted Class Indices\nMAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}')
plt.xlabel('True Class Index')
plt.ylabel('Predicted Class Index')

plt.tight_layout()
plt.show()

# === 8. Evaluate per Hierarchy Level ===
taxonomy_true = taxonomy_df.rename(columns={col: f"{col}_TRUE" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})
taxonomy_pred = taxonomy_df.rename(columns={col: f"{col}_PRED" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})

merged = validat_df.copy()
merged['COMMODITY_LEAF_PREDICTED'] = y_pred
merged['PREDICTION_CONFIDENCE'] = y_pred_proba

merged = merged.merge(taxonomy_true, on='COMMODITY_LEAF', how='left')
merged = merged.merge(taxonomy_pred, left_on='COMMODITY_LEAF_PREDICTED', right_on='COMMODITY_LEAF', how='left', suffixes=('', '_DROP'))
merged.drop([col for col in merged.columns if col.endswith('_DROP')], axis=1, inplace=True)

levels = [1, 2, 3, 4]
for level in levels:
    true_col = f'COMMODITY CODE_Level {level}_TRUE'
    pred_col = f'COMMODITY CODE_Level {level}_PRED'

    level_mask = merged[true_col].notna() & merged[pred_col].notna()
    y_true_lvl = merged.loc[level_mask, true_col]
    y_pred_lvl = merged.loc[level_mask, pred_col]
    conf_lvl = merged.loc[level_mask, 'PREDICTION_CONFIDENCE']
    correct_lvl = y_true_lvl == y_pred_lvl

    acc = accuracy_score(y_true_lvl, y_pred_lvl)
    prec = precision_score(y_true_lvl, y_pred_lvl, average='macro')
    rec = recall_score(y_true_lvl, y_pred_lvl, average='macro')
    f1s = f1_score(y_true_lvl, y_pred_lvl, average='macro')

    print(f"\n--- Level {level} ---")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1s:.2f}")

    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]]
    bin_indices = np.digitize(conf_lvl, bins) - 1
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

    axes[0].hist([conf_lvl[correct_lvl], conf_lvl[~correct_lvl]], bins=10, range=(0, 1), stacked=True,
                 label=['Correct', 'Incorrect'], color=['green', 'red'], alpha=0.7)
    axes[0].set_title(f'Level {level} - Confidence Score Distribution')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].set_xticks(range(len(bin_labels)))
    axes[0].set_xticklabels(bin_labels)

    hist_df = pd.DataFrame({
        'correct': correct_lvl,
        'bin': [bin_labels[i] for i in bin_indices]
    })
    sum_df = hist_df.groupby(['bin', 'correct']).size().unstack(fill_value=0)
    sum_percent = sum_df.div(sum_df.sum(axis=1), axis=0) * 100

    sum_percent.plot(kind='bar', stacked=True, color=['red', 'green'], ax=axes[1])
    axes[1].set_title(f'Level {level} - % Correct vs Incorrect per Confidence Bin')
    axes[1].set_xlabel('Confidence Bin')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(['Incorrect', 'Correct'])
    axes[1].set_xticks(range(len(bin_labels)))
    axes[1].set_xticklabels(bin_labels, rotation=45)
    plt.tight_layout()
    plt.show()

# === 9. Finalize Export ===
predict_df['COMMODITY_LEAF'] = validat_df['COMMODITY_LEAF']
predict_df['COMMODITY_LEAF_PREDICTED'] = y_pred
predict_df['PREDICTION_CONFIDENCE'] = y_pred_proba
predict_df[['PRODUCT_ID', 'description', 'COMMODITY_LEAF', 'COMMODITY_LEAF_PREDICTED', 'PREDICTION_CONFIDENCE']].to_csv('data/prediction_set.csv', index=False)
