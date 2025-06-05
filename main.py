import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import re
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1. Load Input Data ===
train_df = pd.read_csv('data/training_set.csv')
validat_df = pd.read_csv('data/validation_set.csv')
taxonomy_df = pd.read_csv('data/taxonomy.csv')
predict_df = validat_df.drop(columns=['COMMODITY_LEAF']).copy()

# === 2. Preprocessing Helpers ===
def combine_fields(df, weight=1):
    """Combine relevant fields into a single string for modeling."""
    df = df.fillna('')
    return (
        df['description'].astype(str) * weight + ' ' +
        df['buy_line_id'].astype(str) + ' ' +
        df['price_line_id'].astype(str) + ' ' +
        df['keywords'].astype(str) + ' ' +
        df['product_family_id'].astype(str) + ' ' +
        df['product_family_description'].astype(str)
    )

def clean_text(text):
    """Lowercase and clean special characters and noise words."""
    text = text.lower()
    noise_words = ['webready']
    for word in noise_words:
        text = re.sub(word, '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# === 3. Train & Evaluate Models for Different Description Weights ===
metrics_by_weight = {}
predictions = {}

for weight in [1, 3, 5]:
    train_df['combined_text'] = combine_fields(train_df, weight).apply(clean_text)
    validat_df['combined_text'] = combine_fields(validat_df, weight).apply(clean_text)

    X_train = train_df['combined_text']
    y_train = train_df['COMMODITY_LEAF']
    X_val = validat_df['combined_text']
    y_val = validat_df['COMMODITY_LEAF']

    model_path = f'data/training_result_w{weight}.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    y_conf = y_proba.max(axis=1)

    metrics_by_weight[weight] = [
        accuracy_score(y_val, y_pred),
        precision_score(y_val, y_pred, average='macro'),
        recall_score(y_val, y_pred, average='macro'),
        f1_score(y_val, y_pred, average='macro')
    ]
    predictions[weight] = (y_pred, y_conf)

# === 4. Plot: Model Metrics by Weight with Benchmark Table ===

labels = ['Accuracy', 'Precision', 'Recall', 'F1']
benchmarks = [0.80, 0.75, 0.75, 0.75]
x = np.arange(len(labels))
bar_width = 0.2

# Set up side-by-side layout
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# --- Left: Bar Chart ---
ax0 = fig.add_subplot(gs[0])
for i, (w, scores) in enumerate(metrics_by_weight.items()):
    positions = x + i * bar_width
    bars = ax0.bar(positions, scores, bar_width, label=f'Weight {w}')
    for bar in bars:
        height = bar.get_height()
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax0.set_xticks(x + bar_width)
ax0.set_xticklabels(labels)
ax0.set_ylim(0, 1.05)
ax0.set_ylabel('Score (0–1)')
ax0.set_title('Classification Metrics by Description Weight\n(Higher is better)')
ax0.legend(title='Weight')

# --- Right: Reference Table ---
ax1 = fig.add_subplot(gs[1])
ax1.axis('off')  # Hide the axis

# Create the reference table
table_data = [
    ['Metric', 'Typical Range', 'Benchmark', 'Notes'],
    ['Accuracy', '0.70–0.95', '≥ 0.80', 'Correct predictions overall'],
    ['Precision', '0.60–0.95', '≥ 0.75', 'How often predictions are correct'],
    ['Recall', '0.60–0.95', '≥ 0.75', 'Coverage of true classes'],
    ['F1 Score', '0.60–0.95', '≥ 0.75', 'Balance of precision & recall']
]

table = ax1.table(cellText=table_data, cellLoc='left', colWidths=[0.2, 0.2, 0.2, 0.5], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

plt.tight_layout()
plt.show()

# === Data Prep ===
train_indices, class_labels = pd.factorize(train_df['COMMODITY_LEAF'])
val_indices = pd.Series(validat_df['COMMODITY_LEAF']).map({label: i for i, label in enumerate(class_labels)})

train_counts = pd.Series(train_indices).value_counts().sort_index()
val_counts = val_indices.value_counts().sort_index().reindex(train_counts.index, fill_value=0)

x = range(1, len(train_counts) + 1)
bar_width = 0.4
x_ticks = list(range(0, len(train_counts) + 10, 10))

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(14, 5))
plt.subplots_adjust(bottom=0.2)

bars1 = ax.bar([i - bar_width/2 for i in x], train_counts.values, width=bar_width, label='Train', color='steelblue', edgecolor='k')
bars2 = ax.bar([i + bar_width/2 for i in x], val_counts.values, width=bar_width, label='Validation', color='firebrick', edgecolor='k')

ax.set_xlabel('Factorized Class Index')
ax.set_ylabel('Number of Samples')
ax.set_title('Training vs Validation Set Class Distribution')
ax.set_xticks(x_ticks)
ax.set_xlim(0.5, 100.5)  # Show first 100 classes
ax.legend()

# === Slider for Horizontal Scrolling ===
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 'Scroll', 1, len(train_counts) - 100, valinit=1, valstep=1)

def update(val):
    start = int(slider.val)
    ax.set_xlim(start, start + 100)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.tight_layout()
plt.show()

# === 5. Scatter Plot: True vs Predicted (Weight = 3) Highlighting Outliers ===

# Extract predictions
y_pred, y_conf = predictions[1]
y_val_num, y_val_labels = pd.factorize(validat_df['COMMODITY_LEAF'])
y_pred_num = pd.Series(y_pred).map({label: idx for idx, label in enumerate(y_val_labels)}).fillna(-1)

# Compute regression errors
errors = np.abs(y_val_num - y_pred_num)
mae = np.mean(errors)
mse = np.mean((y_val_num - y_pred_num) ** 2)
rmse = np.sqrt(mse)

# Define outliers: absolute error > threshold
outlier_threshold = 5
is_outlier = errors > outlier_threshold
colors = np.where(is_outlier, 'red', 'lightgray')

# Layout setup
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# --- Left: Scatter Plot ---
ax0 = fig.add_subplot(gs[0])
sc = ax0.scatter(
    y_val_num, y_pred_num,
    c=colors, alpha=0.6, edgecolor='k', linewidth=0.2
)
ax0.plot([min(y_val_num), max(y_val_num)], [min(y_val_num), max(y_val_num)], 'gray', linestyle='--')

# Highlight formatting
ax0.set_xlabel('True Class Index')
ax0.set_ylabel('Predicted Class Index')
ax0.set_title(f'Scatter Plot: True vs Predicted Class Index (Weight=1)\nOutliers = error > {outlier_threshold}')
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor='red', markersize=8, markeredgecolor='k'),
    plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='lightgray', markersize=8, markeredgecolor='k')
]
ax0.legend(handles=legend_handles)

# --- Right: Metrics Table ---
ax1 = fig.add_subplot(gs[1])
ax1.axis('off')

table_data = [
    ['Metric', 'Value', 'Standard', 'Explanation'],
    ['MAE', f'{mae:.2f}', '↓ 0.50', 'Avg. absolute error'],
    ['MSE', f'{mse:.2f}', '↓ 0.50', 'Avg. squared error'],
    ['RMSE', f'{rmse:.2f}', '↓ 0.70', 'Root mean square error'],
    ['Outliers', f'{is_outlier.sum()}', f'< {len(is_outlier) * 0.05:.0f}', 'Pred error > 5']
]

table = ax1.table(cellText=table_data, cellLoc='left', colWidths=[0.18, 0.15, 0.15, 0.52], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

plt.tight_layout()
plt.show()

# === 6. Hierarchy-Level Evaluation ===
# Merge true and predicted hierarchy levels
validat_df['COMMODITY_LEAF_PREDICTED'] = y_pred
validat_df['PREDICTION_CONFIDENCE'] = y_conf

taxonomy_true = taxonomy_df.rename(columns={col: f"{col}_TRUE" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})
taxonomy_pred = taxonomy_df.rename(columns={col: f"{col}_PRED" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})

merged = validat_df.merge(taxonomy_true, on='COMMODITY_LEAF', how='left')
merged = merged.merge(taxonomy_pred, left_on='COMMODITY_LEAF_PREDICTED', right_on='COMMODITY_LEAF', how='left', suffixes=('', '_DROP'))
merged.drop(columns=[col for col in merged.columns if col.endswith('_DROP')], inplace=True)

# === Hierarchy Evaluation Setup ===
levels = [1, 2, 3, 4]
metrics_by_level = {}
confidence_bins = np.arange(0, 1.1, 0.1)
bin_labels = [f'{b:.1f}-{b + 0.1:.1f}' for b in confidence_bins[:-1]]

# Containers for histogram + % plots
all_hist_data = []
all_percent_data = []

# === Collect per-level data ===
for level in levels:
    true_col = f'COMMODITY CODE_Level {level}_TRUE'
    pred_col = f'COMMODITY CODE_Level {level}_PRED'
    mask = merged[true_col].notna() & merged[pred_col].notna()
    y_true = merged.loc[mask, true_col]
    y_pred = merged.loc[mask, pred_col]
    conf = merged.loc[mask, 'PREDICTION_CONFIDENCE']
    correct = y_true == y_pred

    # --- 1. Metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1s = f1_score(y_true, y_pred, average='macro')
    metrics_by_level[f'Level {level}'] = [acc, prec, rec, f1s]

    # --- 2. Histogram Data ---
    bin_indices = np.digitize(conf, confidence_bins) - 1
    hist_df = pd.DataFrame({
        'bin': [bin_labels[i] for i in bin_indices],
        'correct': correct
    })

    hist_counts = hist_df.groupby(['bin', 'correct']).size().unstack(fill_value=0)
    all_hist_data.append((level, hist_counts))  # Save raw counts

    # --- 3. % Correct vs Incorrect ---
    hist_percent = hist_counts.div(hist_counts.sum(axis=1), axis=0) * 100
    hist_percent = hist_percent.reindex(columns=[False, True], fill_value=0)
    all_percent_data.append((level, hist_percent))  # Save percentages

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# === Plot: Metric Comparison by Hierarchy Level with Benchmark Table (Weight = 1) ===
labels = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(labels))
bar_width = 0.2

# Create layout: bar plot (left) + table (right)
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# --- Left: Metrics Bar Plot ---
ax0 = fig.add_subplot(gs[0])
for i, (lvl, scores) in enumerate(metrics_by_level.items()):
    bars = ax0.bar(x + i * bar_width, scores, width=bar_width, label=lvl)
    for bar in bars:
        height = bar.get_height()
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax0.set_xticks(x + bar_width * 1.5)
ax0.set_xticklabels(labels)
ax0.set_ylim(0, 1.05)
ax0.set_ylabel('Score (0–1)')
ax0.set_title('Metrics by Hierarchy Level (Prediction from Weight = 1)')
ax0.legend(title='Hierarchy Level')

# --- Right: Reference Table ---
ax1 = fig.add_subplot(gs[1])
ax1.axis('off')

table_data = [
    ['Metric', 'Typical Range', 'Benchmark', 'Notes'],
    ['Accuracy', '0.70–0.95', '≥ 0.80', 'Correct predictions overall'],
    ['Precision', '0.60–0.95', '≥ 0.75', 'How often predictions are correct'],
    ['Recall', '0.60–0.95', '≥ 0.75', 'Coverage of true classes'],
    ['F1 Score', '0.60–0.95', '≥ 0.75', 'Balance of precision & recall']
]

table = ax1.table(cellText=table_data, cellLoc='left', colWidths=[0.2, 0.2, 0.2, 0.5], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

plt.tight_layout()
plt.show()

for level, hist_counts in all_hist_data:
    hist_percent = dict(all_percent_data)[level]

    # Ensure correct=True is always first in the stacking
    hist_counts = hist_counts.reindex(columns=[True, False], fill_value=0)
    hist_percent = hist_percent.reindex(columns=[True, False], fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Histogram (Top) ---
    bins = hist_counts.index
    correct_vals = hist_counts[True].values
    incorrect_vals = hist_counts[False].values
    total_vals = correct_vals + incorrect_vals

    bars_correct = axes[0].bar(bins, correct_vals, color='green', label='Correct')
    bars_incorrect = axes[0].bar(bins, incorrect_vals, bottom=correct_vals, color='red', label='Incorrect')

    # Add count inside correct bar (centered)
    for bar, count in zip(bars_correct, correct_vals):
        if count > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                str(int(count)),
                ha='center', va='center',
                fontsize=9, color='white'
            )

    # Add % label above full bar
    for i, total in enumerate(total_vals):
        if total > 0:
            axes[0].text(
                i, total + max(total_vals) * 0.02,
                f'{(total / total_vals.sum() * 100):.1f}%',
                ha='center', va='bottom',
                fontsize=9
            )


    axes[0].set_title(f'Level {level}: Prediction Count by Confidence')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # --- % Breakdown (Bottom) ---
    # Stack with correct at the bottom
    hist_percent = hist_percent.reindex(columns=[False, True], fill_value=0)
    # Incorrect on top, correct on bottom
    hist_percent.plot(
        kind='bar', stacked=True, color=['red', 'green'], ax=axes[1]
    )

    # --- Add value labels to green (Correct) section of % bar plot ---
    # Get the green (correct) bars
    correct_vals = hist_percent[True].values
    bars = axes[1].containers[1]  # container[1] is the second (green/correct) stack

    for bar, val in zip(bars, correct_vals):
        if val > 0:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%',
                ha='center',
                va='center',
                fontsize=9,
                color='white' if val > 15 else 'black'  # readable color
            )

    axes[1].set_title(f'Level {level}: % Correct vs Incorrect by Confidence')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xlabel('Confidence Bin')
    axes[1].legend(['Incorrect', 'Correct'])
    axes[1].set_xticklabels(hist_percent.index, rotation=45)

    plt.tight_layout()
    plt.show()

# === Export Final Predictions ===
# Start from predict_df and merge true labels from validat_df
prediction_set = predict_df.copy()
prediction_set['COMMODITY_LEAF'] = validat_df['COMMODITY_LEAF']
prediction_set['COMMODITY_LEAF_PREDICTED'] = y_pred
prediction_set['PREDICTION_CONFIDENCE'] = y_conf

# Export the results
prediction_set[['PRODUCT_ID', 'description', 'COMMODITY_LEAF', 'COMMODITY_LEAF_PREDICTED', 'PREDICTION_CONFIDENCE']]\
    .to_csv('data/prediction_set.csv', index=False)

print("✅ Final predictions exported to 'data/prediction_set.csv'")




