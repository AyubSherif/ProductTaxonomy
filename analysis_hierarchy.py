# === file: analysis_hierarchy.py ===
# Hierarchy-level evaluation and confidence analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_hierarchy_levels(validat_df, taxonomy_df, y_pred, y_conf):
    validat_df = validat_df.copy()
    validat_df['COMMODITY_LEAF_PREDICTED'] = y_pred
    validat_df['PREDICTION_CONFIDENCE'] = y_conf

    taxonomy_true = taxonomy_df.rename(columns={col: f"{col}_TRUE" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})
    taxonomy_pred = taxonomy_df.rename(columns={col: f"{col}_PRED" for col in taxonomy_df.columns if col.startswith('COMMODITY CODE_Level')})

    merged = validat_df.merge(taxonomy_true, on='COMMODITY_LEAF', how='left')
    merged = merged.merge(taxonomy_pred, left_on='COMMODITY_LEAF_PREDICTED', right_on='COMMODITY_LEAF', how='left', suffixes=('', '_DROP'))
    merged.drop(columns=[col for col in merged.columns if col.endswith('_DROP')], inplace=True)

    levels = [1, 2, 3, 4]
    metrics_by_level = {}
    confidence_bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f'{b:.1f}-{b + 0.1:.1f}' for b in confidence_bins[:-1]]
    all_hist_data = []
    all_percent_data = []

    for level in levels:
        true_col = f'COMMODITY CODE_Level {level}_TRUE'
        pred_col = f'COMMODITY CODE_Level {level}_PRED'
        mask = merged[true_col].notna() & merged[pred_col].notna()

        y_true = merged.loc[mask, true_col]
        y_pred = merged.loc[mask, pred_col]
        conf = merged.loc[mask, 'PREDICTION_CONFIDENCE']
        correct = y_true == y_pred

        # Metrics
        metrics_by_level[f'Level {level}'] = [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='macro'),
            recall_score(y_true, y_pred, average='macro'),
            f1_score(y_true, y_pred, average='macro')
        ]

        # Histogram
        bin_indices = np.digitize(conf, confidence_bins) - 1
        hist_df = pd.DataFrame({
            'bin': [bin_labels[i] for i in bin_indices],
            'correct': correct
        })
        hist_counts = hist_df.groupby(['bin', 'correct']).size().unstack(fill_value=0)
        hist_percent = hist_counts.div(hist_counts.sum(axis=1), axis=0) * 100
        hist_counts = hist_counts.reindex(columns=[True, False], fill_value=0)
        hist_percent = hist_percent.reindex(columns=[False, True], fill_value=0)

        all_hist_data.append((level, hist_counts))
        all_percent_data.append((level, hist_percent))

    return metrics_by_level, all_hist_data, all_percent_data


def plot_hierarchy_metrics(metrics_by_level):
    import matplotlib.gridspec as gridspec

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(labels))
    bar_width = 0.2

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0])
    for i, (lvl, scores) in enumerate(metrics_by_level.items()):
        bars = ax0.bar(x + i * bar_width, scores, width=bar_width, label=lvl)
        for bar in bars:
            height = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)

    ax0.set_xticks(x + bar_width * 1.5)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Score (0–1)')
    ax0.set_title('Metrics by Hierarchy Level (Prediction from Weight = 1)')
    ax0.legend(title='Hierarchy Level')

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


def plot_confidence_distributions(hierarchy_eval):
    metrics_by_level, all_hist_data, all_percent_data = hierarchy_eval

    for level, hist_counts in all_hist_data:
        hist_percent = dict(all_percent_data)[level]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Top: Histogram
        bins = hist_counts.index
        correct_vals = hist_counts[True].values
        incorrect_vals = hist_counts[False].values
        total_vals = correct_vals + incorrect_vals

        bars_correct = axes[0].bar(bins, correct_vals, color='green', label='Correct')
        bars_incorrect = axes[0].bar(bins, incorrect_vals, bottom=correct_vals, color='red', label='Incorrect')

        for bar, count in zip(bars_correct, correct_vals):
            if count > 0:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                             str(int(count)), ha='center', va='center', fontsize=9, color='white')

        for i, total in enumerate(total_vals):
            if total > 0:
                axes[0].text(i, total + max(total_vals) * 0.02, f'{(total / total_vals.sum() * 100):.1f}%',
                             ha='center', va='bottom', fontsize=9)

        axes[0].set_title(f'Level {level}: Prediction Count by Confidence')
        axes[0].set_ylabel('Count')
        axes[0].legend()

        # Bottom: % Breakdown
        hist_percent.plot(kind='bar', stacked=True, color=['red', 'green'], ax=axes[1])

        correct_vals = hist_percent[True].values
        bars = axes[1].containers[1]  # green
        for bar, val in zip(bars, correct_vals):
            if val > 0:
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                             f'{val:.1f}%', ha='center', va='center', fontsize=9,
                             color='white' if val > 15 else 'black')

        axes[1].set_title(f'Level {level}: % Correct vs Incorrect by Confidence')
        axes[1].set_ylabel('Percentage')
        axes[1].set_xlabel('Confidence Bin')
        axes[1].legend(['Incorrect', 'Correct'])
        axes[1].set_xticklabels(hist_percent.index, rotation=45)
        plt.tight_layout()
        plt.show()
