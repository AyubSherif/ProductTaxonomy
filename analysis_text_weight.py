# === file: analysis_text_weight.py ===
# Analysis and visualization of model performance by text weight

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

# --- Metric Comparison Plot ---
def plot_metrics_by_weight(metrics_by_weight):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    benchmarks = [0.80, 0.75, 0.75, 0.75]
    x = np.arange(len(labels))
    bar_width = 0.2

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0])
    for i, (w, scores) in enumerate(metrics_by_weight.items()):
        positions = x + i * bar_width
        bars = ax0.bar(positions, scores, bar_width, label=f'Weight {w}')
        for bar in bars:
            height = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)

    ax0.set_xticks(x + bar_width)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Score (0–1)')
    ax0.set_title('Classification Metrics by Description Weight\n(Higher is better)')
    ax0.legend(title='Weight')

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

# --- Class Distribution Plot ---
def plot_class_distributions(train_df, validat_df):
    train_indices, class_labels = pd.factorize(train_df['COMMODITY_LEAF'])
    val_indices = pd.Series(validat_df['COMMODITY_LEAF']).map({label: i for i, label in enumerate(class_labels)})

    train_counts = pd.Series(train_indices).value_counts().sort_index()
    val_counts = val_indices.value_counts().sort_index().reindex(train_counts.index, fill_value=0)

    x = range(1, len(train_counts) + 1)
    bar_width = 0.4
    x_ticks = list(range(0, len(train_counts) + 10, 10))

    fig, ax = plt.subplots(figsize=(14, 5))
    plt.subplots_adjust(bottom=0.2)

    ax.bar([i - bar_width/2 for i in x], train_counts.values, width=bar_width, label='Train', color='steelblue', edgecolor='k')
    ax.bar([i + bar_width/2 for i in x], val_counts.values, width=bar_width, label='Validation', color='firebrick', edgecolor='k')

    ax.set_xlabel('Factorized Class Index')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Training vs Validation Set Class Distribution')
    ax.set_xticks(x_ticks)
    ax.set_xlim(0.5, 100.5)
    ax.legend()

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, 'Scroll', 1, len(train_counts) - 100, valinit=1, valstep=1)

    def update(val):
        start = int(slider.val)
        ax.set_xlim(start, start + 100)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

# --- Outlier Scatter Plot ---
def plot_scatter_with_outliers(validat_df, prediction_tuple):
    y_pred, y_conf = prediction_tuple
    y_val_num, y_val_labels = pd.factorize(validat_df['COMMODITY_LEAF'])
    y_pred_num = pd.Series(y_pred).map({label: idx for idx, label in enumerate(y_val_labels)}).fillna(-1)

    errors = np.abs(y_val_num - y_pred_num)
    mae = np.mean(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    outlier_threshold = 5
    is_outlier = errors > outlier_threshold
    colors = np.where(is_outlier, 'red', 'lightgray')

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    sc = ax0.scatter(y_val_num, y_pred_num, c=colors, alpha=0.6, edgecolor='k', linewidth=0.2)
    ax0.plot([min(y_val_num), max(y_val_num)], [min(y_val_num), max(y_val_num)], 'gray', linestyle='--')

    ax0.set_xlabel('True Class Index')
    ax0.set_ylabel('Predicted Class Index')
    ax0.set_title(f'Scatter Plot: True vs Predicted Class Index (Weight=1)\nOutliers = error > {outlier_threshold}')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor='red', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='lightgray', markersize=8, markeredgecolor='k')
    ]
    ax0.legend(handles=legend_handles)

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
