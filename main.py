# === file: main.py ===
# Main entry point: handles I/O, configuration, and orchestration

from text_pipeline import load_data
from model_training import train_models_and_collect_metrics
from analysis_text_weight import plot_metrics_by_weight, plot_class_distributions, plot_scatter_with_outliers
from analysis_hierarchy import evaluate_hierarchy_levels, plot_hierarchy_metrics, plot_confidence_distributions
from export_results import export_predictions

def main():
    # Load datasets
    train_df, validat_df, taxonomy_df, predict_df = load_data()

    # Train models for weights 1, 3, 5
    metrics_by_weight, predictions = train_models_and_collect_metrics(train_df, validat_df)

    # Visualizations
    plot_metrics_by_weight(metrics_by_weight)
    plot_class_distributions(train_df, validat_df)
    plot_scatter_with_outliers(validat_df, predictions[1])  # Weight 1

    # Hierarchy-level evaluation and visualization
    y_pred, y_conf = predictions[1]  # Weight 1
    metrics_by_level, all_hist_data, all_percent_data = evaluate_hierarchy_levels(validat_df, taxonomy_df, y_pred, y_conf)
    plot_hierarchy_metrics(metrics_by_level)
    plot_confidence_distributions((metrics_by_level, all_hist_data, all_percent_data))

    # Export final predictions
    export_predictions(predict_df, validat_df, y_pred, y_conf)
    print("✅ Final predictions exported to 'data/prediction_set.csv'")

if __name__ == '__main__':
    main()
