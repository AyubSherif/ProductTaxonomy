# === file: main.py ===
# Main entry point: handles I/O, configuration, and orchestration

import os
import joblib
from text_pipeline import load_data
from model_training import train_models_and_collect_metrics, train_model
from analysis_text_weight import plot_metrics_by_weight, plot_class_distributions, plot_scatter_with_outliers
from analysis_hierarchy import evaluate_hierarchy_levels, plot_hierarchy_metrics, plot_confidence_distributions
from export_results import export_predictions


def main():
    # Load datasets
    train_df, validat_df, taxonomy_df = load_data()

    # Prompt user for which description weights to evaluate
    description_weights = [1]
    models_to_train = []

    for weight in description_weights:
        model_path = f'data/training_result_w{weight}.pkl'
        if os.path.exists(model_path):
            user_choice = input(f"Model for weight {weight} already exists. Use existing model? (y/n): ").strip().lower()
            if user_choice == 'n':
                models_to_train.append(weight)
            else:
                print(f"✔️ Using existing model for weight {weight}.")
        else:
            models_to_train.append(weight)

    # If user chose to retrain any models
    if models_to_train:
        for weight in models_to_train:
            model = train_model(train_df.copy(), weight)
            model_path = f'data/training_result_w{weight}.pkl'
            joblib.dump(model, model_path)
            print(f"✅ Model for weight {weight} saved to {model_path}")

    # Load + evaluate all models (training or existing)
    metrics_by_weight, predictions = train_models_and_collect_metrics(train_df, validat_df, description_weights)

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
    export_predictions(validat_df, y_pred, y_conf)
    print("✅ Final predictions exported to 'data/prediction_set.csv'")


if __name__ == '__main__':
    main()
