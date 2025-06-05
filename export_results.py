# === file: export_results.py ===
# Export final prediction results to CSV

import pandas as pd

def export_predictions(validat_df, y_pred, y_conf):
    prediction_set = validat_df.copy()
    prediction_set['COMMODITY_LEAF_PREDICTED'] = y_pred
    prediction_set['PREDICTION_CONFIDENCE'] = y_conf

    prediction_set[['PRODUCT_ID', 'description', 'COMMODITY_LEAF', 'COMMODITY_LEAF_PREDICTED', 'PREDICTION_CONFIDENCE']].to_csv('data/prediction_set.csv', index=False)
