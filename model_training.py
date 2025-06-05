# === file: model_training.py ===
# Train models with varying text weights and collect performance metrics

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from text_pipeline import combine_fields, clean_text

def train_models_and_collect_metrics(train_df, validat_df):
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

    return metrics_by_weight, predictions
