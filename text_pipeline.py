# === file: text_pipeline.py ===
# Handles data loading and text preprocessing

import pandas as pd
import re

def load_data():
    train_df = pd.read_csv('data/training_set.csv')
    validat_df = pd.read_csv('data/validation_set.csv')
    taxonomy_df = pd.read_csv('data/taxonomy.csv')
    predict_df = validat_df.drop(columns=['COMMODITY_LEAF']).copy()
    return train_df, validat_df, taxonomy_df, predict_df

def combine_fields(df, weight=1):
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
    text = text.lower()
    text = re.sub(r'webready|n/a|none', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()
