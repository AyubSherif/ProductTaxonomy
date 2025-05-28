import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read data
train_df = pd.read_csv('data/products_catogarized.csv')
predict_df = pd.read_csv('data/products.csv')

# Combine relevant fields into a single text string
def combine_fields(df):
    fields = [
        'description'
    ]
    df = df.fillna('')
    return df[fields].astype(str).agg(' '.join, axis=1)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['combined_text'] = combine_fields(train_df).apply(clean_text)
predict_df['combined_text'] = combine_fields(predict_df).apply(clean_text)

# Train classifier
def train_text_classifier(X, y):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

print("Training COMMODITY_LEAF classifier...")
X = train_df['combined_text']
y = train_df['COMMODITY_LEAF']
model = train_text_classifier(X, y)

# Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict
predict_df['COMMODITY_LEAF_PREDICTED'] = model.predict(predict_df['combined_text'])

# Export result
predict_df[['description', 'COMMODITY_LEAF_PREDICTED']].to_csv('products_categorized_output3.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt

# Get classification report as dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame and drop accuracy, macro avg, weighted avg
report_df = pd.DataFrame(report_dict).drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)

# Transpose to make classes rows
report_df = report_df.T[['precision', 'recall', 'f1-score']]

# Plot heatmap
plt.figure(figsize=(10, len(report_df) * 0.4 + 2))
sns.heatmap(report_df, annot=True, cmap='Blues', fmt=".2f")
plt.title('Classification Report Heatmap')
plt.ylabel('Class Label (COMMODITY_LEAF)')
plt.xlabel('Metric')
plt.tight_layout()
plt.show()