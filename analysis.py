# 1. Install required libraries
# !pip install pandas transformers matplotlib seaborn

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset (ensure file is named data.csv)
df = pd.read_csv('data.csv')

# 3. Load AI sentiment analysis model
classifier = pipeline("sentiment-analysis")

# 4. Define function for analysis
def get_sentiment(text):
    result = classifier(text)[0]
    # Return score: Positive (+score) or Negative (-score)
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    return score

# Apply analysis to each sentence
df['score'] = df['sentence'].apply(get_sentiment)

# 5. Data Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='score', data=df, palette='viridis')
plt.title('AI Sentiment Bias: Male vs Female')
plt.ylabel('Sentiment Score (Positive to Negative)')
plt.show()

# 6. Print average metrics for conclusions
print(df.groupby('gender')['score'].mean())
