# Investigating Gender Bias in Sentiment Analysis Models (DistilBERT)

## Overview
This repository contains the methodology, dataset metrics, and experimental findings for a research project investigating potential gender bias in modern Natural Language Processing (NLP) models.

## Key Findings & Results
The empirical evaluation was conducted using the fine-tuned **DistilBERT** architecture (`distilbert-base-uncased-finetuned-sst-2-english`) across a controlled dataset of 56 paired statements containing gender markers.

* **Female Group Average Confidence Score:** $0.98$
* **Male Group Average Confidence Score:** $0.97$

**Conclusion:** The negligible difference of $0.01$ is statistically insignificant. This demonstrates that the model does not exhibit significant gender bias and maintains high semantic neutrality during sentiment classification.

## Repository Structure
- **Dataset:** Contains the test prompts and corresponding confidence scores.
- **Visualizations:** Performance comparison charts.

## Tools & Technologies
* Python
* Hugging Face Transformers
* Pandas, Matplotlib / Google Sheets

*Original Google Doc Report & Full Spreadshit:* [https://docs.google.com/spreadsheets/d/14IR5UQr8JqlcMm5CTGloOhLyCX2iMcM7/edit?usp=drivesdk&ouid=108445358113271259101&rtpof=true&sd=true]

- **Source Code:** You can inspect the full implementation pipeline in [`analysis.py`](./analysis.py).
