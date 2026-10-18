# Sentiment Analysis Dashboard

A Streamlit dashboard for exploring sentiment in Amazon customer reviews for cell phones and accessories. The app uses review ratings to label reviews as **Negative**, **Neutral**, or **Positive**, then trains text-classification models and presents interactive analyses.

## Features

- Review overview with positive, neutral, and negative percentages
- Rating distribution and average ratings over time
- Product-level sentiment comparisons
- Text cleaning and TF-IDF feature extraction
- Three sentiment classifiers:
  - Logistic Regression
  - Linear Support Vector Classifier (Linear SVC)
  - Multinomial Naive Bayes
- Model evaluation using accuracy, classification reports, and confusion matrices
- Word-level contribution views based on the Logistic Regression model
- Filters for sentiment, year, and product (ASIN)

## Technologies

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- Altair and Plotly
- Matplotlib and Seaborn

## Project files

Place these files in the same directory:

```text
project/
├── app.py
├── Cell_Phones_and_Accessories_5.json
├── requirements.txt
└── README.md
```

The dataset is read from `Cell_Phones_and_Accessories_5.json` in JSON Lines format. Keep the filename unchanged unless you also update the path in `app.py`.

## Installation

Python 3.10 or newer is a reasonable starting point. From the project directory, create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard

From the directory containing `app.py` and the dataset, run:

```bash
streamlit run app.py
```

Streamlit will print a local address in the terminal (usually `http://localhost:8501`) where you can open the dashboard in your browser.

## Dataset requirements

The app expects the dataset to contain these fields:

- `overall`
- `reviewText`
- `summary`
- `reviewTime`
- `asin`
- `reviewerID`
- `helpful`

The app derives sentiment labels from the rating: 1–2 stars are Negative, 3 stars are Neutral, and 4–5 stars are Positive.


