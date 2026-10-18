import re

import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

SENTIMENTS = [
    "Negative",
    "Neutral",
    "Positive"
]

RATING_TO_SENTIMENT = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}


# ============================================================
# FUNCTIONS
# ============================================================

def cleaning_text(text):
    """Clean review text before machine-learning processing."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def sentiment_classification(x):
    """Convert Amazon rating into sentiment."""

    if pd.isna(x):
        return "Neutral"

    if x >= 4:
        return "Positive"

    if x == 3:
        return "Neutral"

    return "Negative"


def contribution_dictionary(x):
    
    if x>0.5:
        
        return "Strong Postive Influence"
    
    if 0.5>=x>=0:
        return "Positive Influence"
    if x==0:
        
        return "Neutral"
    
    if 0>x>=-0.5:
        return "Negative Influence"
    
    if -0.5>x:
        return "Strong Neative Influence"
        

def helpful_ratio(votes):
    """Calculate the helpful-vote ratio."""

    if not isinstance(votes, (list, tuple)):
        return 0.0

    if len(votes) != 2:
        return 0.0

    try:
        helpful = float(votes[0])
        total = float(votes[1])
    except (TypeError, ValueError):
        return 0.0

    if total <= 0:
        return 0.0

    return helpful / total


def create_empty_sentiment_columns(df):
    """Ensure all sentiment columns exist."""

    for sentiment in SENTIMENTS:
        if sentiment not in df.columns:
            df[sentiment] = 0

    return df


def calculate_sentiment_percentages(df):
    """Calculate sentiment percentages by ASIN."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "asin",
                "Negative",
                "Neutral",
                "Positive",
                "Total",
                "negative_%",
                "neutral_%",
                "positive_%"
            ]
        )

    sentiment_by_asin = (
        df.groupby(
            ["asin", "sentiment_classification"]
        )
        .size()
        .unstack(fill_value=0)
    )

    sentiment_by_asin = create_empty_sentiment_columns(
        sentiment_by_asin
    )

    sentiment_by_asin["Total"] = (
        sentiment_by_asin["Negative"]
        + sentiment_by_asin["Neutral"]
        + sentiment_by_asin["Positive"]
    )

    denominator = sentiment_by_asin["Total"].replace(
        0,
        1
    )

    sentiment_by_asin["negative_%"] = (
        100 * sentiment_by_asin["Negative"] / denominator
    ).round(1)

    sentiment_by_asin["neutral_%"] = (
        100 * sentiment_by_asin["Neutral"] / denominator
    ).round(1)

    sentiment_by_asin["positive_%"] = (
        100 * sentiment_by_asin["Positive"] / denominator
    ).round(1)

    return sentiment_by_asin.reset_index()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    try:
        reviews_dataset = pd.read_json(
            "Cell_Phones_and_Accessories_5.json",
            lines=True
        )

    except FileNotFoundError:
        st.error(
            "Could not find 'Cell_Phones_and_Accessories_5.json'. "
            "Make sure the JSON file is in the same folder as app.py."
        )
        st.stop()

    required_columns = [
        "overall",
        "reviewText",
        "summary",
        "reviewTime",
        "asin",
        "reviewerID",
        "helpful"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in reviews_dataset.columns
    ]

    if missing_columns:
        st.error(
            "The dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    reviews_dataset["reviewText"] = (
        reviews_dataset["reviewText"]
        .fillna("")
        .astype(str)
    )

    reviews_dataset["summary"] = (
        reviews_dataset["summary"]
        .fillna("")
        .astype(str)
    )

    reviews_dataset["helpful_ratio"] = (
        reviews_dataset["helpful"]
        .apply(helpful_ratio)
        .round(2)
    )

    reviews_dataset["sentiment_classification"] = (
        reviews_dataset["overall"]
        .apply(sentiment_classification)
    )

    reviews_dataset["review_length"] = (
        reviews_dataset["reviewText"]
        .apply(lambda x: len(str(x).split()))
    )

    reviews_dataset["reviewTime"] = pd.to_datetime(
        reviews_dataset["reviewTime"],
        errors="coerce"
    )

    reviews_dataset["Year"] = (
        reviews_dataset["reviewTime"].dt.year
    )

    reviews_dataset["reviews_combined"] = (
        reviews_dataset["reviewText"]
        + " "
        + reviews_dataset["summary"]
    )

    reviews_dataset["reviews_cleaned"] = (
        reviews_dataset["reviews_combined"]
        .apply(cleaning_text)
    )

    reviews_dataset = reviews_dataset[
        reviews_dataset["reviews_cleaned"].str.strip() != ""
    ].copy()

    return reviews_dataset.reset_index(drop=True)


reviews_dataset = load_data()


# ============================================================
# MACHINE-LEARNING DATA
# ============================================================

X = reviews_dataset["reviews_cleaned"]
y = reviews_dataset["sentiment_classification"]

available_sentiments = sorted(
    y.dropna().unique().tolist()
)

if len(available_sentiments) < 2:
    st.error(
        "The dataset does not contain enough sentiment classes "
        "to train the models."
    )
    st.stop()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

groups = reviews_dataset["asin"]

group_split = GroupShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42
)

# Use one reproducible split
train_idx, test_idx = next(
    group_split.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


st.sidebar.write(
    "The dataset is split at product level using ASIN. "
    "Therefore, no product appears in both the training and "
    "test sets."
)

# ============================================================
# TRAIN MODELS
# ============================================================
@st.cache_resource
def train_models(X_train, y_train):

    model_logistic = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32
            )
        ),
        (
            "Classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    model_svm = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32
            )
        ),
        (
            "Classifier",
            LinearSVC()
        )
    ])

    model_nb = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32
            )
        ),
        (
            "Classifier",
            MultinomialNB()
        )
    ])

    model_logistic.fit(X_train, y_train)
    model_svm.fit(X_train, y_train)
    model_nb.fit(X_train, y_train)

    return model_logistic, model_svm, model_nb


# ============================================================
# TRAIN MODELS
# ============================================================

with st.spinner("Training machine-learning models..."):

    (
        model_logistic,
        model_svm,
        model_nb
    ) = train_models(X_train, y_train)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_lr_pred = model_logistic.predict(X_test)
y_svm_pred = model_svm.predict(X_test)
y_nb_pred = model_nb.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_lr_pred)
svm_accuracy = accuracy_score(y_test, y_svm_pred)
nb_accuracy = accuracy_score(y_test, y_nb_pred)

sentiments_pred_value = (
    model_logistic.predict_proba(X_test).max(axis=1)
)



# ============================================================
# TEST DATASET
# ============================================================

TEST_reviews_dataset = (
    reviews_dataset.loc[X_test.index].copy()
)

TEST_reviews_dataset["sentiments_pred"] = y_lr_pred

TEST_reviews_dataset["sentiments_pred_value"] = (
    sentiments_pred_value
)

TEST_reviews_dataset.index.name = "index"


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("Sentiment Analysis Dashboard")

st.markdown(
    """
    This analysis investigates whether customer review text can be
    used to automatically classify sentiment and identifies the
    characteristics and topics associated with positive and negative
    customer experiences.
    """
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

selected_sentiment = st.sidebar.selectbox(
    "Select Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

available_years = sorted(
    TEST_reviews_dataset["Year"]
    .dropna()
    .unique()
    .tolist()
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All"] + available_years
)

available_items = sorted(
    TEST_reviews_dataset["asin"]
    .dropna()
    .unique()
    .tolist()
)

selected_item = st.sidebar.selectbox(
    "Select Item",
    ["All"] + available_items
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = TEST_reviews_dataset.copy()

if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["sentiment_classification"]
        == selected_sentiment
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

if selected_item != "All":
    filtered_df = filtered_df[
        filtered_df["asin"] == selected_item
    ]


# ============================================================
# REVIEW OVERVIEW
# ============================================================

st.header("Review Overview")

total_reviews = len(filtered_df)

total_positive = filtered_df[
    filtered_df["sentiment_classification"] == "Positive"
]

total_negative = filtered_df[
    filtered_df["sentiment_classification"] == "Negative"
]

total_neutral = filtered_df[
    filtered_df["sentiment_classification"] == "Neutral"
]

if total_reviews > 0:
    nPositive = round(len(total_positive) * 100 / total_reviews, 2)
    nNegative = round(len(total_negative) * 100 / total_reviews, 2)
    nNeutral = round(len(total_neutral) * 100 / total_reviews, 2)
else:
    nPositive = nNegative = nNeutral = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Positive Reviews", f"{nPositive}%")
col3.metric("Negative Reviews", f"{nNegative}%")
col4.metric("Neutral Reviews", f"{nNeutral}%")


# ============================================================
# RATING OVERVIEW
# ============================================================

st.subheader("Rating Overview")

if filtered_df.empty:

    st.info("No reviews match the selected filters.")

else:

    rating_overview = (
        filtered_df.groupby("overall")
        .size()
        .reset_index(name="Review_Count")
    )

    total_rating_reviews = rating_overview["Review_Count"].sum()

    rating_overview["Rating_percentage"] = (
        100 * rating_overview["Review_Count"]
        / max(total_rating_reviews, 1)
    ).round(1)

    chart_rating_overview = (
        alt.Chart(rating_overview)
        .mark_bar()
        .encode(
            x=alt.X(
                "overall:O",
                title="Rating",
                scale=alt.Scale(domain=[1, 2, 3, 4, 5])
            ),
            y=alt.Y(
                "Rating_percentage:Q",
                title="Reviews (%)"
            ),
            color=alt.Color(
                "Review_Count:Q",
                title="Review Count"
            ),
            tooltip=[
                alt.Tooltip("overall:O", title="Rating"),
                alt.Tooltip("Review_Count:Q", title="Reviews"),
                alt.Tooltip(
                    "Rating_percentage:Q",
                    title="Percentage"
                )
            ]
        )
    )

    st.altair_chart(
        chart_rating_overview,
        use_container_width=True
    )


# ============================================================
# RATING OVER TIME
# ============================================================

st.subheader("Average Rating Over Time")

if filtered_df.empty:

    st.info("No data available for the selected filters.")

elif selected_year == "All":

    rating_per_year = (
        filtered_df.dropna(subset=["Year"])
        .groupby("Year")["overall"]
        .mean()
        .reset_index(name="Rating")
    )

    if rating_per_year.empty:

        st.info("There is no valid date information for the selected data.")

    else:

        line_rating_year = (
            alt.Chart(rating_per_year)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:Q", title="Year"),
                y=alt.Y(
                    "Rating:Q",
                    title="Average Rating",
                    scale=alt.Scale(domain=[1, 5])
                ),
                tooltip=[
                    alt.Tooltip("Year:Q", title="Year"),
                    alt.Tooltip(
                        "Rating:Q",
                        title="Average Rating",
                        format=".2f"
                    )
                ]
            )
        )

        st.altair_chart(
            line_rating_year,
            use_container_width=True
        )

else:

    rating_per_month = (
        filtered_df.dropna(subset=["reviewTime"])
        .assign(
            Month=lambda df: df["reviewTime"].dt.month
        )
        .groupby("Month")["overall"]
        .mean()
        .reindex(range(1, 13))
        .dropna()
        .reset_index(name="Rating")
    )

    if rating_per_month.empty:

        st.info("There is no valid date information for the selected data.")

    else:

        line_rating_month = (
            alt.Chart(rating_per_month)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Month:O",
                    title="Month",
                    sort=list(range(1, 13))
                ),
                y=alt.Y(
                    "Rating:Q",
                    title="Average Rating",
                    scale=alt.Scale(domain=[1, 5])
                ),
                tooltip=[
                    alt.Tooltip("Month:O", title="Month"),
                    alt.Tooltip(
                        "Rating:Q",
                        title="Average Rating",
                        format=".2f"
                    )
                ]
            )
        )

        st.altair_chart(
            line_rating_month,
            use_container_width=True
        )


# ============================================================
# PRODUCT SENTIMENT COMPARISON
# ============================================================

st.subheader("Product Sentiment Comparison")

sentiment_by_asin = calculate_sentiment_percentages(filtered_df)

if sentiment_by_asin.empty:

    st.info("No products match the selected filters.")

elif selected_sentiment == "All":

    col1, col2, col3 = st.columns(3)

    with col1:

        top_positive = (
            sentiment_by_asin.sort_values(
                "positive_%",
                ascending=False
            ).head(10)
        )

        fig_positive = px.bar(
            top_positive,
            x="positive_%",
            y="asin",
            orientation="h",
            title="Most Positive",
            labels={
                "positive_%": "Positive (%)",
                "asin": "ASIN"
            },
            hover_data=["Total", "Positive"]
        )

        fig_positive.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_positive,
            use_container_width=True
        )

    with col2:

        top_neutral = (
            sentiment_by_asin.sort_values(
                "neutral_%",
                ascending=False
            ).head(10)
        )

        fig_neutral = px.bar(
            top_neutral,
            x="neutral_%",
            y="asin",
            orientation="h",
            title="Most Neutral",
            labels={
                "neutral_%": "Neutral (%)",
                "asin": "ASIN"
            },
            hover_data=["Total", "Neutral"]
        )

        fig_neutral.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_neutral,
            use_container_width=True
        )

    with col3:

        top_negative = (
            sentiment_by_asin.sort_values(
                "negative_%",
                ascending=False
            ).head(10)
        )

        fig_negative = px.bar(
            top_negative,
            x="negative_%",
            y="asin",
            orientation="h",
            title="Most Negative",
            labels={
                "negative_%": "Negative (%)",
                "asin": "ASIN"
            },
            hover_data=["Total", "Negative"]
        )

        fig_negative.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_negative,
            use_container_width=True
        )

else:

    selected_columns = {
        "Positive": "positive_%",
        "Negative": "negative_%",
        "Neutral": "neutral_%"
    }

    selected_option = selected_columns[selected_sentiment]

    top_sentiment = (
        sentiment_by_asin.sort_values(
            selected_option,
            ascending=False
        ).head(10)
    )

    fig_sentiment = px.bar(
        top_sentiment,
        x=selected_option,
        y="asin",
        orientation="h",
        title=f"Products with Highest {selected_sentiment} Sentiment",
        labels={
            selected_option: "Sentiment (%)",
            "asin": "ASIN"
        },
        hover_data=["Total", selected_sentiment]
    )

    fig_sentiment.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )


# ============================================================
# WORDS MOST ASSOCIATED WITH EACH SENTIMENT
# ============================================================

st.subheader("Words Most Associated With Each Sentiment")

classifier = model_logistic.named_steps["Classifier"]
tfidf = model_logistic.named_steps["tfidf"]
feature_names = tfidf.get_feature_names_out()

# Logistic Regression coefficients
word_coefficients = pd.DataFrame(
    classifier.coef_,
    columns=feature_names,
    index=classifier.classes_
)




# ============================================================
# FUNCTION TO GET WORD DATA FOR A SINGLE REVIEW
# ============================================================

def get_word_data(review, sentiment, n_words=10):

    # Transform the single review
    review_tfidf = tfidf.transform(
        [review if pd.notna(review) else ""]
    )

    # Get TF-IDF values for this single review
    tfidf_values = review_tfidf.toarray().ravel()

    # Coefficients for the selected sentiment
    coefficients = word_coefficients.loc[sentiment].values

    # Create word DataFrame
    word_data = pd.DataFrame({
        "word": feature_names,
        "tfidf": tfidf_values,
        "coefficient": coefficients
    })

    # Calculate contribution
    word_data["contribution"] = (
        word_data["tfidf"] * word_data["coefficient"]
    )

    # Keep only words present in this review
    word_data = word_data[
        word_data["tfidf"] > 0
    ]

    # Select words with the largest positive contributions
    word_data = (
        word_data.sort_values(
            "contribution",
            ascending=False
        )
        .head(n_words)
        .reset_index(drop=True)
    )

    return word_data


# ============================================================
# FUNCTION TO PLOT WORD DATA USING STREAMLIT
# ============================================================
def plot_word_data(data, sentiment):

    if data.empty:
        st.info(
            f"No word information available for {sentiment}."
        )
        return
    
    
    data["Influence"]=data["contribution"].apply(lambda x:contribution_dictionary(x))
    # Prepare chart data
    chart_data = data.set_index("word")[
        ["tfidf", "coefficient", "contribution"]
    ].copy()

    chart_data.columns = [
        "TF-IDF",
        "Model coefficient",
        "Final contribution"
    ]

    # Streamlit-native line chart
    st.line_chart(
        chart_data,
        x_label="Word",
        y_label="Value"
    )

    # Display word values in an expandable table
    with st.expander(f"View {sentiment} word values"):

        display_data = data.rename(
            columns={
                "word": "Word",
                "tfidf": "TF-IDF",
                "coefficient": "Model coefficient",
                "contribution": "Final contribution",
                "Influence":"Influence"
            }
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# DISPLAY WORD CHARTS
# ============================================================

if filtered_df.empty:

    st.info(
        "No reviews match the selected filters for word analysis."
    )

elif selected_sentiment == "All":

    # --------------------------------------------------------
    # FIND THE PRODUCTS WITH THE HIGHEST SENTIMENT PERCENTAGES
    # --------------------------------------------------------

    top_positive = (
        sentiment_by_asin.sort_values(
            "positive_%",
            ascending=False
        ).head(1)
    )

    top_neutral = (
        sentiment_by_asin.sort_values(
            "neutral_%",
            ascending=False
        ).head(1)
    )

    top_negative = (
        sentiment_by_asin.sort_values(
            "negative_%",
            ascending=False
        ).head(1)
    )

    st.subheader("Three Representative Reviews")

    # --------------------------------------------------------
    # CHECK WHETHER SUFFICIENT PRODUCT DATA EXISTS
    # --------------------------------------------------------

    if (
        top_positive.empty
        or top_neutral.empty
        or top_negative.empty
    ):

        st.info(
            "There is not enough product data for word analysis."
        )

    else:

        # ----------------------------------------------------
        # GET REVIEWS FOR EACH SELECTED PRODUCT
        # ----------------------------------------------------

        review_pos = filtered_df[
            filtered_df["asin"] == top_positive.iloc[0]["asin"]
        ]

        review_neutral = filtered_df[
            filtered_df["asin"] == top_neutral.iloc[0]["asin"]
        ]

        review_neg = filtered_df[
            filtered_df["asin"] == top_negative.iloc[0]["asin"]
        ]

        # ----------------------------------------------------
        # SELECT ONE REPRESENTATIVE REVIEW PER PRODUCT
        # ----------------------------------------------------

        # Lowest sentiment prediction value
        review_pos = review_pos.sort_values(
            "sentiments_pred_value",
            ascending=True
        )

        review_neutral = review_neutral.sort_values(
            "sentiments_pred_value",
            ascending=True
        )

        review_neg = review_neg.sort_values(
            "sentiments_pred_value",
            ascending=True
        )

        # Extract the individual review text
        review_pos_str = (
            review_pos.iloc[0]["reviews_cleaned"]
            if not review_pos.empty
            else ""
        )

        review_neutral_str = (
            review_neutral.iloc[0]["reviews_cleaned"]
            if not review_neutral.empty
            else ""
        )

        review_neg_str = (
            review_neg.iloc[0]["reviews_cleaned"]
            if not review_neg.empty
            else ""
        )

        # ----------------------------------------------------
        # GET WORD DATA FOR EACH INDIVIDUAL REVIEW
        # ----------------------------------------------------

        top_word_positive = (
            get_word_data(review_pos_str, "Positive")
            if not review_pos.empty
            else pd.DataFrame()
        )

        top_word_neutral = (
            get_word_data(review_neutral_str, "Neutral")
            if not review_neutral.empty
            else pd.DataFrame()
        )

        top_word_negative = (
            get_word_data(review_neg_str, "Negative")
            if not review_neg.empty
            else pd.DataFrame()
        )

        # ----------------------------------------------------
        # DISPLAY THE THREE REPRESENTATIVE REVIEWS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        review_groups = [
            (col1, review_pos, review_pos_str, "Positive"),
            (col2, review_neutral, review_neutral_str, "Neutral"),
            (col3, review_neg, review_neg_str, "Negative"),
        ]

        for col, review_df, review_text, sentiment in review_groups:

            with col:

                st.markdown(f"### {sentiment}")

                if review_df.empty:
                    st.info(
                        f"No {sentiment.lower()} review available."
                    )
                    continue

                # Get the selected representative review row
                row = review_df.iloc[0]

                # Display review text and metadata
                st.write(review_text)

                st.caption(
                    f"Rating: {row['overall']}/5"
                )

                st.caption(
                    f"ASIN: {row['asin']}"
                )

        # ----------------------------------------------------
        # DISPLAY POSITIVE WORD CHART
        # ----------------------------------------------------

        with col1:

            st.subheader("Positive")

            st.caption(
                f"Product: {top_positive.iloc[0]['asin']}"
            )

            plot_word_data(
                top_word_positive,
                "Positive"
            )

        # ----------------------------------------------------
        # DISPLAY NEUTRAL WORD CHART
        # ----------------------------------------------------

        with col2:

            st.subheader("Neutral")

            st.caption(
                f"Product: {top_neutral.iloc[0]['asin']}"
            )

            plot_word_data(
                top_word_neutral,
                "Neutral"
            )

        # ----------------------------------------------------
        # DISPLAY NEGATIVE WORD CHART
        # ----------------------------------------------------

        with col3:

            st.subheader("Negative")

            st.caption(
                f"Product: {top_negative.iloc[0]['asin']}"
            )

            plot_word_data(
                top_word_negative,
                "Negative"
            )

else:

    if selected_sentiment not in word_coefficients.index:

        st.info(
            f"No coefficient information is available "
            f"for {selected_sentiment}."
        )

    else:

        selected_review = filtered_df[
            filtered_df["sentiment_classification"]
            == selected_sentiment
        ]

        if selected_review.empty:

            st.info(
                f"No {selected_sentiment.lower()} reviews "
                "match the selected filters."
            )

        else:

            selected_review_text=selected_review.iloc[0]["reviews_cleaned"]
            top_filtered_word = get_word_data(
                selected_review_text,
                selected_sentiment
            )

            st.subheader(selected_sentiment)

            plot_word_data(
                top_filtered_word,
                selected_sentiment
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Linear SVM",
        "Multinomial Naive Bayes"
    ],
    "Accuracy": [
        lr_accuracy,
        svm_accuracy,
        nb_accuracy
    ]
})

performance["Accuracy"] = performance["Accuracy"].round(4)

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL ACCURACY CHART
# ============================================================

fig_model_accuracy = px.bar(
    performance,
    x="Model",
    y="Accuracy",
    title="Model Accuracy",
    text="Accuracy",
    color="Accuracy",
    color_continuous_scale="Blues"
)

fig_model_accuracy.update_traces(
    texttemplate="%{text:.2%}",
    textposition="outside"
)

fig_model_accuracy.update_yaxes(
    range=[
        max(0, performance["Accuracy"].min() - 0.05),
        1
    ]
)

st.plotly_chart(
    fig_model_accuracy,
    use_container_width=True
)


# ============================================================
# DETAILED MODEL EVALUATION
# ============================================================

with st.expander("Detailed Model Evaluation"):

    st.markdown(
        """
        The classification report provides precision, recall,
        F1-score and support for each sentiment class.
        """
    )

    model_choice = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Linear SVM",
            "Multinomial Naive Bayes"
        ],
        key="classification_model"
    )

    if model_choice == "Logistic Regression":
        selected_predictions = y_lr_pred

    elif model_choice == "Linear SVM":
        selected_predictions = y_svm_pred

    else:
        selected_predictions = y_nb_pred

    report = classification_report(
        y_test,
        selected_predictions,
        labels=SENTIMENTS,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose().round(3)

    st.dataframe(
        report_df,
        use_container_width=True
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

with st.expander("Confusion Matrix"):

    model_choice_cm = st.selectbox(
        "Select Model for Confusion Matrix",
        [
            "Logistic Regression",
            "Linear SVM",
            "Multinomial Naive Bayes"
        ],
        key="confusion_model"
    )

    if model_choice_cm == "Logistic Regression":
        predictions_cm = pd.Series(
            y_lr_pred,
            index=y_test.index
        )

    elif model_choice_cm == "Linear SVM":
        predictions_cm = pd.Series(
            y_svm_pred,
            index=y_test.index
        )

    else:
        predictions_cm = pd.Series(
            y_nb_pred,
            index=y_test.index
        )

    selected_predictions_filtered = (
        predictions_cm.reindex(filtered_df.index).dropna()
    )

    selected_y_test = y_test.reindex(
        selected_predictions_filtered.index
    )

    cm = confusion_matrix(
        selected_y_test,
        selected_predictions_filtered,
        labels=SENTIMENTS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    row_totals = cm_df.sum(axis=1).replace(0, 1)

    cm_df_percentage = (
        cm_df.div(row_totals, axis=0) * 100
    ).round(2)

    st.write("Confusion Matrix (%)")

    st.dataframe(
        cm_df_percentage,
        use_container_width=True
    )


# ============================================================
# MISCLASSIFIED REVIEWS
# ============================================================

st.header("Misclassified Reviews")

# Use Logistic Regression predictions by default
default_predictions = pd.Series(
    y_lr_pred,
    index=y_test.index
)

selected_predictions_filtered = (
    default_predictions.reindex(filtered_df.index).dropna()
)

selected_y_test = y_test.reindex(
    selected_predictions_filtered.index
)

wrong_reviews_idx = (
    selected_predictions_filtered[
        selected_predictions_filtered != selected_y_test
    ].index
)

wrong_reviews = filtered_df.loc[
    filtered_df.index.intersection(wrong_reviews_idx)
]




if wrong_reviews.empty:

    st.info(
        "No misclassified reviews were found for the selected filters."
    )

else:
    
    st.write(
    f"Found **{len(wrong_reviews):,}** misclassified reviews."
    )

st.subheader("Examples of Misclassified Reviews")

examples = wrong_reviews.head(3)
columns = st.columns(3)

feature_names = tfidf.get_feature_names_out()

for (idx, row), col in zip(examples.iterrows(), columns):

    actual = selected_y_test.loc[idx]
    predicted = selected_predictions_filtered.loc[idx]

    with col:
        st.markdown(
            f"**Actual:** {actual}\n\n"
            f"**Predicted:** {predicted}"
        )

        review = row["reviews_cleaned"]
        st.write(review)

        # Calculate TF-IDF values for this review
        X_tfidf_array = tfidf.transform([review]).toarray()[0]

        # Get coefficients for the predicted class
        selected_word_coefficients = word_coefficients.loc[predicted]

        # Build word-level contribution data
        tfidf_coeff_contribution = pd.DataFrame({
            "word": feature_names,
            "tfidf": X_tfidf_array,
            "coefficient": selected_word_coefficients
                .reindex(feature_names)
                .to_numpy()
        })

        # Calculate each word's contribution
        tfidf_coeff_contribution["contribution"] = (
            tfidf_coeff_contribution["tfidf"]
            * tfidf_coeff_contribution["coefficient"]
        )

        # Keep only words present in the review
        tfidf_coeff_contribution = (
            tfidf_coeff_contribution[
                tfidf_coeff_contribution["tfidf"] > 0
            ].copy()
        )
        
        # Separate words supporting and opposing the predicted class
        supporting_words = tfidf_coeff_contribution[
            tfidf_coeff_contribution["contribution"] > 0
        ]
        
        
        opposing_words = tfidf_coeff_contribution[
            tfidf_coeff_contribution["contribution"] < 0
        ]
        

        st.subheader(f"Predicted: {predicted}")

        st.markdown("**Words supporting the predicted class**")
        plot_word_data(supporting_words, f"{predicted} — Supporting")

        st.markdown("**Words opposing the predicted class**")
        plot_word_data(opposing_words, f"{predicted} — Opposing")
            