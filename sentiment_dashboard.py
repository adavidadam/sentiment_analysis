import praw
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
from datetime import datetime
import sqlite3

# Inject CSS for smaller row height
st.markdown(
    """
    <style>
    table.dataframe tbody tr {
        height: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load pre-trained RoBERTa sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

@st.cache_data
def analyze_sentiment(text):
    """Analyze sentiment using RoBERTa."""
    result = sentiment_pipeline(text, truncation=True, max_length=512)
    return result[0]  # Extract label and score

@st.cache_data
def fetch_reddit_data(subreddits, limit=100):
    """Fetch Reddit posts and analyze sentiment."""
    reddit = praw.Reddit(
        client_id="ddbBQCCdIOytuCRI45Ejiw",
        client_secret="eQX87HpmlpUZ-vu0ZOztK8p9wN1wTw",
        user_agent="sentiment"
    )
    data = []

    for subreddit in subreddits:
        st.write(f"Fetching posts from r/{subreddit}...")
        posts = reddit.subreddit(subreddit).hot(limit=limit)
        for post in posts:
            sentiment = analyze_sentiment(post.title)
            data.append({
                'subreddit': subreddit,
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'created': datetime.utcfromtimestamp(post.created_utc),
                'sentiment_label': sentiment['label'],  # Positive/Negative/Neutral
                'sentiment_score': sentiment['score'],  # Confidence score
                'permalink': f"https://www.reddit.com{post.permalink}"
            })

    return pd.DataFrame(data)

# Streamlit Sidebar Filters
st.sidebar.header("Filters")
subreddits = st.sidebar.multiselect("Select Subreddits", ["wallstreetbets", "stocks", "cryptocurrency"], default=["wallstreetbets"])
rolling_window = st.sidebar.slider("Rolling Average Window", min_value=1, max_value=50, value=10)
limit = st.sidebar.slider("Number of Posts per Subreddit", min_value=10, max_value=500, value=100, step=10)

# Refresh Button
if st.button("Refresh Data"):
    data = fetch_reddit_data(subreddits, limit)
else:
    data = st.session_state.get("data", pd.DataFrame())

if not data.empty:
    # Save data to session state for later use
    st.session_state["data"] = data

    # Add Rolling Average
    data['rolling_sentiment'] = data.groupby('subreddit')['sentiment_score'].transform(lambda x: x.rolling(rolling_window).mean())

    # Word Cloud for Titles
    st.header("Word Cloud for Post Titles")
    all_titles = " ".join(data['title'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_titles)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Sentiment Label Distribution
    st.header("Sentiment Label Distribution")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x='sentiment_label', hue='subreddit')
    plt.title("Distribution of Sentiment Labels")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Rolling Sentiment Trend with Plotly
    st.header("Rolling Sentiment Trend")

    # Ensure datetime conversion
    data['created'] = pd.to_datetime(data['created'])

    # Select only numeric columns for aggregation
    numeric_columns = data.select_dtypes(include=['number']).columns
    aggregated_data = data.groupby(
        [pd.Grouper(key='created', freq='1H'), 'subreddit']
    )[numeric_columns].mean().reset_index()

    # Recalculate rolling sentiment after aggregation
    aggregated_data['rolling_sentiment'] = aggregated_data.groupby('subreddit')['sentiment_score'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean()
    )

    # Plot with Plotly
    fig = px.line(
        aggregated_data,
        x='created',
        y='rolling_sentiment',
        color='subreddit',
        title="Rolling Sentiment Trend (Hourly Aggregated)",
        labels={"created": "Date", "rolling_sentiment": "Sentiment Score"}
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score (Rolling Avg)",
        legend_title="Subreddit",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

    # Insights
    st.header("Insights")
    most_positive = data.loc[data['sentiment_score'].idxmax()]
    most_negative = data.loc[data['sentiment_score'].idxmin()]
    st.subheader("Most Positive Post")
    st.write(f"Subreddit: {most_positive['subreddit']}")
    st.write(f"Title: {most_positive['title']}")
    st.write(f"Score: {most_positive['sentiment_score']}")
    st.write(f"[Go to post]({most_positive['permalink']})")

    st.subheader("Most Negative Post")
    st.write(f"Subreddit: {most_negative['subreddit']}")
    st.write(f"Title: {most_negative['title']}")
    st.write(f"Score: {most_negative['sentiment_score']}")
    st.write(f"[Go to post]({most_negative['permalink']})")

    # Raw Data
    st.header("Raw Data")
    data['permalink'] = data['permalink'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
    st.markdown(
        data.to_html(escape=False, index=False), 
        unsafe_allow_html=True
    )
else:
    st.warning("No data fetched. Check your subreddit selection or click 'Refresh Data'.")
