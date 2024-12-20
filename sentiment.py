import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
import streamlit as st

# Fetch and Process Reddit Data
@st.cache_data
def fetch_reddit_data(subreddits, limit=100):
    """Fetch Reddit posts and analyze sentiment."""
    reddit = praw.Reddit(
        client_id="ddbBQCCdIOytuCRI45Ejiw",
        client_secret="eQX87HpmlpUZ-vu0ZOztK8p9wN1wTw",
        user_agent="sentiment"
    )
    analyzer = SentimentIntensityAnalyzer()
    data = []

    for subreddit in subreddits:
        st.text(f"Fetching posts from r/{subreddit}...")
        try:
            posts = reddit.subreddit(subreddit).hot(limit=limit)
            for post in posts:
                sentiment = analyzer.polarity_scores(post.title)
                data.append({
                    'subreddit': subreddit,
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created': datetime.utcfromtimestamp(post.created_utc),
                    'neg': sentiment['neg'],
                    'neu': sentiment['neu'],
                    'pos': sentiment['pos'],
                    'compound': sentiment['compound']
                })
        except Exception as e:
            st.error(f"Error fetching data from r/{subreddit}: {e}")
            continue

    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv('latest_sentiment_data.csv', index=False)  # Save data for backup
        st.text("Data saved to latest_sentiment_data.csv.")
    return df
