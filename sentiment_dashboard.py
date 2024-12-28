import os
import openai
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
from datetime import datetime

import os

from dotenv import load_dotenv
load_dotenv()

# Debugging info
print(f"OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY')}")






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

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
print(f"Current API Key: {api_key}")

if not api_key:
    st.error("OpenAI API Key not found in environment variables")
    client = None
else:
    client = openai.OpenAI(
        api_key=api_key,
        organization=os.getenv("OPENAI_ORG_ID", None)  # Optional: Add if you have a specific org ID
    )

def generate_summary(text):
    """Generate a summary using OpenAI GPT."""
    if not client:
        return "OpenAI client not initialized - check API key"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant summarizing Reddit sentiment data."},
                {"role": "user", "content": f"Summarize the following data:\n{text[:3000]}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return "Error generating summary"

from transformers import pipeline

@st.cache_resource  # Streamlit caching for models
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

sentiment_pipeline = load_sentiment_pipeline()

@st.cache_data
def analyze_sentiment(text):
    """Analyze sentiment using RoBERTa."""
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)
        return result[0]
    except Exception as e:
        st.warning(f"Error analyzing sentiment: {e}")
        return {"label": "Neutral", "score": 0.0}

@st.cache_data
def fetch_reddit_data(subreddits, limit=100):
    """Fetch Reddit posts and analyze sentiment."""
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
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
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'permalink': f"https://www.reddit.com{post.permalink}"
                })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Reddit API Error: {e}")
        return pd.DataFrame()

# Add API key status check
st.sidebar.write(f"OpenAI API Key set: {'OPENAI_API_KEY' in os.environ}")

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
    data['rolling_sentiment'] = data.groupby('subreddit')['sentiment_score'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())

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
    data['created'] = pd.to_datetime(data['created'])
    numeric_columns = data.select_dtypes(include=['number']).columns
    aggregated_data = data.groupby(
        [pd.Grouper(key='created', freq='1H'), 'subreddit']
    )[numeric_columns].mean().reset_index()
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

    # AI Summary
    st.header("AI Summary")
    text_to_summarize = " ".join(data['title'].tolist())
    if text_to_summarize:
        try:
            summary = generate_summary(text_to_summarize)
            st.write("**AI-Generated Summary:**", summary)
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")

    # Raw Data
    st.header("Raw Data")
    data['permalink'] = data['permalink'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
    st.markdown(
        data.to_html(escape=False, index=False), 
        unsafe_allow_html=True
    )
else:
    st.warning("No data fetched. Check your subreddit selection or click 'Refresh Data'.")