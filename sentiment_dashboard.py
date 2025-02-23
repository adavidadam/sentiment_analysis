import os
import re
import openai
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------
# ENHANCED CSS INJECTION FOR TEXT FIXES
# ----------------------------------
st.markdown(
    """
    <style>
    /* Reset all text containers */
    .stMarkdown, .stText, .element-container, 
    [data-testid="stMarkdownContainer"],
    .streamlit-expanderContent {
        all: revert;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
        text-align: left !important;
    }

    /* Force normal text rendering for all paragraph elements */
    .stMarkdown p, 
    .stText p,
    [data-testid="stMarkdownContainer"] p {
        white-space: normal !important;
        word-wrap: break-word !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
        margin-bottom: 1em !important;
        font-feature-settings: normal !important;
        font-variant: normal !important;
        font-kerning: auto !important;
    }

    /* Ensure proper text display in special containers */
    .element-container div {
        text-align: left !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    /* Table styles */
    table.dataframe {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-size: 14px !important;
    }

    table.dataframe tbody tr {
        height: 20px;
    }

    /* Headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
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
        organization=os.getenv("OPENAI_ORG_ID", None)  # Optional org ID
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

        TICKER_PATTERN = r"\$[A-Za-z]{1,5}\b"  

        for subreddit in subreddits:
            st.write(f"Fetching posts from r/{subreddit}...")
            posts = reddit.subreddit(subreddit).hot(limit=limit)
            for post in posts:
                sentiment = analyze_sentiment(post.title)
                tickers_found = re.findall(TICKER_PATTERN, post.title.upper())  
                tickers_found = list(set(tickers_found))  # unique tickers

                data.append({
                    'subreddit': subreddit,
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'permalink': f"https://www.reddit.com{post.permalink}",
                    'tickers': tickers_found
                })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Reddit API Error: {e}")
        return pd.DataFrame()

# Add API key status check
st.sidebar.write(f"OpenAI API Key set: {'OPENAI_API_KEY' in os.environ}")

# Streamlit Sidebar Filters
st.sidebar.header("Filters")
subreddits = st.sidebar.multiselect(
    "Select Subreddits", 
    ["wallstreetbets", "stocks", "cryptocurrency"], 
    default=["wallstreetbets"]
)
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
    data['rolling_sentiment'] = data.groupby('subreddit')['sentiment_score'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean()
    )

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
        [pd.Grouper(key='created', freq='1h'), 'subreddit']
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
    st.markdown(f"""
    **Subreddit:** {most_positive['subreddit']}  
    **Title:** {most_positive['title']}  
    **Score:** {most_positive['sentiment_score']}  
    [Go to post]({most_positive['permalink']})
    """)

    st.subheader("Most Negative Post")
    st.markdown(f"""
    **Subreddit:** {most_negative['subreddit']}  
    **Title:** {most_negative['title']}  
    **Score:** {most_negative['sentiment_score']}  
    [Go to post]({most_negative['permalink']})
    """)

    # Top Ticker Mentions (Fixed with Plotly for unnormalized bars)
    st.header("Top Ticker Mentions")
    ticker_series = data['tickers'].explode()  # flatten the list of tickers
    if not ticker_series.dropna().empty:
        top_tickers = ticker_series.value_counts().head(10)
        # Use plotly for a clear, unnormalized bar chart
        fig = px.bar(top_tickers, 
                     x=top_tickers.index, 
                     y=top_tickers.values, 
                     title="Top Ticker Mentions", 
                     labels={"x": "Tickers", "y": "Number of Mentions"})
        fig.update_layout(xaxis_tickangle=45,  # Rotate x-axis labels for readability
                         yaxis_range=[0, top_tickers.max() * 1.1])  # Set y-axis to show actual counts
        st.plotly_chart(fig)
    else:
        st.write("No tickers found in post titles.")

    # Ticker-based GPT Summaries
    st.header("Ticker-Specific Summaries")
    if not ticker_series.dropna().empty:
        top_3_tickers = ticker_series.value_counts().head(3).index.tolist()
        for t in top_3_tickers:
            t_df = data[data['tickers'].apply(lambda x: t in x if x else False)]
            ticker_titles = " ".join(t_df['title'].tolist())
            if ticker_titles:
                st.subheader(f"Summary for {t}")
                try:
                    summary = generate_summary(ticker_titles)
                    st.markdown(f"**Analysis:** {summary}")
                except Exception as e:
                    st.error(f"Error generating summary for {t}: {str(e)}")

    # Sentiment Change Alerts
    st.header("Sentiment Change Alerts")
    if not ticker_series.dropna().empty:
        ticker_threshold = data.explode('tickers').groupby('tickers')['sentiment_score'].count()
        tickers_to_watch = ticker_threshold[ticker_threshold >= 5].index

        alert_messages = []
        for t in tickers_to_watch:
            t_timesorted = data[data['tickers'].apply(lambda x: t in x if x else False)].sort_values('created')
            if len(t_timesorted) > 1:
                first_sentiment = t_timesorted.iloc[0]['sentiment_score']
                last_sentiment = t_timesorted.iloc[-1]['sentiment_score']
                change = last_sentiment - first_sentiment
                if abs(change) > 0.5:
                    alert_messages.append(
                        f"{t} sentiment changed by {change:.2f} from {first_sentiment:.2f} to {last_sentiment:.2f}"
                    )
        if alert_messages:
            for msg in alert_messages:
                st.warning(msg)
        else:
            st.write("No significant sentiment shifts detected for the top-mentioned tickers.")

    # AI Summary (overall)
    st.header("AI Summary")
    text_to_summarize = " ".join(data['title'].tolist())
    if text_to_summarize:
        try:
            summary = generate_summary(text_to_summarize)
            st.markdown(f"**AI-Generated Summary:**\n\n{summary}")
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