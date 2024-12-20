# Sentiment Analysis Dashboard

This project provides a sentiment analysis dashboard built using **Streamlit** and **RoBERTa**, designed to analyze Reddit posts from selected subreddits. It visualizes sentiment trends, distributions, and key insights from posts.

## Features

- **Fetch Reddit Data**: Retrieves posts from specified subreddits using the Reddit API.
- **Sentiment Analysis**: Leverages the `siebert/sentiment-roberta-large-english` model to classify post titles into positive, negative, or neutral sentiments.
- **Interactive Dashboard**:
  - Word cloud visualization for post titles.
  - Sentiment label distribution chart.
  - Rolling sentiment trends with hourly aggregation.
  - Insights into the most positive and negative posts.
- **Customizable Filters**:
  - Select subreddits.
  - Adjust the rolling average window.
  - Specify the number of posts to fetch per subreddit.

## Requirements

- Python 3.8+
- Required libraries:
  - `praw`
  - `transformers`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `plotly`
  - `streamlit`

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/sentiment_analysis.git
   cd sentiment_analysis
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv sentiment_env
   source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Reddit API Credentials**:
   - Create an app in your Reddit account: [Reddit App Settings](https://www.reddit.com/prefs/apps)
   - Update the `client_id`, `client_secret`, and `user_agent` in the script.

4. **Run the Dashboard**:
   ```bash
   streamlit run sentiment_dashboard.py
   ```

## Usage

- **Select Subreddits**: Choose one or more subreddits to analyze.
- **Adjust Parameters**: Modify the rolling average window and the number of posts to fetch.
- **Visualize Results**:
  - View sentiment trends and insights.
  - Explore raw data with direct links to Reddit posts.

## File Structure

- `sentiment_dashboard.py`: Main script for the Streamlit dashboard.
- `requirements.txt`: List of dependencies.
- Other Python scripts and data files (if applicable).

## Visualizations

### Word Cloud
- A visualization of the most frequently used words in post titles.

### Sentiment Label Distribution
- Bar chart showing the count of positive, negative, and neutral posts.

### Rolling Sentiment Trend
- Line chart displaying sentiment trends aggregated hourly.

## Known Issues

- Some posts may be excluded if the title exceeds the RoBERTa model's token limit.
- Errors can occur if Reddit API credentials are invalid.

## Contributions

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

### Contact

If you have any questions or feedback, please reach out through GitHub or email.

