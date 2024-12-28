import os
import praw

# Initialize Reddit API connection
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Test if the connection works
print("Reddit Read-Only Mode:", reddit.read_only)
