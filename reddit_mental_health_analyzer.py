import praw
import pandas as pd
import re
import csv
import nltk
import datetime
import os
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define mental health related keywords for filtering
MENTAL_HEALTH_KEYWORDS = [
    "depressed", "depression", "anxiety", "anxious", "suicide", "suicidal", 
    "addiction", "addicted", "substance abuse", "alcoholism", "drug abuse", 
    "overwhelmed", "hopeless", "self-harm", "therapy", "mental health"
]

# Define relevant subreddits
RELEVANT_SUBREDDITS = [
    "depression", "anxiety", "mentalhealth", "SuicideWatch", "addiction", 
    "alcoholism", "leaves", "StopDrinking", "ADHD", "bipolar", "BPD", 
    "ptsd", "OCD", "schizophrenia", "selfharm"
]

class RedditMentalHealthAnalyzer:
    def __init__(self):
        # Load Reddit API credentials from environment variables
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT")
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        # Initialize NLTK stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize model and tokenizer using download_model module
        from download_model import get_model_and_tokenizer, generate_text
        print("Initializing Gemma 3 model...")
        self.tokenizer, self.model = get_model_and_tokenizer()
        self.generate_text = generate_text
        
        # Set up file paths
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create v31_data directory if it doesn't exist
        self.output_dir = "v31_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.all_posts_file = os.path.join(self.output_dir, f"reddit_mental_health_posts_{self.timestamp}.csv")
        self.filtered_posts_file = os.path.join(self.output_dir, f"filtered_mental_health_posts_{self.timestamp}.csv")
    
    def extract_posts(self, limit=5):  # Changed default from 100 to 5 posts per subreddit
        """Extract posts from relevant subreddits using relevance filter"""
        all_posts = []
        
        for subreddit_name in RELEVANT_SUBREDDITS:
            try:
                print(f"Extracting posts from r/{subreddit_name} by relevance...")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get posts sorted by relevance instead of hot
                # Using search with mental health keywords for better relevance
                search_query = "OR ".join(MENTAL_HEALTH_KEYWORDS)
                for post in subreddit.search(search_query, sort="relevance", limit=limit):
                    # Extract post data
                    post_data = {
                        'post_id': post.id,
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'content': post.selftext,
                        'author': str(post.author),
                        'created_utc': datetime.datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'upvote_ratio': post.upvote_ratio,
                        'location': self.extract_location(post.title + " " + post.selftext)
                    }
                    all_posts.append(post_data)
            except Exception as e:
                print(f"Error extracting posts from r/{subreddit_name}: {e}")
        
        # Save all posts to CSV
        self.save_to_csv(all_posts, self.all_posts_file)
        print(f"Extracted {len(all_posts)} posts and saved to {self.all_posts_file}")
        
        return all_posts
    
    def extract_location(self, text):
        """Extract location information from text if available"""
        # Simple regex patterns to find common location formats
        location_patterns = [
            r'\bin ([A-Z][a-z]+(?:, [A-Z]{2})?)\b',  # in City or in City, ST
            r'\bfrom ([A-Z][a-z]+(?:, [A-Z]{2})?)\b',  # from City or from City, ST
            r'\b([A-Z]{2}, USA)\b',  # ST, USA
            r'\b([A-Z][a-z]+ County)\b',  # County name
            r'\b(North|South|East|West|New) ([A-Z][a-z]+)\b'  # Directional city names
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
        
        return "Unknown"
    
    def preprocess_text(self, text):
        """Remove stopwords, emojis, and special characters"""
        if not text or pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove emojis (basic approach)
        text = re.sub(r'[^\w\s,.!?]', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(filtered_words)
    
    def filter_relevant_posts(self, posts):
        """Filter posts based on mental health keywords"""
        filtered_posts = []
        
        for post in posts:
            combined_text = (post['title'] + " " + post['content']).lower()
            
            # Check if any keyword is in the post
            if any(keyword.lower() in combined_text for keyword in MENTAL_HEALTH_KEYWORDS):
                # Create a copy of the post
                filtered_post = post.copy()
                
                # Add which keywords were found
                found_keywords = [keyword for keyword in MENTAL_HEALTH_KEYWORDS 
                                if keyword.lower() in combined_text]
                filtered_post['keywords_found'] = ', '.join(found_keywords)
                
                # Add summarized content
                filtered_post['summarized_content'] = self.summarize_post(post, found_keywords)
                
                # Add preprocessed content
                filtered_post['preprocessed_content'] = self.preprocess_text(post['content'])
                
                filtered_posts.append(filtered_post)
        
        # Save filtered posts to CSV
        self.save_to_csv(filtered_posts, self.filtered_posts_file)
        print(f"Filtered {len(filtered_posts)} relevant posts and saved to {self.filtered_posts_file}")
        
        return filtered_posts
    
    def summarize_post(self, post, keywords):
        """Summarize post content using Gemma 3 model"""
        try:
            # Combine title and content
            full_text = f"Title: {post['title']}\n\nContent: {post['content']}"
            
            # Create prompt for the model
            prompt = f"""Summarize the following Reddit post in the context of mental health issues, 
            focusing on these keywords: {', '.join(keywords)}. 
            Keep the summary under 450 tokens and preserve the main points and emotional context.
            
            {full_text}
            
            Summary:"""
            
            # Use the generate_text function from download_model module
            summary = self.generate_text(self.tokenizer, self.model, prompt)
            
            # Extract only the summary part (after "Summary:")
            if "Summary:" in summary:
                summary = summary.split("Summary:")[1].strip()
            
            return summary
        
        except Exception as e:
            print(f"Error summarizing post {post['post_id']}: {e}")
            return "Error generating summary"
    
    def save_to_csv(self, posts, filename):
        """Save posts to CSV file"""
        if not posts:
            print(f"No posts to save to {filename}")
            return
        
        # Get all unique keys from all posts
        fieldnames = set()
        for post in posts:
            fieldnames.update(post.keys())
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(posts)
    
    def run(self):
        """Run the full analysis pipeline"""
        print("Starting Reddit Mental Health Content Analysis...")
        
        # Extract posts from Reddit
        posts = self.extract_posts()
        
        # Filter relevant posts
        filtered_posts = self.filter_relevant_posts(posts)
        
        print(f"Analysis complete. Found {len(filtered_posts)} relevant posts out of {len(posts)} total posts.")
        print(f"All posts saved to: {self.all_posts_file}")
        print(f"Filtered posts saved to: {self.filtered_posts_file}")

# Run the analyzer if script is executed directly
if __name__ == "__main__":
    analyzer = RedditMentalHealthAnalyzer()
    analyzer.run()