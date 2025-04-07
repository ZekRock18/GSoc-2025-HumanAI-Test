import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import os
import torch
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MentalHealthSentimentRiskAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with the path to the filtered mental health posts CSV file.
        
        Args:
            csv_file_path (str): Path to the filtered mental health posts CSV file
        """
        print("Initializing Mental Health Sentiment and Risk Analyzer...")
        self.csv_file_path = csv_file_path
        self.data = None
        
        # Load BERT model for crisis term detection
        print("Loading BERT all-MiniLM-L6-v2 model...")
        self.model = self.load_bert_model('all-MiniLM-L6-v2')
        
        # Define crisis language patterns for risk level detection
        self.high_risk_phrases = [
            "kill myself", "end my life", "suicide", "don't want to be here", 
            "don't want to live", "want to die", "better off dead", "no reason to live",
            "can't take it anymore", "giving up", "ending it all", "final goodbye",
            "hurt myself", "self harm", "cut myself", "overdose", "no hope",
            "plan to end", "how to kill", "painless way", "last day"
        ]
        
        self.moderate_concern_phrases = [
            "need help", "feeling lost", "struggling", "depressed", "anxious",
            "therapy", "medication", "treatment", "counseling", "psychiatrist",
            "psychologist", "mental health", "overwhelmed", "stressed", "panic",
            "afraid", "worried", "lonely", "isolated", "hopeless", "helpless"
        ]
    
    def load_bert_model(self, model_name='all-MiniLM-L6-v2'):
        """
        Load the BERT model with custom caching and error handling.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to load
            
        Returns:
            SentenceTransformer: The loaded model
        """
        # Set up a local cache directory in the project folder
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Using model cache directory: {cache_dir}")
        
        # Try loading with custom cache directory
        try:
            print(f"Attempting to load {model_name} from cache...")
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
            print("Model loaded successfully from cache.")
            return model
        except Exception as e:
            print(f"Error loading model from cache: {e}")
            
            # Try loading with reduced precision to save space
            try:
                print("Attempting to load model with reduced precision...")
                # Set environment variables to reduce memory usage
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                
                # Try to load with half precision
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    model = SentenceTransformer(model_name, cache_folder=cache_dir)
                print("Model loaded successfully with reduced precision.")
                return model
            except Exception as e2:
                print(f"Error loading model with reduced precision: {e2}")
                
                # Final fallback - try loading without custom cache
                try:
                    print("Attempting to load model without custom cache...")
                    model = SentenceTransformer(model_name)
                    print("Model loaded successfully without custom cache.")
                    return model
                except Exception as e3:
                    print(f"All attempts to load model failed: {e3}")
                    raise RuntimeError(f"Failed to load BERT model: {e3}")
    
    def load_data(self):
        """
        Load and preprocess the data from the CSV file.
        """
        print(f"Loading data from {self.csv_file_path}...")
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.data)} posts.")
            
            # Check if summarized_content column exists
            if 'summarized_content' not in self.data.columns:
                print("Error: 'summarized_content' column not found in the CSV file.")
                return False
            
            # Remove rows with empty summarized_content
            self.data = self.data.dropna(subset=['summarized_content'])
            print(f"After removing rows with empty summarized content: {len(self.data)} posts.")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_sentiment(self):
        """
        Analyze sentiment of each post using TextBlob and add sentiment columns to the dataframe.
        """
        print("Analyzing sentiment using TextBlob...")
        
        # Initialize sentiment columns
        self.data['sentiment_polarity'] = np.nan
        self.data['sentiment_subjectivity'] = np.nan
        self.data['sentiment_category'] = ''
        
        for idx, row in self.data.iterrows():
            text = row['summarized_content']
            if isinstance(text, str) and text.strip():
                # Analyze sentiment using TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Assign sentiment values
                self.data.at[idx, 'sentiment_polarity'] = polarity
                self.data.at[idx, 'sentiment_subjectivity'] = subjectivity
                
                # Categorize sentiment
                if polarity > 0.1:
                    self.data.at[idx, 'sentiment_category'] = 'Positive'
                elif polarity < -0.1:
                    self.data.at[idx, 'sentiment_category'] = 'Negative'
                else:
                    self.data.at[idx, 'sentiment_category'] = 'Neutral'
        
        print("Sentiment analysis completed.")
        
        # Print sentiment distribution
        sentiment_counts = self.data['sentiment_category'].value_counts()
        print("\nSentiment Distribution:")
        for category, count in sentiment_counts.items():
            print(f"{category}: {count} posts ({count/len(self.data)*100:.1f}%)")
    
    def detect_risk_level(self):
        """
        Detect risk level of each post based on crisis language and add risk level column to the dataframe.
        """
        print("\nDetecting risk levels using BERT model and pattern matching...")
        
        # Initialize risk level column
        self.data['risk_level'] = ''
        
        # Encode high risk and moderate concern phrases
        high_risk_embeddings = self.model.encode(self.high_risk_phrases)
        moderate_concern_embeddings = self.model.encode(self.moderate_concern_phrases)
        
        for idx, row in self.data.iterrows():
            text = row['summarized_content']
            if isinstance(text, str) and text.strip():
                # Convert text to lowercase for pattern matching
                text_lower = text.lower()
                
                # Direct pattern matching for high risk phrases
                if any(phrase in text_lower for phrase in self.high_risk_phrases):
                    self.data.at[idx, 'risk_level'] = 'High-Risk'
                    continue
                
                # Direct pattern matching for moderate concern phrases
                if any(phrase in text_lower for phrase in self.moderate_concern_phrases):
                    self.data.at[idx, 'risk_level'] = 'Moderate Concern'
                    continue
                
                # Use BERT embeddings for semantic similarity
                text_embedding = self.model.encode([text_lower])
                
                # Calculate similarity with high risk phrases
                high_risk_similarities = cosine_similarity(text_embedding, high_risk_embeddings)[0]
                max_high_risk_similarity = max(high_risk_similarities)
                
                # Calculate similarity with moderate concern phrases
                moderate_concern_similarities = cosine_similarity(text_embedding, moderate_concern_embeddings)[0]
                max_moderate_concern_similarity = max(moderate_concern_similarities)
                
                # Assign risk level based on similarity scores
                if max_high_risk_similarity > 0.7:
                    self.data.at[idx, 'risk_level'] = 'High-Risk'
                elif max_moderate_concern_similarity > 0.6:
                    self.data.at[idx, 'risk_level'] = 'Moderate Concern'
                else:
                    self.data.at[idx, 'risk_level'] = 'Low Concern'
        
        print("Risk level detection completed.")
        
        # Print risk level distribution
        risk_counts = self.data['risk_level'].value_counts()
        print("\nRisk Level Distribution:")
        for level, count in risk_counts.items():
            print(f"{level}: {count} posts ({count/len(self.data)*100:.1f}%)")
    
    def visualize_results(self):
        """
        Create visualizations of sentiment and risk level distributions.
        """
        print("\nCreating visualizations...")
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Sentiment Distribution
        sentiment_counts = self.data['sentiment_category'].value_counts()
        ax1.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
        ax1.set_title('Sentiment Distribution')
        ax1.set_ylabel('Number of Posts')
        
        # Plot 2: Risk Level Distribution
        risk_counts = self.data['risk_level'].value_counts()
        ax2.bar(risk_counts.index, risk_counts.values, color=['red', 'orange', 'blue'])
        ax2.set_title('Risk Level Distribution')
        ax2.set_ylabel('Number of Posts')
        plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
        
        # Plot 3: Heatmap of Sentiment vs Risk Level
        cross_tab = pd.crosstab(self.data['sentiment_category'], self.data['risk_level'])
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)
        ax3.set_title('Sentiment vs Risk Level')
        
        plt.tight_layout()
        plt.savefig('sentiment_risk_analysis.png')
        print("Visualization saved as 'sentiment_risk_analysis.png'")
        plt.show()
        
        # Create a table of the results
        print("\nSentiment vs Risk Level Distribution:")
        print(cross_tab)
        
        # Save the analyzed data to CSV
        output_file = 'analyzed_mental_health_posts.csv'
        self.data.to_csv(output_file, index=False)
        print(f"\nAnalyzed data saved to {output_file}")
    
    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("Starting Mental Health Sentiment and Risk Analysis...")
        
        # Load data
        if not self.load_data():
            print("Analysis aborted due to data loading error.")
            return
        
        # Analyze sentiment
        self.analyze_sentiment()
        
        # Detect risk level
        self.detect_risk_level()
        
        # Visualize results
        self.visualize_results()
        
        print("\nAnalysis completed successfully!")

# Run the analyzer if script is executed directly
if __name__ == "__main__":
    # Use the latest filtered mental health posts CSV file
    import glob
    csv_files = glob.glob("filtered_mental_health_posts_*.csv")
    if csv_files:
        latest_csv = max(csv_files)
        analyzer = MentalHealthSentimentRiskAnalyzer(latest_csv)
        analyzer.run_analysis()
    else:
        print("Error: No filtered mental health posts CSV file found.")