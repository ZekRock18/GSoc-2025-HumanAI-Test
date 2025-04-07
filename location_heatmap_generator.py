import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import spacy
import re
import matplotlib.pyplot as plt
from collections import Counter
import os
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download spaCy model if not already installed
try:
    import en_core_web_sm
except ImportError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    import en_core_web_sm

class LocationHeatmapGenerator:
    def __init__(self, csv_file_path):
        """
        Initialize the generator with the path to the analyzed mental health posts CSV file.
        
        Args:
            csv_file_path (str): Path to the analyzed mental health posts CSV file
        """
        print("Initializing Location Heatmap Generator...")
        self.csv_file_path = csv_file_path
        self.data = None
        self.nlp = en_core_web_sm.load()
        
        # Dictionary of common US cities and their coordinates
        self.city_coordinates = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863),
            'Austin': (30.2672, -97.7431),
            'Jacksonville': (30.3322, -81.6557),
            'Fort Worth': (32.7555, -97.3308),
            'Columbus': (39.9612, -82.9988),
            'San Francisco': (37.7749, -122.4194),
            'Charlotte': (35.2271, -80.8431),
            'Indianapolis': (39.7684, -86.1581),
            'Seattle': (47.6062, -122.3321),
            'Denver': (39.7392, -104.9903),
            'Washington': (38.9072, -77.0369),
            'Boston': (42.3601, -71.0589),
            'Nashville': (36.1627, -86.7816),
            'Baltimore': (39.2904, -76.6122),
            'Portland': (45.5051, -122.6750),
            'Las Vegas': (36.1699, -115.1398),
            'Milwaukee': (43.0389, -87.9065),
            'Albuquerque': (35.0844, -106.6504),
            'Tucson': (32.2226, -110.9747),
            'Fresno': (36.7378, -119.7871),
            'Sacramento': (38.5816, -121.4944),
            'Atlanta': (33.7490, -84.3880),
            'Miami': (25.7617, -80.1918),
            'Cleveland': (41.4993, -81.6944),
            'Minneapolis': (44.9778, -93.2650),
            'Tampa': (27.9506, -82.4572),
            'Orlando': (28.5383, -81.3792),
            'Detroit': (42.3314, -83.0458),
            'Pittsburgh': (40.4406, -79.9959),
            'Cincinnati': (39.1031, -84.5120),
            'St. Louis': (38.6270, -90.1994),
            'Kansas City': (39.0997, -94.5786),
            'Raleigh': (35.7796, -78.6382),
            'Omaha': (41.2565, -95.9345),
            'Colorado Springs': (38.8339, -104.8214),
            'Oakland': (37.8044, -122.2711),
            'Tulsa': (36.1540, -95.9928),
            'Arlington': (32.7357, -97.1081),
            'New Orleans': (29.9511, -90.0715),
            'Wichita': (37.6872, -97.3301),
            'Bakersfield': (35.3733, -119.0187)
        }
        
        # State abbreviations to full names mapping
        self.state_mapping = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 
            'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
    
    def load_data(self):
        """
        Load the CSV data into a pandas DataFrame
        """
        print(f"Loading data from {self.csv_file_path}...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Loaded {len(self.data)} posts.")
        return self.data
    
    def extract_location_with_nlp(self, text):
        """
        Extract location information from text using spaCy NER and pattern matching
        
        Args:
            text (str): The text to extract location from
            
        Returns:
            tuple: (location_name, latitude, longitude) or (None, None, None) if no location found
        """
        if pd.isna(text) or text == "":
            return None, None, None
        
        # First try the existing regex patterns
        location_patterns = [
            r'\bin ([A-Z][a-z]+(?:, [A-Z]{2})?)\b',  # in City or in City, ST
            r'\bfrom ([A-Z][a-z]+(?:, [A-Z]{2})?)\b',  # from City or from City, ST
            r'\b([A-Z]{2}, USA)\b',  # ST, USA
            r'\b([A-Z][a-z]+ County)\b',  # County name
            r'\b(North|South|East|West|New) ([A-Z][a-z]+)\b',  # Directional city names
            r'\bliving in ([A-Z][a-z]+)\b',  # living in City
            r'\bmoved to ([A-Z][a-z]+)\b',  # moved to City
            r'\bvisiting ([A-Z][a-z]+)\b',  # visiting City
            r'\bin the ([A-Z][a-z]+) area\b',  # in the City area
            r'\bnear ([A-Z][a-z]+)\b',  # near City
            r'\baround ([A-Z][a-z]+)\b',  # around City
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                location = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                # Check if it's a city, state format
                city_state_match = re.match(r'([A-Za-z\s]+), ([A-Z]{2})', location)
                if city_state_match:
                    city = city_state_match.group(1).strip()
                    state = city_state_match.group(2)
                    # Check if city is in our coordinates dictionary
                    if city in self.city_coordinates:
                        return city, self.city_coordinates[city][0], self.city_coordinates[city][1]
                    # If not, return the city name but no coordinates
                    return f"{city}, {state}", None, None
                # Check if it's just a city
                elif location in self.city_coordinates:
                    return location, self.city_coordinates[location][0], self.city_coordinates[location][1]
                # If it's a state abbreviation
                elif location in self.state_mapping:
                    return self.state_mapping[location], None, None
                # Return the location name but no coordinates
                return location, None, None
        
        # If regex didn't find anything, try spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text
                # Check if it's a known city
                if location in self.city_coordinates:
                    return location, self.city_coordinates[location][0], self.city_coordinates[location][1]
                # Check if it's a state abbreviation
                elif location in self.state_mapping:
                    return self.state_mapping[location], None, None
                # Return the location name but no coordinates
                return location, None, None
        
        return None, None, None
    
    def process_locations(self):
        """
        Process all posts to extract and enhance location data
        
        Returns:
            pandas.DataFrame: DataFrame with enhanced location information
        """
        if self.data is None:
            self.load_data()
        
        print("Processing locations from posts...")
        
        # Create new columns for enhanced location data
        self.data['enhanced_location'] = None
        self.data['latitude'] = None
        self.data['longitude'] = None
        
        # Process each post to extract location
        for idx, row in self.data.iterrows():
            # Combine title and content for better location extraction
            text = f"{row['title']} {row['content']}"
            
            # Extract location using NLP
            location, lat, lng = self.extract_location_with_nlp(text)
            
            # Update the DataFrame
            self.data.at[idx, 'enhanced_location'] = location if location else "Unknown"
            self.data.at[idx, 'latitude'] = lat
            self.data.at[idx, 'longitude'] = lng
        
        # Count the number of posts with identified locations
        location_count = len(self.data[self.data['enhanced_location'] != "Unknown"])
        print(f"Identified locations in {location_count} out of {len(self.data)} posts.")
        
        return self.data
    
    def generate_heatmap(self, risk_level=None):
        """
        Generate a heatmap of posts with location data
        
        Args:
            risk_level (str, optional): Filter by risk level (High-Risk, Moderate-Risk, Low-Risk)
            
        Returns:
            folium.Map: Folium map object with heatmap layer
        """
        if self.data is None or 'enhanced_location' not in self.data.columns:
            self.process_locations()
        
        # Filter data by risk level if specified
        if risk_level:
            filtered_data = self.data[self.data['risk_level'] == risk_level]
        else:
            filtered_data = self.data
        
        # Filter out rows without coordinates
        geo_data = filtered_data.dropna(subset=['latitude', 'longitude'])
        
        if len(geo_data) == 0:
            print("No posts with valid coordinates found.")
            return None
        
        print(f"Generating heatmap with {len(geo_data)} posts...")
        
        # Create a base map centered on the US
        map_center = [39.8283, -98.5795]  # Center of the US
        heatmap = folium.Map(location=map_center, zoom_start=4)
        
        # Prepare data for heatmap
        heat_data = [[row['latitude'], row['longitude']] for _, row in geo_data.iterrows()]
        
        # Add the heatmap layer
        HeatMap(heat_data).add_to(heatmap)
        
        # Save the map to an HTML file
        risk_suffix = f"_{risk_level.lower().replace('-', '_')}" if risk_level else ""
        output_file = f"mental_health_heatmap{risk_suffix}.html"
        heatmap.save(output_file)
        
        print(f"Heatmap saved to {output_file}")
        return heatmap
    
    def get_top_locations(self, n=5):
        """
        Get the top N locations with the highest number of crisis-related posts
        
        Args:
            n (int): Number of top locations to return
            
        Returns:
            list: List of tuples (location, count)
        """
        if self.data is None or 'enhanced_location' not in self.data.columns:
            self.process_locations()
        
        # Filter to only include high and moderate risk posts with known locations
        crisis_posts = self.data[
            (self.data['risk_level'].isin(['High-Risk', 'Moderate-Risk'])) & 
            (self.data['enhanced_location'] != "Unknown")
        ]
        
        # Count occurrences of each location
        location_counts = Counter(crisis_posts['enhanced_location'])
        
        # Get the top N locations
        top_locations = location_counts.most_common(n)
        
        return top_locations
    
    def visualize_top_locations(self, n=5):
        """
        Create a bar chart of the top N locations with the highest number of crisis-related posts
        
        Args:
            n (int): Number of top locations to visualize
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the bar chart
        """
        top_locations = self.get_top_locations(n)
        
        if not top_locations:
            print("No locations with crisis-related posts found.")
            return None
        
        # Extract locations and counts
        locations = [loc for loc, _ in top_locations]
        counts = [count for _, count in top_locations]
        
        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(locations, counts, color='#ff7f0e')
        
        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Top Locations with Crisis-Related Mental Health Posts', fontsize=16)
        plt.xlabel('Location', fontsize=14)
        plt.ylabel('Number of Posts', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('top_crisis_locations.png')
        print("Top locations visualization saved to 'top_crisis_locations.png'")
        
        return plt.gcf()

# Main execution
if __name__ == "__main__":
    # Initialize the generator
    generator = LocationHeatmapGenerator("analyzed_mental_health_posts.csv")
    
    # Process locations
    generator.process_locations()
    
    # Generate heatmap for all posts
    generator.generate_heatmap()
    
    # Generate heatmap for high-risk posts
    generator.generate_heatmap(risk_level="High-Risk")
    
    # Visualize top 5 locations with crisis-related posts
    generator.visualize_top_locations(5)
    
    print("\nTop 5 locations with the highest crisis discussions:")
    top_locations = generator.get_top_locations(5)
    for i, (location, count) in enumerate(top_locations, 1):
        print(f"{i}. {location}: {count} posts")