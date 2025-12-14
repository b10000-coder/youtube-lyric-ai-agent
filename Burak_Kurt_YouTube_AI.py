import os
import sys
import json
import time
import random
import hashlib
from dotenv import load_dotenv
from apify_client import ApifyClient
import tiktoken
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

class YouTubeMusicAgent:
    def __init__(self, apify_token, openrouter_api_key):
        """
        Initialize the agent with API credentials.
        
        Args:
            apify_token: Apify API token
            openrouter_api_key: OpenRouter API key
        """
        self.apify_client = ApifyClient(apify_token)
        self.openrouter_api_key = openrouter_api_key
        
        # Initialize tiktoken encoder
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        
    def human_delay(self, min_seconds=2, max_seconds=5):
        """
        Add a random delay to mimic human behavior.
        
        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds
        """
        delay = random.uniform(min_seconds, max_seconds)
        print(f"Waiting {delay:.1f} seconds...")
        time.sleep(delay)
        
    def get_artist_from_youtube(self, youtube_url):
        """
        Extract artist name from a YouTube video URL using Apify.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            str: Artist name
        """
        print(f"Scraping YouTube URL: {youtube_url}")
        
        try:
            # Run the Apify YouTube Scraper
            run_input = {
                "startUrls": [{"url": youtube_url}],
                "maxResults": 1,
            }
            
            run = self.apify_client.actor("streamers/youtube-scraper").call(run_input=run_input)
            
            # Fetch results
            results = []
            for item in self.apify_client.dataset(run["defaultDatasetId"]).iterate_items():
                results.append(item)
            
            if not results:
                raise Exception("No results from YouTube scraper")
            
            # Extract artist/channel name
            artist_name = results[0].get("channelName") or results[0].get("author")
            print(f"Found artist: {artist_name}")
            
            return artist_name
        except Exception as e:
            print(f"Error getting artist from YouTube: {e}")
            raise
    
    def get_first_album_songs(self, artist_name):
        """
        Use OpenRouter LLM to find all songs from the artist's first album.
        
        Args:
            artist_name: Name of the artist
            
        Returns:
            tuple: (album_name, list of song titles)
        """
        print(f"Asking OpenRouter LLM about {artist_name}'s first album...")
        
        prompt = f"""Please provide a list of all songs from {artist_name}'s first studio album in the exact order they appear on the album (track listing order).

Return the response in the following JSON format only, with no additional text:
{{
    "album_name": "Album Name",
    "songs": ["Song 1", "Song 2", "Song 3"]
}}

Important: List the songs in their original album order (track 1, track 2, etc.)."""
        
        try:
            import requests
            
            # Use OpenRouter API (OpenAI-compatible format)
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional but recommended
                    "X-Title": "YouTube AI Scraper",  # Optional but recommended
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tngtech/tng-r1t-chimera:free",  # Free model
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            # Check for errors
            if response.status_code != 200:
                print(f"OpenRouter API error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
            
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content']
            
            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            album_data = json.loads(response_text)
            album_name = album_data["album_name"]
            songs = album_data["songs"]
            
            print(f"First album: {album_name}")
            print(f"Found {len(songs)} songs")
            
            return album_name, songs
        except Exception as e:
            print(f"Error getting album info: {e}")
            raise
    
    def scrape_genius_lyrics(self, song_title, artist_name):
        """
        Scrape lyrics from Genius.com using Apify with human-like behavior.
        
        Args:
            song_title: Title of the song
            artist_name: Name of the artist
            
        Returns:
            str: Raw lyrics text or None if failed
        """
        # Construct Genius URL
        song_slug = f"{artist_name}-{song_title}".lower()
        song_slug = song_slug.replace(" ", "-").replace("'", "").replace('"', '')
        song_slug = ''.join(c if c.isalnum() or c == '-' else '' for c in song_slug)
        while '--' in song_slug:
            song_slug = song_slug.replace('--', '-')
        song_slug = song_slug.strip('-')
        
        genius_url = f"https://genius.com/{song_slug}-lyrics"
        
        print(f"Scraping lyrics from: {genius_url}")
        
        try:
            run_input = {
                "startUrls": [{"url": genius_url}],
                "maxRequestsPerCrawl": 1,
                "maxConcurrency": 1,
                "pageFunction": """
                    async function pageFunction(context) {
                        const $ = context.jQuery;
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        
                        let lyrics = '';
                        const lyricsContainers = $('[data-lyrics-container="true"]');
                        
                        if (lyricsContainers.length > 0) {
                            lyricsContainers.each(function() {
                                lyrics += $(this).text() + '\\n\\n';
                            });
                        }
                        
                        return {
                            url: context.request.url,
                            lyrics: lyrics.trim()
                        };
                    }
                """,
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                },
                "maxRequestRetries": 2,
                "navigationTimeoutSecs": 60,
                "pageLoadTimeoutSecs": 60
            }
            
            run = self.apify_client.actor("apify/web-scraper").call(run_input=run_input)
            
            results = []
            for item in self.apify_client.dataset(run["defaultDatasetId"]).iterate_items():
                results.append(item)
            
            if results and results[0].get("lyrics"):
                lyrics = results[0]["lyrics"]
                if len(lyrics) > 50:
                    print(f"✓ Successfully scraped lyrics ({len(lyrics)} characters)")
                    return lyrics
                else:
                    print(f"✗ Lyrics too short, might be blocked")
                    return None
            else:
                print(f"✗ No lyrics found")
                return None
                
        except Exception as e:
            print(f"✗ Error scraping: {str(e)}")
            return None
    
    def calculate_token_metrics(self, lyrics):
        """
        Calculate token metrics for lyrics.
        
        Args:
            lyrics: Raw lyrics text
            
        Returns:
            dict: Metrics including token count, char count, word count, tokens-per-word
        """
        if not lyrics:
            return {
                "chars": 0,
                "words": 0,
                "tokens": 0,
                "tokens_per_word": 0.0,
                "hash": hashlib.md5("".encode()).hexdigest()
            }
        
        # Token count using tiktoken
        tokens = self.tokenizer.encode(lyrics)
        token_count = len(tokens)
        
        # Character count
        char_count = len(lyrics)
        
        # Word count
        word_count = len(lyrics.split())
        
        # Tokens per word ratio
        tokens_per_word = round(token_count / word_count, 2) if word_count > 0 else 0.0
        
        # MD5 hash of raw lyrics
        lyrics_hash = hashlib.md5(lyrics.encode('utf-8')).hexdigest()
        
        return {
            "chars": char_count,
            "words": word_count,
            "tokens": token_count,
            "tokens_per_word": tokens_per_word,
            "hash": lyrics_hash
        }
    
    def generate_embedding_hash(self, token_counts):
        """
        Generate embedding from token counts and return MD5 hash.
        
        Args:
            token_counts: List of token counts for each song
            
        Returns:
            str: MD5 hash of the formatted embedding
        """
        # Create concatenated token counts string
        token_string = ",".join(map(str, token_counts))
        print(f"Token counts string: {token_string}")
        
        # Generate embedding
        embedding = self.embedding_model.encode(token_string, convert_to_numpy=True)
        
        # Format embedding: each float with exactly 10 decimal places, comma-separated
        formatted_embedding = ",".join([f"{float(val):.10f}" for val in embedding])
        
        # Calculate MD5 hash
        embedding_hash = hashlib.md5(formatted_embedding.encode('utf-8')).hexdigest()
        
        return embedding_hash
    
    def run(self, youtube_url):
        """
        Main method to run the complete agent workflow.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            tuple: (json_output, embedding_hash)
        """
        print("=" * 60)
        print("Starting YouTube AI Scraping Agent")
        print("=" * 60)
        
        try:
            # Step 1: Get artist name from YouTube
            artist_name = self.get_artist_from_youtube(youtube_url)
            self.human_delay(2, 4)
            
            # Step 2: Get first album songs using LLM
            album_name, songs = self.get_first_album_songs(artist_name)
            
            # Step 3-5: Scrape lyrics and calculate metrics
            songs_data = []
            token_counts = []
            
            for i, song_title in enumerate(songs, 1):
                print(f"\n[{i}/{len(songs)}] Processing: {song_title}")
                
                if i > 1:
                    self.human_delay(3, 6)
                
                lyrics = self.scrape_genius_lyrics(song_title, artist_name)
                metrics = self.calculate_token_metrics(lyrics)
                
                songs_data.append({
                    "name": song_title,
                    "lyrics_length_chars": metrics["chars"],
                    "lyrics_length_words": metrics["words"],
                    "lyrics_length_tokens": metrics["tokens"],
                    "tokens_per_word": metrics["tokens_per_word"],
                    "lyrics_hash": metrics["hash"]
                })
                
                token_counts.append(metrics["tokens"])
            
            # Calculate totals
            total_tokens = sum(token_counts)
            avg_tokens = round(total_tokens / len(songs), 2) if songs else 0
            
            # Create JSON output
            json_output = {
                "artist": artist_name,
                "album_name": album_name,
                "songs": songs_data,
                "total_tokens_all_songs": total_tokens,
                "avg_tokens_per_song": avg_tokens
            }
            
            # Generate embedding hash
            embedding_hash = self.generate_embedding_hash(token_counts)
            
            print("\n" + "=" * 60)
            print("Agent completed successfully!")
            print("=" * 60)
            
            return json_output, embedding_hash
            
        except Exception as e:
            print(f"\nError in workflow: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <youtube_url> <output_type>")
        print("  output_type: 'json' or 'hash'")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    output_type = sys.argv[2].lower()
    
    if output_type not in ['json', 'hash']:
        print("Error: output_type must be 'json' or 'hash'")
        sys.exit(1)
    
    # Load API keys from environment
    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    if not APIFY_TOKEN or not OPENROUTER_API_KEY:
        print("Error: Missing API keys in .env file")
        print("Required: APIFY_TOKEN and OPENROUTER_API_KEY")
        sys.exit(1)
    
    # Initialize and run agent
    try:
        agent = YouTubeMusicAgent(APIFY_TOKEN, OPENROUTER_API_KEY)
        json_output, embedding_hash = agent.run(youtube_url)
        
        # Output based on requested type
        if output_type == 'json':
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        else:  # hash
            print(embedding_hash)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()