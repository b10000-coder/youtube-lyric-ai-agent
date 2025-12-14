# YouTube AI Scraping Agent

A Python agent that discovers songs from an artist's first album using YouTube, LLMs, and web scraping, then analyzes their lyrics using token metrics and embeddings.

## Features

- Extracts artist name from YouTube video URLs
- Uses LLM (via OpenRouter LLMs) to find first album songs
- Scrapes lyrics from Genius.com with anti-bot detection measures
- Calculates token metrics using tiktoken (cl100k_base encoding)
- Generates embeddings using nomic-ai/nomic-embed-text-v1.5
- Produces deterministic MD5 hashes of lyrics and embeddings

## Prerequisites

- Python 3.8 or higher
- Apify account (free trial available at https://apify.com)
- LLM API key (via OpenRouter: https://openrouter.ai)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bk-kurt/aimultiple-youtube-agent.git
cd aimultiple-youtube-agent
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```bash
# Required
APIFY_TOKEN='your_apify_token'
OPENROUTER_API_KEY='your_openrouter_api_key'
```

## Usage

The script accepts two command-line arguments: YouTube URL and output type.

### Get JSON Analysis:
```bash
python3 Name_Surname_YouTube_AI.py "https://www.youtube.com/watch?v=rSaC-YbSDpo" json
```

Output format:
```json
{
  "artist": "Madonna",
  "album_name": "Madonna",
  "songs": [
    {
      "name": "Lucky Star",
      "lyrics_length_chars": 1482,
      "lyrics_length_words": 264,
      "lyrics_length_tokens": 395,
      "tokens_per_word": 1.5,
      "lyrics_hash": "1933a103bbc2c282de279e859608e189"
    },
    ...
  ],
  "total_tokens_all_songs": 5047,
  "avg_tokens_per_song": 630.88
}
```

### Get Embedding Hash:
```bash
python3 Name_Surname_YouTube_AI.py "https://www.youtube.com/watch?v=rSaC-YbSDpo" hash
```

Output:
```
4ae6acfbd3ccc56d4a62b3f3a4f3eb1f
```

## How It Works

1. **YouTube Scraping**: Uses Apify's YouTube scraper to extract the artist/channel name from the video
2. **Album Discovery**: Queries an LLM to identify all songs from the artist's first studio album in track order
3. **Lyrics Scraping**: For each song, scrapes lyrics from Genius.com using Apify with:
   - Residential proxies to avoid bot detection
   - Human-like delays between requests (3-6 seconds)
   - Browser-like headers
4. **Token Analysis**: Calculates metrics using tiktoken's cl100k_base encoding:
   - Total token count
   - Character count
   - Word count
   - Tokens-per-word ratio
   - MD5 hash of raw lyrics
5. **Embedding Generation**: 
   - Creates a comma-separated string of token counts
   - Generates 768-dimensional embedding using nomic-ai/nomic-embed-text-v1.5
   - Formats each float to exactly 10 decimal places
   - Computes MD5 hash of the formatted embedding string

## Error Handling

The script includes comprehensive error handling:
- Missing API keys
- Failed YouTube scraping
- LLM API errors
- Lyrics scraping failures (continues with remaining songs)
- Invalid command-line arguments

All errors are logged to stderr with descriptive messages.

## Cost Considerations

**Apify Credits**: 
- Residential proxies cost more than standard proxies
- Each lyrics scrape uses ~0.1-0.2 credits
- Free trial includes $5 of credits

**LLM API Costs**:
- OpenAI GPT-4: ~$0.01-0.03 per album
- Anthropic Claude: ~$0.01-0.03 per album
- Free alternatives available through OpenRouter

## Determinism

The system is designed to be deterministic:
- Same YouTube URL produces same artist name
- LLM responses for album info are consistent (though may vary slightly)
- Lyrics scraping from Genius.com is deterministic
- Token counting is deterministic (same lyrics → same tokens)
- Embedding generation uses fixed random seed for reproducibility
- MD5 hashing is deterministic by nature

## Limitations

- Genius.com may occasionally block requests despite anti-bot measures
- LLM responses may vary slightly between runs
- Some songs may not be available on Genius.com
- Residential proxy usage increases Apify costs

## Troubleshooting

**"403 Forbidden" errors from Genius.com**:
- The script uses residential proxies and delays, but heavy blocking may still occur
- Try increasing delays in the code or running at different times

**Missing lyrics**:
- Some songs may not be available on Genius.com
- Song title formatting may not match Genius URL structure
- Check the console output for specific error messages

**LLM errors**:
- Verify API key is correct and has available credits
- Check if the selected model is available in your region
- Try switching to a different LLM provider

## Dependencies

- `apify-client`: Apify API interaction
- `python-dotenv`: Environment variable management
- `tiktoken`: OpenAI's token counting library
- `sentence-transformers`: Embedding generation
