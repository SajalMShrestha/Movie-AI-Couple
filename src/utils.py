"""
Utility functions and constants for the movie recommendation system.
"""

import streamlit as st
import requests
import concurrent.futures
from sentence_transformers import SentenceTransformer
from tmdbv3api import Movie
import torch

# Global configuration constants
RECOMMENDATION_WEIGHTS = {
    "mood_tone": 0.15,
    "genre_similarity": 0.10,
    "cast_crew": 0.10,
    "narrative_style": 0.08,
    "ratings": 0.05,
    "trending_factor": 0.07,
    "release_year": 0.05,
    "discovery_boost": 0.05,
    "age_alignment": 0.0,
    "embedding_similarity": 0.35
}

STREAMING_PLATFORM_PRIORITY = {
    "netflix": 1.0,
    "disney_plus": 0.9,
    "hbo_max": 0.85,
    "hulu": 0.8,
    "prime_video": 0.75,
    "apple_tv": 0.7,
    "peacock": 0.6,
    "paramount_plus": 0.5
}

MOOD_TONE_MAP = {
    "feel_good": {"Comedy", "Romance", "Music", "Adventure"},
    "gritty": {"Crime", "Thriller", "Mystery", "Drama"},
    "cerebral": {"Sci-Fi", "Mystery", "History"},
    "intense": {"Action", "War", "Horror"},
    "melancholic": {"Drama", "History"},
    "classic": {"Western", "Film-Noir"}
}

@st.cache_resource
def get_embedding_model():
    """Get the sentence transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def get_mood_score(genres, preferred_moods):
    """Calculate mood score based on genre overlap with preferred moods."""
    matched_moods = set()
    for g in genres:
        genre_name = g.get('name', '') if isinstance(g, dict) else getattr(g, 'name', '')
        if genre_name:
            for mood, tags in MOOD_TONE_MAP.items():
                if genre_name in tags:
                    matched_moods.add(mood)
    overlap = matched_moods & preferred_moods
    return len(overlap) / max(len(preferred_moods), 1)

def fetch_similar_movie_details(m_id, fetch_cache=None):
    """
    Fetch detailed movie information including credits and generate embeddings.
    
    Args:
        m_id: Movie ID from TMDB
        fetch_cache: Cache dictionary to store results
    
    Returns:
        Tuple of (movie_id, (movie_details, embedding)) or (movie_id, None)
    """
    # ADD DEBUG AT START
    # st.write(f"🔍 Fetching details for movie ID: {m_id}")
    if fetch_cache is None:
        fetch_cache = {}
    
    try:
        # Check cache first
        if m_id in fetch_cache:
            # st.write(f"✅ Found in cache: {m_id}")
            return m_id, fetch_cache[m_id]
        
        movie_api = Movie()
        m_details = movie_api.details(m_id)
        # st.write(f"📄 Got details for: {getattr(m_details, 'title', 'Unknown')}")
        m_credits = movie_api.credits(m_id)

        # Robust genre, cast, director extraction
        genres = []
        genres_list = getattr(m_details, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                genres.append(name)

        cast_list_raw = m_credits.get('cast', []) if isinstance(m_credits, dict) else getattr(m_credits, 'cast', [])
        crew_list = m_credits.get('crew', []) if isinstance(m_credits, dict) else getattr(m_credits, 'crew', [])
        
        if hasattr(cast_list_raw, '__iter__'):
            m_details.cast = list(cast_list_raw)[:3] if cast_list_raw else []
        else:
            m_details.cast = []

        directors = []
        for c in crew_list:
            is_director = False
            name = ''
            if isinstance(c, dict):
                is_director = c.get('job', '') == 'Director'
                name = c.get('name', '')
            else:
                is_director = getattr(c, 'job', '') == 'Director'
                name = getattr(c, 'name', '')
            
            if is_director and name:
                directors.append(name)
        
        m_details.directors = directors
        m_details.plot = getattr(m_details, 'overview', '') or ''

        # Skip if plot is missing or too short
        overview = m_details.plot
        if not overview:
            # st.write(f"❌ No overview for movie ID: {m_id}")
            fetch_cache[m_id] = None
            return m_id, None
        # st.write(f"📝 Overview length: {len(overview)}")
        if len(overview.split()) < 5:
            fetch_cache[m_id] = None
            return m_id, None

        # COMMENTED OUT narrative analysis
        # from narrative_analysis import infer_narrative_style
        # m_details.narrative_style = infer_narrative_style(m_details.plot)
        
        # Add a dummy narrative style instead:
        m_details.narrative_style = {"tone": "neutral", "complexity": "simple"}

        # Generate embedding
        embedding_model = get_embedding_model()
        embedding = embedding_model.encode(m_details.plot, convert_to_tensor=True)
        # st.write(f"🧠 Generated embedding shape: {embedding.shape}")

        fetch_cache[m_id] = (m_details, embedding)
        return m_id, (m_details, embedding)

    except Exception as e:
        # st.write(f"❌ Error fetching movie {m_id}: {e}")
        fetch_cache[m_id] = None
        return m_id, None

def get_trending_popularity(api_key):
    """Fetch and normalize trending popularity scores."""
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if not data:
                return {}
            max_pop = max([m.get("popularity", 1) for m in data])
            return {m["id"]: m.get("popularity", 0) / max_pop for m in data}
    except:
        return {}

def estimate_user_age(years):
    """Estimate user age based on release years of favorite movies."""
    if not years:
        return 30
    from datetime import datetime
    median = sorted(years)[len(years)//2]
    return datetime.now().year - median + 18

def fetch_multiple_movie_details(movie_ids, fetch_cache):
    """Fetch details for multiple movies using threading."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {
            executor.submit(fetch_similar_movie_details, mid, fetch_cache): mid 
            for mid in movie_ids[:50]
        }
        for future in concurrent.futures.as_completed(future_to_id):
            mid = future_to_id[future]
            try:
                result = future.result()
                if result and result[1]:
                    results[mid] = result[1]
            except:
                pass
    return results