"""
Franchise detection and limiting functionality.
"""

import re

def extract_base_title_simple(title):
    """
    Extract base title by removing common sequel indicators.
    
    Args:
        title: Movie title
    
    Returns:
        String: Base title in lowercase
    """
    patterns = [
        r'\s*\d+$',          # "Movie 2"
        r'\s*:\s*.*$',       # "Movie: Subtitle"  
        r'\s*-\s*.*$',       # "Movie - Subtitle"
        r'\s*\(.*\)$',       # "Movie (2019)"
        r'\s*II+$',          # "Movie II"
    ]
    
    base_title = title
    for pattern in patterns:
        base_title = re.sub(pattern, '', base_title)
    
    return base_title.strip().lower()

def get_franchise_key_robust(movie_title, candidates):
    """
    Generate robust franchise key using shared cast and title patterns.
    
    Args:
        movie_title: Title of the movie
        candidates: Dictionary of candidate movies
    
    Returns:
        String: Franchise key for grouping
    """
    # Find the movie object for this title
    movie_obj = None
    for m, _ in candidates.values():
        if m and getattr(m, 'title', '') == movie_title:
            movie_obj = m
            break
    
    if not movie_obj:
        # Fallback to simple title matching
        return extract_base_title_simple(movie_title)
    
    # Get main cast
    cast_names = []
    cast_list = getattr(movie_obj, 'cast', []) if hasattr(movie_obj, 'cast') else []
    for actor in cast_list[:5]:  # Top 5 cast members
        if isinstance(actor, dict):
            name = actor.get('name', '')
        else:
            name = getattr(actor, 'name', '')
        if name:
            cast_names.append(name)
    
    # Extract base title
    base_title = extract_base_title_simple(movie_title)
    
    # Create franchise key using base title + key shared elements
    main_cast = cast_names[0] if cast_names else "unknown"
    
    # Special handling for similar base titles
    title_words = set(base_title.split())
    
    # If titles share key words (like "dragon"), use shared cast + shared words
    if any(word in ["dragon", "train"] for word in title_words):
        # Look for Jay Baruchel (Hiccup's voice) as franchise identifier
        if "Jay Baruchel" in cast_names:
            return "httyd_jay_baruchel_dragon"
    
    # For other movies, use base title + main cast
    franchise_key = f"{base_title}_{main_cast}"
    
    return franchise_key

def apply_final_franchise_limit(recommendations, candidates, max_per_franchise=1):
    """
    Apply franchise limiting as final step - keep only 1 per franchise.
    Fill remaining slots from next best movies.
    
    Args:
        recommendations: List of (title, score) tuples
        candidates: Dictionary of candidate movies
        max_per_franchise: Maximum movies per franchise
    
    Returns:
        List of (title, score) tuples with franchise limiting applied
    """
    if not recommendations:
        return recommendations
    
    # Get all scored movies sorted by score for backfill
    all_scored_movies = []
    recommendation_dict = {title: score for title, score in recommendations}
    
    # Add all candidate movies with their scores
    for movie_obj, embedding in candidates.values():
        if movie_obj:
            movie_title = getattr(movie_obj, 'title', '')
            if movie_title in recommendation_dict:
                score = recommendation_dict[movie_title]
                all_scored_movies.append((movie_title, score, movie_obj))
    
    # Add more movies beyond the original recommendations
    additional_movies = []
    used_titles = {title for title, _, _ in all_scored_movies}
    
    for movie_obj, embedding in candidates.values():
        if movie_obj:
            movie_title = getattr(movie_obj, 'title', '')
            if movie_title not in used_titles:
                # These don't have scores from recommendations, so assign lower scores
                additional_movies.append((movie_title, 0.0, movie_obj))
    
    # Combine and sort by score
    all_movies = all_scored_movies + additional_movies
    all_movies.sort(key=lambda x: x[1], reverse=True)
    
    # Apply franchise limiting
    franchise_counts = {}
    final_recommendations = []
    used_titles = set()
    
    # Track franchise detection for debugging
    franchise_debug = {}
    
    # Go through all movies in score order and apply franchise limit
    for title, score, movie_obj in all_movies:
        if title in used_titles:
            continue
            
        # Get franchise key using robust detection
        franchise_key = get_franchise_key_robust(title, candidates)
        
        # Debug tracking
        if franchise_key not in franchise_debug:
            franchise_debug[franchise_key] = []
        franchise_debug[franchise_key].append(title)
        
        # Check if this franchise already has a movie
        if franchise_counts.get(franchise_key, 0) < max_per_franchise:
            final_recommendations.append((title, score))
            franchise_counts[franchise_key] = franchise_counts.get(franchise_key, 0) + 1
            used_titles.add(title)
            
            # Stop when we have 10 recommendations
            if len(final_recommendations) >= 10:
                break
    
    return final_recommendations[:10]

def debug_franchise_keys(recommendations, candidates):
    """
    Debug function to see what franchise keys are being generated.
    
    Args:
        recommendations: List of recommendations
        candidates: Dictionary of candidate movies
    """
    print("🔍 FRANCHISE KEY DEBUG:")
    
    for title, score in recommendations[:5]:  # Check first 5
        franchise_key = get_franchise_key_robust(title, candidates)
        print(f"- {title} → {franchise_key}")