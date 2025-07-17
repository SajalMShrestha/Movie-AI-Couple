"""
Movie recommendation scoring and candidate pool building.
"""

import streamlit as st
import requests
import concurrent.futures
import numpy as np
import torch
from datetime import datetime
from tmdbv3api import Movie
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers.util import cos_sim

from utils import (
    RECOMMENDATION_WEIGHTS, get_embedding_model, get_mood_score, 
    get_trending_popularity, estimate_user_age, fetch_similar_movie_details
)
from narrative_analysis import infer_narrative_style, infer_mood_from_plot, compute_narrative_similarity
from franchise_detection import apply_final_franchise_limit
from movie_search import fuzzy_search_movies

def build_custom_candidate_pool(favorite_genre_ids, favorite_cast_ids, favorite_director_ids, favorite_years, tmdb_api_key):
    """
    Build a pool of 100-200 candidate movies using custom criteria.
    
    Args:
        favorite_genre_ids: Set of favorite genre IDs
        favorite_cast_ids: Set of favorite cast member IDs
        favorite_director_ids: Set of favorite director IDs
        favorite_years: List of favorite movie years
        tmdb_api_key: TMDB API key
    
    Returns:
        Set of candidate movie IDs
    """
    candidate_movie_ids = set()
    
    # Strategy 1: Discover by Genre (40-60 movies)
    for genre_id in list(favorite_genre_ids)[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 50,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:20]])
        except Exception as e:
            st.warning(f"Error discovering by genre {genre_id}: {e}")
    
    # Strategy 2: Discover by Cast (30-40 movies)
    for person_id in list(favorite_cast_ids)[:5]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_cast": str(person_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 30,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:8]])
        except Exception as e:
            st.warning(f"Error discovering by cast {person_id}: {e}")
    
    # Strategy 3: Discover by Directors (20-30 movies)
    for person_id in list(favorite_director_ids)[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_crew": str(person_id),
                "sort_by": "popularity.desc", 
                "vote_count.gte": 30,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            st.warning(f"Error discovering by director {person_id}: {e}")
    
    # Strategy 4: Year-based Discovery (20-30 movies)
    if favorite_years:
        decades = set()
        for year in favorite_years:
            decade_start = (year // 10) * 10
            decades.add(decade_start)
        
        for decade_start in list(decades)[:2]:
            try:
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": tmdb_api_key,
                    "primary_release_date.gte": f"{decade_start}-01-01",
                    "primary_release_date.lte": f"{decade_start + 9}-12-31",
                    "sort_by": "vote_average.desc",
                    "vote_count.gte": 100,
                    "page": 1
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    movies = response.json().get("results", [])
                    candidate_movie_ids.update([m["id"] for m in movies[:15]])
            except Exception as e:
                st.warning(f"Error discovering by decade {decade_start}: {e}")
    
    # Strategy 5: Multi-criteria Discovery (20-30 movies)
    try:
        top_genres = ",".join(str(id) for id in list(favorite_genre_ids)[:2])
        top_cast = ",".join(str(id) for id in list(favorite_cast_ids)[:3])
        
        if top_genres and top_cast:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": top_genres,
                "with_cast": top_cast,
                "sort_by": "popularity.desc",
                "vote_count.gte": 20,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:20]])
    except Exception as e:
        st.warning(f"Error with multi-criteria discovery: {e}")
    
    # Strategy 6: Trending/Popular Backup (10-20 movies)
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week"
        params = {"api_key": tmdb_api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            movies = response.json().get("results", [])
            candidate_movie_ids.update([m["id"] for m in movies[:15]])
    except Exception as e:
        st.warning(f"Error getting trending movies: {e}")
    
    # Strategy 7: High-rated movies in favorite genres (backup)
    for genre_id in list(favorite_genre_ids)[:2]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "vote_average.desc",
                "vote_count.gte": 200,
                "vote_average.gte": 7.0,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            st.warning(f"Error getting high-rated movies for genre {genre_id}: {e}")
    
    return candidate_movie_ids

def identify_taste_clusters(favorite_embeddings, favorite_movies_info):
    """
    Identify distinct taste clusters from user's favorite movies.
    
    Args:
        favorite_embeddings: List of movie embeddings
        favorite_movies_info: List of movie info dictionaries
    
    Returns:
        Tuple of (cluster_centers, cluster_labels)
    """
    if len(favorite_embeddings) <= 2:
        return None, None
    
    # Convert embeddings to numpy array
    embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
    
    # Determine optimal number of clusters
    n_clusters = min(3, max(2, len(favorite_embeddings) // 2))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    cluster_centers = kmeans.cluster_centers_
    
    # Convert back to torch tensors
    cluster_centers_torch = [torch.from_numpy(center) for center in cluster_centers]
    
    return cluster_centers_torch, cluster_labels

def compute_multi_cluster_similarity(candidate_embedding, cluster_centers):
    """
    Compute similarity to multiple cluster centers.
    
    Args:
        candidate_embedding: Embedding of candidate movie
        cluster_centers: List of cluster center embeddings
    
    Returns:
        Float: Maximum similarity to any cluster
    """
    if cluster_centers is None:
        return 0.0
    
    max_similarity = 0.0
    for center in cluster_centers:
        similarity = float(cos_sim(candidate_embedding, center))
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years):
    """
    Analyze how diverse the user's taste is.
    
    Args:
        favorite_embeddings: List of movie embeddings
        favorite_genres: Set of favorite genres
        favorite_years: List of favorite years
    
    Returns:
        Dictionary with diversity metrics and taste profile
    """
    diversity_metrics = {
        "genre_diversity": len(favorite_genres) / 5.0,
        "temporal_spread": 0.0,
        "embedding_variance": 0.0,
        "taste_profile": "focused"
    }
    
    # Temporal spread
    if len(favorite_years) > 1:
        year_range = max(favorite_years) - min(favorite_years)
        diversity_metrics["temporal_spread"] = min(year_range / 50.0, 1.0)
    
    # Embedding variance
    if len(favorite_embeddings) > 1:
        embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
        pairwise_similarities = sklearn_cosine_similarity(embeddings_array)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        avg_similarity = pairwise_similarities[mask].mean()
        diversity_metrics["embedding_variance"] = 1.0 - avg_similarity
    
    # Determine taste profile
    overall_diversity = (diversity_metrics["genre_diversity"] + 
                        diversity_metrics["temporal_spread"] + 
                        diversity_metrics["embedding_variance"]) / 3.0
    
    if overall_diversity < 0.3:
        diversity_metrics["taste_profile"] = "focused"
    elif overall_diversity < 0.6:
        diversity_metrics["taste_profile"] = "diverse"
    else:
        diversity_metrics["taste_profile"] = "eclectic"
    
    return diversity_metrics

def compute_score(m, cluster_centers, diversity_metrics, favorite_genres, favorite_actors, 
                 user_prefs, trending_scores, favorite_narrative_styles, candidate_movies):
    """
    Compute recommendation score for a movie.
    
    Args:
        m: Movie object
        cluster_centers: Taste cluster centers
        diversity_metrics: User taste diversity metrics
        favorite_genres: Set of favorite genres
        favorite_actors: Set of favorite actors
        user_prefs: User preferences dictionary
        trending_scores: Trending popularity scores
        favorite_narrative_styles: Favorite narrative styles
        candidate_movies: Dictionary of candidate movies
    
    Returns:
        Float: Recommendation score
    """
    try:
        narrative = getattr(m, 'narrative_style', {})
        score = 0.0
        
        # Genre similarity
        genres = set()
        genres_list = getattr(m, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                genres.add(name)
        
        score += RECOMMENDATION_WEIGHTS['genre_similarity'] * (len(genres & favorite_genres) / max(len(favorite_genres),1))
        
        # Cast and crew similarity
        cast_names = set()
        cast_list = getattr(m, 'cast', [])
        for actor in cast_list:
            if isinstance(actor, dict):
                name = actor.get('name', '')
            else:
                name = getattr(actor, 'name', '')
            if name:
                cast_names.add(name)
        
        directors = getattr(m, 'directors', [])
        director_names = set(directors) if isinstance(directors, list) else set()
        
        cast_dir = cast_names | director_names
        score += RECOMMENDATION_WEIGHTS['cast_crew'] * (len(cast_dir & favorite_actors) / max(len(favorite_actors),1))
        
        # Release year scoring
        try:
            release_date = getattr(m, 'release_date', None)
            if release_date:
                year_diff = datetime.now().year - int(release_date[:4])
                if year_diff<=2: score += RECOMMENDATION_WEIGHTS['release_year']*0.6
                elif year_diff<=5: score += RECOMMENDATION_WEIGHTS['release_year']*0.4
                elif year_diff<=10: score += RECOMMENDATION_WEIGHTS['release_year']*0.25
                elif year_diff<=20: score += RECOMMENDATION_WEIGHTS['release_year']*0.1
        except (ValueError, TypeError, AttributeError):
            pass
        
        # Ratings score
        vote_average = getattr(m, 'vote_average', 0) or 0
        score += RECOMMENDATION_WEIGHTS['ratings'] * (vote_average/10)
        
        # Mood/tone score
        movie_genres = getattr(m, 'genres', [])
        score += RECOMMENDATION_WEIGHTS['mood_tone'] * get_mood_score(movie_genres, user_prefs['preferred_moods'])

        # Narrative style score
        plot = getattr(m, 'plot', '') or getattr(m, 'overview', '') or ''
        narrative = infer_narrative_style(plot)
        narrative_match_score = compute_narrative_similarity(narrative, favorite_narrative_styles)
        score += RECOMMENDATION_WEIGHTS['narrative_style'] * narrative_match_score

        # Embedding similarity score
        movie_id = getattr(m, 'id', None)
        if movie_id and movie_id in candidate_movies:
            candidate_data = candidate_movies[movie_id]
            if len(candidate_data) >= 2 and candidate_data[1] is not None:
                candidate_embedding = candidate_data[1]
                
                # Use multi-cluster similarity for diverse tastes
                if cluster_centers and diversity_metrics['taste_profile'] in ['diverse', 'eclectic']:
                    embedding_sim_score = compute_multi_cluster_similarity(candidate_embedding, cluster_centers)
                else:
                    # For focused tastes, use average embedding
                    embedding_model = get_embedding_model()
                    if hasattr(user_prefs, 'favorite_embeddings') and user_prefs['favorite_embeddings']:
                        avg_embedding = torch.mean(torch.stack(user_prefs['favorite_embeddings']), dim=0)
                        embedding_sim_score = float(cos_sim(candidate_embedding, avg_embedding))
                    else:
                        embedding_sim_score = 0.0
                
                score += RECOMMENDATION_WEIGHTS['embedding_similarity'] * embedding_sim_score

        # Trending factor
        movie_trend_score = trending_scores.get(getattr(m, 'id', 0), 0)
        mood_match_score = get_mood_score(movie_genres, user_prefs['preferred_moods'])
        genre_overlap_score = len(genres & favorite_genres) / max(len(favorite_genres), 1)

        # Adjust trending boost based on taste diversity
        if diversity_metrics['taste_profile'] == 'eclectic':
            trending_weight = RECOMMENDATION_WEIGHTS['trending_factor'] * 1.5
        elif diversity_metrics['taste_profile'] == 'focused':
            if mood_match_score > 0.5 and genre_overlap_score > 0.4:
                trending_weight = RECOMMENDATION_WEIGHTS['trending_factor']
            else:
                trending_weight = 0
        else:
            if mood_match_score > 0.3 and genre_overlap_score > 0.2:
                trending_weight = RECOMMENDATION_WEIGHTS['trending_factor']
            else:
                trending_weight = 0
        
        score += trending_weight * movie_trend_score

        # Discovery boost for eclectic users
        if diversity_metrics['taste_profile'] == 'eclectic':
            if 0.1 < genre_overlap_score < 0.5:
                score += RECOMMENDATION_WEIGHTS['discovery_boost'] * 1.5

        # Age penalty for very old movies
        try:
            if release_date:
                release_year = int(release_date[:4])
                if datetime.now().year - release_year > 20:
                    score -= 0.03
        except (ValueError, TypeError):
            pass

        # Age alignment scoring
        try:
            if release_date:
                release_year = int(release_date[:4])
                user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
                if 15 <= user_age_at_release <= 25:
                    score += RECOMMENDATION_WEIGHTS['age_alignment']
                elif 10 <= user_age_at_release < 15 or 25 < user_age_at_release <= 30:
                    score += RECOMMENDATION_WEIGHTS['age_alignment'] * 0.5
        except (ValueError, TypeError):
            pass
            
        return max(score, 0)
    except Exception as e:
        st.warning(f"Error computing score for movie: {e}")
        return 0

def recommend_movies(favorite_titles):
    """
    Main recommendation function that processes user's favorite movies and returns recommendations.
    
    Args:
        favorite_titles: List of user's favorite movie titles
    
    Returns:
        Tuple of (recommendations, candidate_movies)
    """
    # Check cache first
    cache_key = "|".join(sorted(favorite_titles))
    
    if cache_key in st.session_state.recommendation_cache:
        cached_result = st.session_state.recommendation_cache[cache_key]
        st.write(f"✅ Using cached results")
        return cached_result
    
    # Initialize collections
    favorite_genres = set()
    favorite_actors = set()
    favorite_directors = set()
    favorite_genre_ids = set()
    favorite_cast_ids = set()
    favorite_director_ids = set()
    plot_moods, favorite_years = set(), []
    favorite_narrative_styles = {"tone": [], "complexity": [], "genre_indicator": [], "setting_context": []}
    favorite_embeddings = []
    favorite_movies_info = []

    # Process favorite movies with fuzzy search fallback
    valid_movies_found = []
    failed_searches = []
    movie_api = Movie()

    for title in favorite_titles:
        try:
            search_result = movie_api.search(title)
            
            if search_result:
                valid_movies_found.append((title, search_result[0]))
            else:
                # Try fuzzy search for this title
                st.write(f"🔍 Trying fuzzy search for '{title}'...")
                fuzzy_results = fuzzy_search_movies(title, max_results=3, similarity_threshold=0.7)
                
                if fuzzy_results:
                    best_match = fuzzy_results[0]
                    st.write(f"📝 Using '{best_match['title']}' as match for '{title}' ({best_match['similarity']:.0%} similarity)")
                    
                    corrected_search = movie_api.search(best_match['title'])
                    if corrected_search:
                        valid_movies_found.append((title, corrected_search[0]))
                    else:
                        failed_searches.append(title)
                else:
                    failed_searches.append(title)
                    
        except Exception as e:
            st.warning(f"Error processing {title}: {e}")
            failed_searches.append(title)

    # Show search results
    if failed_searches:
        st.warning(f"⚠️ Could not find matches for: {', '.join(failed_searches)}")
        st.info("💡 Try using more common titles or check spelling for better results")

    if len(valid_movies_found) < 3:
        st.error("❌ Need at least 3 valid movies to generate good recommendations")
        st.info("💡 Please add more movies or try different titles")
        return [], {}

    # Process valid movies and extract features
    for original_title, search_result in valid_movies_found:
        try:
            movie_id = search_result.id
            
            # Check cache
            if movie_id in st.session_state.movie_details_cache:
                details = st.session_state.movie_details_cache[movie_id]
                credits = st.session_state.movie_credits_cache[movie_id]
            else:
                details = movie_api.details(movie_id)
                credits = movie_api.credits(movie_id)
                st.session_state.movie_details_cache[movie_id] = details
                st.session_state.movie_credits_cache[movie_id] = credits
            
            # Extract features (genres, cast, directors, etc.)
            movie_info = {"title": original_title, "genres": [], "year": None}
            
            # Process genres
            genres_list = getattr(details, 'genres', [])
            for g in genres_list:
                if isinstance(g, dict):
                    name = g.get('name', '')
                    if hasattr(g, 'id'):
                        favorite_genre_ids.add(g.id)
                    elif 'id' in g:
                        favorite_genre_ids.add(g['id'])
                else:
                    name = getattr(g, 'name', '')
                    if hasattr(g, 'id'):
                        favorite_genre_ids.add(g.id)
                if name:
                    favorite_genres.add(name)
                    movie_info["genres"].append(name)

            # Process cast and crew
            cast_list_raw = credits.get('cast', []) if isinstance(credits, dict) else getattr(credits, 'cast', [])
            crew_list = credits.get('crew', []) if isinstance(credits, dict) else getattr(credits, 'crew', [])
            
            if hasattr(cast_list_raw, '__iter__'):
                cast_list = list(cast_list_raw)[:3] if cast_list_raw else []
            else:
                cast_list = []
            
            for c in cast_list:
                if isinstance(c, dict):
                    name = c.get('name', '')
                    cast_id = c.get('id', 0)
                else:
                    name = getattr(c, 'name', '')
                    cast_id = getattr(c, 'id', 0)
                if name:
                    favorite_actors.add(name)
                if cast_id:
                    favorite_cast_ids.add(cast_id)

            for c in crew_list:
                is_director = False
                name = ''
                person_id = 0
                if isinstance(c, dict):
                    is_director = c.get('job', '') == 'Director'
                    name = c.get('name', '')
                    person_id = c.get('id', 0)
                else:
                    is_director = getattr(c, 'job', '') == 'Director'
                    name = getattr(c, 'name', '')
                    person_id = getattr(c, 'id', 0)
                
                if is_director and name:
                    favorite_directors.add(name)
                if is_director and person_id:
                    favorite_director_ids.add(person_id)

            # Process plot and narrative
            overview = getattr(details, 'overview', '') or ''
            plot_moods.add(infer_mood_from_plot(overview))
            narr_style = infer_narrative_style(overview)
            for key in favorite_narrative_styles:
                favorite_narrative_styles[key].append(narr_style.get(key, ""))
            
            # Process release date
            release_date = getattr(details, 'release_date', None)
            if release_date:
                try:
                    year = int(release_date[:4])
                    favorite_years.append(year)
                    movie_info["year"] = year
                except (ValueError, TypeError):
                    pass
            
            # Generate embedding
            embedding_model = get_embedding_model()
            emb = embedding_model.encode(overview, convert_to_tensor=True)
            favorite_embeddings.append(emb)
            favorite_movies_info.append(movie_info)
                
        except Exception as e:
            st.warning(f"Error processing {original_title}: {e}")
            continue

    # Build candidate pool and analyze taste
    from tmdbv3api import TMDb
    tmdb = TMDb()
    candidate_movie_ids = build_custom_candidate_pool(
        favorite_genre_ids, favorite_cast_ids, favorite_director_ids, 
        favorite_years, tmdb.api_key
    )

    # Limit candidates
    candidate_movie_ids = list(candidate_movie_ids)[:150]

    # Analyze taste diversity
    diversity_metrics = analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years)
    
    # Identify taste clusters
    cluster_centers, cluster_labels = identify_taste_clusters(favorite_embeddings, favorite_movies_info)

    # Set up user preferences
    user_prefs = {
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years),
        "taste_diversity": diversity_metrics,
        "favorite_embeddings": favorite_embeddings
    }

    # Fetch candidate movie details
    candidate_movies = {}
    fetch_cache = st.session_state.fetch_cache
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid, fetch_cache): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result is None:
                    continue
                mid, payload = result
                if payload is None:
                    continue
                m, embedding = payload
                if m is None or embedding is None:
                    continue
                vote_count = getattr(m, 'vote_count', 0)
                if vote_count < 20:
                    continue
                candidate_movies[mid] = (m, embedding)
            except Exception as e:
                st.warning(f"Error processing candidate movie: {e}")
                continue

    # Update session cache
    st.session_state.fetch_cache.update(fetch_cache)

    if not candidate_movies:
        st.warning("No candidate movies with valid plots or embeddings were found.")
        return [], {}

    # Get trending scores
    trending_scores = get_trending_popularity(tmdb.api_key)

    # Score all candidate movies
    scored = []
    for movie_obj, embedding in candidate_movies.values():
        if movie_obj is None or embedding is None:
            continue
        try:
            score = compute_score(
                movie_obj, cluster_centers, diversity_metrics, favorite_genres, 
                favorite_actors, user_prefs, trending_scores, favorite_narrative_styles, 
                candidate_movies
            )
            vote_count = getattr(movie_obj, 'vote_count', 0)
            score += min(vote_count, 500) / 50000
            scored.append((movie_obj, score))
        except Exception as e:
            st.warning(f"Error scoring movie {getattr(movie_obj, 'title', 'Unknown')}: {e}")
            continue

    scored.sort(key=lambda x:x[1], reverse=True)
    
    # Apply diversity and filtering
    top = []
    low_votes = 0
    used_genres = set()
    favorite_titles_set = {title.lower() for title in favorite_titles}

    for m, s in scored:
        vote_count = getattr(m, 'vote_count', 0)
        movie_title = getattr(m, 'title', 'Unknown Title')
        
        # Skip if this movie is in the user's favorites
        if movie_title.lower() in favorite_titles_set:
            continue
        
        # Get movie genres for diversity
        movie_genres = set()
        genres_list = getattr(m, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                movie_genres.add(name)
        
        # For eclectic users, ensure genre diversity
        if diversity_metrics['taste_profile'] == 'eclectic' and len(top) >= 3:
            genre_overlap = len(movie_genres & used_genres) / max(len(movie_genres), 1)
            if genre_overlap > 0.7:
                continue
        
        if vote_count < 100:
            if low_votes >= 2: 
                continue
            low_votes += 1
        
        top.append((movie_title, s))
        used_genres.update(movie_genres)
        
        if len(top) == 10: 
            break

    # Apply final franchise limiting
    franchise_limited_top = apply_final_franchise_limit(top, candidate_movies, max_per_franchise=1)

    # Cache and return result
    result = (franchise_limited_top, candidate_movies)
    st.session_state.recommendation_cache[cache_key] = result
    return result