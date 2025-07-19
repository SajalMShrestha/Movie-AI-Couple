"""
Couples Movie Recommendation Scoring Module
K-Means clustering and taste fusion algorithms
"""

import streamlit as st
import numpy as np
import torch
from sklearn.cluster import KMeans
from sentence_transformers.util import cos_sim
from collections import Counter
import concurrent.futures
import requests # Added for find_compatible_expansion_movie

# Import other modules from src - FIXED function names
from src.movie_scoring import (
    build_custom_candidate_pool,  # ✅ Correct function name
    fetch_similar_movie_details
)
from utils import get_embedding_model
from narrative_analysis import infer_narrative_style
from movie_search import fuzzy_search_movies
from tmdbv3api import Movie
movie_api = Movie()
from tmdbv3api import TMDb

def extract_movie_features(movie_titles):
    """
    Extract comprehensive features from a list of movie titles.
    
    Args:
        movie_titles: List of movie title strings
    
    Returns:
        Dictionary containing extracted features
    """
    features = {
        'genres': set(),
        'actors': set(),
        'directors': set(),
        'genre_ids': set(),
        'cast_ids': set(),
        'director_ids': set(),
        'years': [],
        'embeddings': [],
        'movies_info': [],
        'narrative_styles': {"tone": [], "complexity": [], "genre_indicator": [], "setting_context": []}
    }
    
    valid_movies = []
    
    # st.write(f"🎬 Processing {len(movie_titles)} movies...")
    
    for title in movie_titles:
        try:
            # Search for movie
            search_result = movie_api.search(title)
            
            if search_result:
                valid_movies.append((title, search_result[0]))
            else:
                # Try fuzzy search as fallback
                fuzzy_results = fuzzy_search_movies(title, max_results=1, similarity_threshold=0.7)
                if fuzzy_results:
                    best_match = fuzzy_results[0]
                    corrected_search = movie_api.search(best_match['title'])
                    if corrected_search:
                        valid_movies.append((title, corrected_search[0]))
                        # st.write(f"   📝 Used '{best_match['title']}' for '{title}'")
                    else:
                        pass
                        # st.warning(f"   ⚠️ Could not find: {title}")
                else:
                    pass
                    # st.warning(f"   ⚠️ Could not find: {title}")
        except Exception as e:
            pass
            # st.warning(f"   ❌ Error processing {title}: {e}")
    
    # Process valid movies
    for original_title, search_result in valid_movies:
        try:
            movie_id = search_result.id
            
            # Get detailed info
            details = movie_api.details(movie_id)
            credits = movie_api.credits(movie_id)
            
            movie_info = {"title": original_title, "genres": [], "year": None}
            
            # Extract genres
            genres_list = getattr(details, 'genres', [])
            for g in genres_list:
                if isinstance(g, dict):
                    name = g.get('name', '')
                    genre_id = g.get('id', 0)
                else:
                    name = getattr(g, 'name', '')
                    genre_id = getattr(g, 'id', 0)
                
                if name:
                    features['genres'].add(name)
                    movie_info["genres"].append(name)
                if genre_id:
                    features['genre_ids'].add(genre_id)
            
            # Extract cast and crew
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
                    features['actors'].add(name)
                if cast_id:
                    features['cast_ids'].add(cast_id)
            
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
                    features['directors'].add(name)
                if is_director and person_id:
                    features['director_ids'].add(person_id)
            
            # Extract year
            release_date = getattr(details, 'release_date', None)
            if release_date:
                try:
                    year = int(release_date[:4])
                    features['years'].append(year)
                    movie_info["year"] = year
                except (ValueError, TypeError):
                    pass
            
            # Extract narrative features
            overview = getattr(details, 'overview', '') or ''
            if overview:
                narrative_style = infer_narrative_style(overview)
                for key in features['narrative_styles']:
                    features['narrative_styles'][key].append(narrative_style.get(key, ""))
                
                # Generate embedding
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode(overview, convert_to_tensor=True)
                features['embeddings'].append(embedding)
            
            features['movies_info'].append(movie_info)
            
        except Exception as e:
            pass
            # st.warning(f"   ❌ Error extracting features from {original_title}: {e}")
    
    # st.write(f"   ✅ Successfully processed {len(features['movies_info'])} movies")
    return features

def perform_taste_clustering(person1_features, person2_features):
    """
    Perform K-Means clustering to find taste intersections.
    
    Args:
        person1_features: Features extracted from person 1's movies
        person2_features: Features extracted from person 2's movies
    
    Returns:
        Dictionary with clustering results and fusion strategy
    """
    # st.write("🧠 Performing taste clustering analysis...")
    
    clustering_results = {
        'person1_clusters': None,
        'person2_clusters': None,
        'fusion_embeddings': [],
        'fusion_strategy': 'hybrid',
        'taste_overlap': 0.0
    }
    
    # Combine all embeddings for analysis
    all_embeddings = person1_features['embeddings'] + person2_features['embeddings']
    
    if len(all_embeddings) < 4:
        # st.warning("   ⚠️ Not enough movies for clustering, using simple average")
        clustering_results['fusion_strategy'] = 'simple_average'
        if all_embeddings:
            clustering_results['fusion_embeddings'] = [torch.mean(torch.stack(all_embeddings), dim=0)]
        return clustering_results
    
    # Convert to numpy for clustering
    embeddings_array = torch.stack(all_embeddings).cpu().numpy()
    
    # Determine optimal clusters
    n_clusters = min(3, max(2, len(all_embeddings) // 2))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    cluster_centers = kmeans.cluster_centers_
    
    # Analyze which clusters each person belongs to
    person1_labels = cluster_labels[:len(person1_features['embeddings'])]
    person2_labels = cluster_labels[len(person1_features['embeddings']):]
    
    person1_clusters = set(person1_labels)
    person2_clusters = set(person2_labels)
    
    # Find overlapping clusters
    overlapping_clusters = person1_clusters & person2_clusters
    
    clustering_results['person1_clusters'] = person1_clusters
    clustering_results['person2_clusters'] = person2_clusters
    clustering_results['taste_overlap'] = len(overlapping_clusters) / max(len(person1_clusters | person2_clusters), 1)
    
    # st.write(f"   📊 Person 1 clusters: {person1_clusters}")
    # st.write(f"   📊 Person 2 clusters: {person2_clusters}")
    # st.write(f"   🎯 Overlapping clusters: {overlapping_clusters}")
    # st.write(f"   📈 Taste overlap: {clustering_results['taste_overlap']:.2%}")
    
    # Create fusion embeddings based on strategy
    if overlapping_clusters:
        # Strategy 1: Use overlapping cluster centers
        clustering_results['fusion_strategy'] = 'overlap_centers'
        for cluster_id in overlapping_clusters:
            center_tensor = torch.from_numpy(cluster_centers[cluster_id]).float()
            clustering_results['fusion_embeddings'].append(center_tensor)
        # st.write(f"   ✨ Using overlap centers strategy with {len(overlapping_clusters)} centers")
    
    else:
        # Strategy 2: Bridge between closest clusters
        clustering_results['fusion_strategy'] = 'bridge_clusters'
        
        # Find closest cluster centers between the two people
        person1_centers = [cluster_centers[i] for i in person1_clusters]
        person2_centers = [cluster_centers[i] for i in person2_clusters]
        
        min_distance = float('inf')
        best_pair = None
        
        for i, p1_center in enumerate(person1_centers):
            for j, p2_center in enumerate(person2_centers):
                distance = np.linalg.norm(p1_center - p2_center)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (p1_center, p2_center)
        
        if best_pair:
            # Create bridge embedding as average of closest centers
            bridge_embedding = torch.from_numpy((best_pair[0] + best_pair[1]) / 2).float()
            clustering_results['fusion_embeddings'].append(bridge_embedding)
            # st.write(f"   🌉 Using bridge strategy between closest clusters")
    
    return clustering_results

def find_sajal_compatible_movie(sajal_features, sneha_features, tmdb_api_key):
    """
    Find a movie that matches Sajal's taste but is compatible with Sneha.
    
    Returns:
        Tuple of (movie_title, score) or (None, 0) if no good match found
    """
    # st.write("🎯 Finding Sajal-focused but Sneha-compatible movie...")
    
    # Step 1: Build Sajal-focused candidate pool (100 movies)
    candidate_movie_ids = set()
    
    # Strategy 1: Genre-based discovery
    for genre_id in list(sajal_features['genre_ids'])[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 100,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:15]])
        except Exception as e:
            continue
    
    # Strategy 2: Cast-based discovery  
    for cast_id in list(sajal_features['cast_ids'])[:4]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_cast": str(cast_id),
                "sort_by": "popularity.desc", 
                "vote_count.gte": 50,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            continue
    
    # Strategy 3: Director-based discovery
    for director_id in list(sajal_features['director_ids'])[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_crew": str(director_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 50, 
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:8]])
        except Exception as e:
            continue
    
    candidate_movie_ids = list(candidate_movie_ids)[:100]
    
    if not candidate_movie_ids:
        return None, 0
    
    # Step 2: Filter by Sneha compatibility
    compatible_candidates = []
    sneha_embeddings = sneha_features['embeddings']
    
    if not sneha_embeddings:
        return None, 0
    
    fetch_cache = st.session_state.get('fetch_cache', {})
    
    for candidate_id in candidate_movie_ids[:50]:  # Process 50 for speed
        try:
            result = fetch_similar_movie_details(candidate_id, fetch_cache)
            
            if result is None or result[1] is None:
                continue
                
            mid, payload = result
            if payload is None:
                continue
                
            movie_details, candidate_embedding = payload
            movie_title = getattr(movie_details, 'title', 'Unknown')
            
            # Check similarity to Sneha's movies
            sneha_similarities = []
            for sneha_emb in sneha_embeddings:
                similarity = float(cos_sim(candidate_embedding, sneha_emb))
                sneha_similarities.append(similarity)
            
            # Count matches above 0.3 threshold
            matches_above_threshold = sum(1 for sim in sneha_similarities if sim >= 0.3)
            
            if matches_above_threshold >= 2:
                max_similarity = max(sneha_similarities)
                compatible_candidates.append((movie_title, max_similarity))
            
        except Exception as e:
            continue
    
    st.session_state['fetch_cache'] = fetch_cache
    
    if not compatible_candidates:
        # Fallback: return best available option
        if candidate_movie_ids:
            try:
                fallback_result = fetch_similar_movie_details(candidate_movie_ids[0], fetch_cache)
                if fallback_result and fallback_result[1]:
                    movie_details, _ = fallback_result[1]
                    return getattr(movie_details, 'title', 'Unknown'), 0.5
            except:
                pass
        return None, 0
    
    # Return best compatible option
    compatible_candidates.sort(key=lambda x: x[1], reverse=True)
    return compatible_candidates[0]

def find_sneha_compatible_movie(sneha_features, sajal_features, tmdb_api_key):
    """
    Find a movie that matches Sneha's taste but is compatible with Sajal.
    
    Returns:
        Tuple of (movie_title, score) or (None, 0) if no good match found
    """
    # st.write("💕 Finding Sneha-focused but Sajal-compatible movie...")
    
    # Step 1: Build Sneha-focused candidate pool (100 movies)
    candidate_movie_ids = set()
    
    # Strategy 1: Genre-based discovery
    for genre_id in list(sneha_features['genre_ids'])[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_genres": str(genre_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 100,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:15]])
        except Exception as e:
            continue
    
    # Strategy 2: Cast-based discovery  
    for cast_id in list(sneha_features['cast_ids'])[:4]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_cast": str(cast_id),
                "sort_by": "popularity.desc", 
                "vote_count.gte": 50,
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            continue
    
    # Strategy 3: Director-based discovery
    for director_id in list(sneha_features['director_ids'])[:3]:
        try:
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "with_crew": str(director_id),
                "sort_by": "popularity.desc",
                "vote_count.gte": 50, 
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:8]])
        except Exception as e:
            continue
    
    candidate_movie_ids = list(candidate_movie_ids)[:100]
    
    if not candidate_movie_ids:
        return None, 0
    
    # Step 2: Filter by Sajal compatibility
    compatible_candidates = []
    sajal_embeddings = sajal_features['embeddings']
    
    if not sajal_embeddings:
        return None, 0
    
    fetch_cache = st.session_state.get('fetch_cache', {})
    
    for candidate_id in candidate_movie_ids[:50]:  # Process 50 for speed
        try:
            result = fetch_similar_movie_details(candidate_id, fetch_cache)
            
            if result is None or result[1] is None:
                continue
                
            mid, payload = result
            if payload is None:
                continue
                
            movie_details, candidate_embedding = payload
            movie_title = getattr(movie_details, 'title', 'Unknown')
            
            # Check similarity to Sajal's movies
            sajal_similarities = []
            for sajal_emb in sajal_embeddings:
                similarity = float(cos_sim(candidate_embedding, sajal_emb))
                sajal_similarities.append(similarity)
            
            # Count matches above 0.3 threshold
            matches_above_threshold = sum(1 for sim in sajal_similarities if sim >= 0.3)
            
            if matches_above_threshold >= 2:
                max_similarity = max(sajal_similarities)
                compatible_candidates.append((movie_title, max_similarity))
            
        except Exception as e:
            continue
    
    st.session_state['fetch_cache'] = fetch_cache
    
    if not compatible_candidates:
        # Fallback: return best available option
        if candidate_movie_ids:
            try:
                fallback_result = fetch_similar_movie_details(candidate_movie_ids[0], fetch_cache)
                if fallback_result and fallback_result[1]:
                    movie_details, _ = fallback_result[1]
                    return getattr(movie_details, 'title', 'Unknown'), 0.5
            except:
                pass
        return None, 0
    
    # Return best compatible option
    compatible_candidates.sort(key=lambda x: x[1], reverse=True)
    return compatible_candidates[0]

def find_critical_darling_discovery(person1_features, person2_features, tmdb_api_key):
    """
    Find a highly-rated "critical darling" from the past 3 years that matches 
    the couple's combined preferences with tiered fallback strategy.
    
    Returns:
        Tuple of (movie_title, score) or (None, 0) if no good match found
    """
    # st.write("🏆 Finding Critical Darling Discovery movie...")
    
    # Get current year and target years (past 3 years)
    from datetime import datetime
    current_year = datetime.now().year
    
    # Tiered search strategy
    search_tiers = [
        {"rating": 7.5, "years": [current_year - 1, current_year - 2, current_year - 3], "desc": "Tier 1: 7.5+ rating, past 3 years"},
        {"rating": 7.0, "years": [current_year - 1, current_year - 2, current_year - 3], "desc": "Tier 2: 7.0+ rating, past 3 years"},
        {"rating": 7.0, "years": [current_year - 1, current_year - 2, current_year - 3, current_year - 4, current_year - 5], "desc": "Tier 3: 7.0+ rating, past 5 years"}
    ]
    
    for tier in search_tiers:
        # st.write(f"   🔍 {tier['desc']}")
        
        # Build candidate pool for this tier
        candidate_movie_ids = set()
        
        # Strategy 1: High-rated movies from target years
        for year in tier["years"]:
            try:
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": tmdb_api_key,
                    "primary_release_year": year,
                    "sort_by": "vote_average.desc",
                    "vote_count.gte": 80,
                    "vote_average.gte": tier["rating"],
                    "page": 1
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    movies = response.json().get("results", [])
                    candidate_movie_ids.update([m["id"] for m in movies[:10]])
            except Exception as e:
                continue
        
        # Strategy 2: Award-contender style films from date range
        try:
            start_year = min(tier["years"])
            end_year = max(tier["years"])
            url = f"https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": tmdb_api_key,
                "primary_release_date.gte": f"{start_year}-01-01",
                "primary_release_date.lte": f"{end_year}-12-31",
                "sort_by": "vote_average.desc",
                "vote_count.gte": 200,
                "vote_average.gte": tier["rating"],
                "page": 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies = response.json().get("results", [])
                candidate_movie_ids.update([m["id"] for m in movies[:15]])
        except Exception as e:
            pass
        
        candidate_movie_ids = list(candidate_movie_ids)[:60]
        
        if not candidate_movie_ids:
            continue
        
        # Filter and score candidates for this tier
        compatible_candidates = []
        person1_embeddings = person1_features['embeddings']
        person2_embeddings = person2_features['embeddings']
        
        if not person1_embeddings or not person2_embeddings:
            continue
        
        fetch_cache = st.session_state.get('fetch_cache', {})
        
        for candidate_id in candidate_movie_ids[:40]:
            try:
                result = fetch_similar_movie_details(candidate_id, fetch_cache)
                
                if result is None or result[1] is None:
                    continue
                    
                mid, payload = result
                if payload is None:
                    continue
                    
                movie_details, candidate_embedding = payload
                movie_title = getattr(movie_details, 'title', 'Unknown')
                vote_average = getattr(movie_details, 'vote_average', 0) or 0
                vote_count = getattr(movie_details, 'vote_count', 0) or 0
                
                # Skip if not meeting tier criteria
                if vote_average < tier["rating"] or vote_count < 80:
                    continue
                
                # Check compatibility with both people
                person1_similarities = []
                for p1_emb in person1_embeddings:
                    similarity = float(cos_sim(candidate_embedding, p1_emb))
                    person1_similarities.append(similarity)
                
                person2_similarities = []
                for p2_emb in person2_embeddings:
                    similarity = float(cos_sim(candidate_embedding, p2_emb))
                    person2_similarities.append(similarity)
                
                # Calculate compatibility scores
                person1_max_sim = max(person1_similarities) if person1_similarities else 0
                person2_max_sim = max(person2_similarities) if person2_similarities else 0
                
                # Require decent compatibility with both (0.2 threshold for discovery)
                if person1_max_sim >= 0.2 and person2_max_sim >= 0.2:
                    combined_compatibility = (person1_max_sim + person2_max_sim) / 2
                    quality_score = vote_average / 10
                    discovery_score = (combined_compatibility * 0.4) + (quality_score * 0.6)
                    
                    compatible_candidates.append((movie_title, discovery_score, vote_average))
                
            except Exception as e:
                continue
        
        st.session_state['fetch_cache'] = fetch_cache
        
        # If we found compatible candidates in this tier, return the best one
        if compatible_candidates:
            compatible_candidates.sort(key=lambda x: x[1], reverse=True)
            best_movie, best_score, rating = compatible_candidates[0]
            
            # st.write(f"   🎬 Critical Darling found: {best_movie} (Rating: {rating:.1f})")
            return best_movie, best_score
    
    # If all tiers failed
    # st.write("   ⚠️ No critical darling found meeting compatibility criteria")
    return None, 0

def find_decade_time_machine_movie(person1_features, person2_features, tmdb_api_key):
    """
    Find a hidden gem from the decade least represented in their favorites.
    Focuses on 1990s-2020s, high ratings (7.0+), low popularity (<2K votes), ranks #5-15.
    
    Returns:
        Tuple of (movie_title, score) or (None, 0) if no good match found
    """
    # st.write("⏰ Finding Decade Time Machine movie...")
    
    # Analyze decade representation from both people's favorites
    all_favorite_years = person1_features['years'] + person2_features['years']
    
    decade_counts = {
        "1990s": 0,
        "2000s": 0, 
        "2010s": 0,
        "2020s": 0
    }
    
    # Count movies per decade
    for year in all_favorite_years:
        if 1990 <= year <= 1999:
            decade_counts["1990s"] += 1
        elif 2000 <= year <= 2009:
            decade_counts["2000s"] += 1
        elif 2010 <= year <= 2019:
            decade_counts["2010s"] += 1
        elif 2020 <= year <= 2029:
            decade_counts["2020s"] += 1
    
    # st.write(f"   📊 Decade representation: {decade_counts}")
    
    # Priority order: least to most represented
    sorted_decades = sorted(decade_counts.items(), key=lambda x: x[1])
    decade_priority = [decade for decade, count in sorted_decades]
    
    # st.write(f"   🎯 Decade priority order: {decade_priority}")
    
    # Rating thresholds for fallback
    rating_thresholds = [7.0, 6.5, 6.0]
    
    # Try each decade in priority order
    for target_decade in decade_priority:
        # st.write(f"   🔍 Searching {target_decade}...")
        
        # Set decade year range
        if target_decade == "1990s":
            start_year, end_year = 1990, 1999
        elif target_decade == "2000s":
            start_year, end_year = 2000, 2009
        elif target_decade == "2010s":
            start_year, end_year = 2010, 2019
        elif target_decade == "2020s":
            start_year, end_year = 2020, 2029
        
        # Try different rating thresholds
        for min_rating in rating_thresholds:
            # st.write(f"     📈 Trying rating threshold: {min_rating}+")
            
            try:
                # Search for hidden gems from this decade
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": tmdb_api_key,
                    "primary_release_date.gte": f"{start_year}-01-01",
                    "primary_release_date.lte": f"{end_year}-12-31