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
    
    st.write(f"🎬 Processing {len(movie_titles)} movies...")
    
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
                        st.write(f"   📝 Used '{best_match['title']}' for '{title}'")
                    else:
                        st.warning(f"   ⚠️ Could not find: {title}")
                else:
                    st.warning(f"   ⚠️ Could not find: {title}")
        except Exception as e:
            st.warning(f"   ❌ Error processing {title}: {e}")
    
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
            st.warning(f"   ❌ Error extracting features from {original_title}: {e}")
    
    st.write(f"   ✅ Successfully processed {len(features['movies_info'])} movies")
    return features

def find_compatible_expansion_movie(person_features, partner_features, person_name, tmdb_api_key, similarity_threshold=0.3, min_partner_matches=2):
    """
    Find a 6th/7th movie that extends person's taste while being compatible with partner.
    
    Args:
        person_features: Features extracted from person's 5 movies
        partner_features: Features extracted from partner's 5 movies  
        person_name: "Sajal" or "Sneha" for logging
        tmdb_api_key: TMDB API key
        similarity_threshold: Minimum cosine similarity required (default 0.3)
        min_partner_matches: Must be similar to at least this many partner movies
    
    Returns:
        Tuple of (movie_title, movie_details) or (None, None) if no good match found
    """
    st.write(f"🎯 Finding compatible expansion movie for {person_name}...")
    
    # Step 1: Build person-focused candidate pool (50 movies)
    st.write(f"📚 Building {person_name}-focused candidate pool...")
    candidate_movie_ids = set()
    
    # Strategy 1: Genre-based discovery (20 movies)
    for genre_id in list(person_features['genre_ids'])[:2]:  # Top 2 genres
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
                candidate_movie_ids.update([m["id"] for m in movies[:10]])
        except Exception as e:
            st.warning(f"Error with genre discovery: {e}")
    
    # Strategy 2: Cast-based discovery (20 movies)  
    for cast_id in list(person_features['cast_ids'])[:3]:  # Top 3 actors
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
                candidate_movie_ids.update([m["id"] for m in movies[:7]])
        except Exception as e:
            st.warning(f"Error with cast discovery: {e}")
    
    # Strategy 3: Director-based discovery (10 movies)
    for director_id in list(person_features['director_ids'])[:2]:  # Top 2 directors
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
                candidate_movie_ids.update([m["id"] for m in movies[:5]])
        except Exception as e:
            st.warning(f"Error with director discovery: {e}")
    
    # Limit to 50 candidates total
    candidate_movie_ids = list(candidate_movie_ids)[:50]
    st.write(f"   📋 Found {len(candidate_movie_ids)} {person_name}-focused candidates")
    
    if not candidate_movie_ids:
        st.warning(f"No candidates found for {person_name}")
        return None, None
    
    # Step 2: Filter by partner compatibility
    st.write(f"💕 Filtering by partner compatibility...")
    compatible_candidates = []
    
    # Get partner embeddings for similarity comparison
    partner_embeddings = partner_features['embeddings']
    if not partner_embeddings:
        st.warning("No partner embeddings available for compatibility check")
        return None, None
    
    # Fetch candidate movie details and check compatibility
    tmdb = TMDb()
    for mid in candidate_movie_ids:
        try:
            movie_details = tmdb.movie(mid)
            if movie_details:
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode(movie_details.overview, convert_to_tensor=True)
                
                # Check similarity to partner's embeddings
                is_compatible = False
                for partner_emb in partner_embeddings:
                    similarity = float(cos_sim(embedding, partner_emb))
                    if similarity >= similarity_threshold:
                        is_compatible = True
                        break
                
                if is_compatible:
                    movie_title = movie_details.title
                    st.write(f"✅ Found compatible movie: {movie_title}")
                    compatible_candidates.append((movie_title, movie_details))
        
        except Exception as e:
            st.warning(f"Error fetching movie details for ID {mid}: {e}")
    
    st.write(f"   📋 Found {len(compatible_candidates)} compatible expansion movies")
    
    if len(compatible_candidates) < min_partner_matches:
        st.warning(f"Not enough compatible movies found for {person_name}. Found {len(compatible_candidates)}, required {min_partner_matches}")
        return None, None
    
    # Select the best compatible movie
    best_match = None
    max_similarity_sum = -1
    
    for movie_title, movie_details in compatible_candidates:
        embedding_model = get_embedding_model()
        embedding = embedding_model.encode(movie_details.overview, convert_to_tensor=True)
        
        similarity_sum = 0
        for partner_emb in partner_embeddings:
            similarity_sum += float(cos_sim(embedding, partner_emb))
        
        if similarity_sum > max_similarity_sum:
            max_similarity_sum = similarity_sum
            best_match = (movie_title, movie_details)
    
    st.write(f"✅ Best compatible expansion movie found: {best_match[0]} (Similarity: {max_similarity_sum:.2f})")
    return best_match

def perform_taste_clustering(person1_features, person2_features):
    """
    Perform K-Means clustering to find taste intersections.
    
    Args:
        person1_features: Features extracted from person 1's movies
        person2_features: Features extracted from person 2's movies
    
    Returns:
        Dictionary with clustering results and fusion strategy
    """
    st.write("🧠 Performing taste clustering analysis...")
    
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
        st.warning("   ⚠️ Not enough movies for clustering, using simple average")
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
    
    st.write(f"   📊 Person 1 clusters: {person1_clusters}")
    st.write(f"   📊 Person 2 clusters: {person2_clusters}")
    st.write(f"   🎯 Overlapping clusters: {overlapping_clusters}")
    st.write(f"   📈 Taste overlap: {clustering_results['taste_overlap']:.2%}")
    
    # Create fusion embeddings based on strategy
    if overlapping_clusters:
        # Strategy 1: Use overlapping cluster centers
        clustering_results['fusion_strategy'] = 'overlap_centers'
        for cluster_id in overlapping_clusters:
            center_tensor = torch.from_numpy(cluster_centers[cluster_id]).float()
            clustering_results['fusion_embeddings'].append(center_tensor)
        st.write(f"   ✨ Using overlap centers strategy with {len(overlapping_clusters)} centers")
    
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
            st.write(f"   🌉 Using bridge strategy between closest clusters")
    
    return clustering_results

def compute_couple_compatibility_score(candidate_embedding, fusion_embeddings, fusion_strategy):
    """
    Compute how well a candidate movie matches the couple's fused taste.
    
    Args:
        candidate_embedding: Embedding of candidate movie
        fusion_embeddings: List of fusion embeddings from clustering
        fusion_strategy: Strategy used for fusion
    
    Returns:
        Float compatibility score
    """
    if not fusion_embeddings:
        return 0.0
    
    if fusion_strategy == 'simple_average':
        return float(cos_sim(candidate_embedding, fusion_embeddings[0]))
    
    elif fusion_strategy == 'overlap_centers':
        max_similarity = 0.0
        for fusion_emb in fusion_embeddings:
            similarity = float(cos_sim(candidate_embedding, fusion_emb))
            max_similarity = max(max_similarity, similarity)
        return max_similarity
    
    elif fusion_strategy == 'bridge_clusters':
        return float(cos_sim(candidate_embedding, fusion_embeddings[0]))
    
    else:
        total_similarity = 0.0
        for fusion_emb in fusion_embeddings:
            total_similarity += float(cos_sim(candidate_embedding, fusion_emb))
        return total_similarity / len(fusion_embeddings)

def generate_couple_explanation(movie_title, person1_features, person2_features, clustering_results):
    """
    Generate an explanation for why this movie is recommended for the couple.
    
    Args:
        movie_title: Title of recommended movie
        person1_features: Person 1's taste features
        person2_features: Person 2's taste features  
        clustering_results: Results from taste clustering
    
    Returns:
        String explanation
    """
    # Find genre overlaps
    genre_overlap = person1_features['genres'] & person2_features['genres']
    
    # Find actor/director overlaps
    people_overlap = (person1_features['actors'] | person1_features['directors']) & \
                    (person2_features['actors'] | person2_features['directors'])
    
    explanation_parts = []
    
    if clustering_results['taste_overlap'] > 0.3:
        explanation_parts.append(f"Strong taste alignment ({clustering_results['taste_overlap']:.0%} overlap)")
    
    if genre_overlap:
        genres_str = ', '.join(list(genre_overlap)[:3])
        explanation_parts.append(f"shared love for {genres_str}")
    
    if people_overlap:
        people_str = ', '.join(list(people_overlap)[:2])
        explanation_parts.append(f"mutual appreciation for {people_str}")
    
    strategy_explanations = {
        'overlap_centers': "Perfect fusion of both your tastes",
        'bridge_clusters': "Bridges your different preferences beautifully",
        'simple_average': "Balanced appeal to both your preferences"
    }
    
    strategy_text = strategy_explanations.get(clustering_results['fusion_strategy'], "Great match for both")
    explanation_parts.append(strategy_text)
    
    if not explanation_parts:
        return "This movie offers a good balance that should appeal to both of you."
    
    return f"{movie_title} works because of your " + " and ".join(explanation_parts) + "."

def recommend_movies_for_couple(person1_movies, person2_movies, target_recommendations=5):
    """
    Main function to generate couple movie recommendations using K-Means clustering.
    
    Args:
        person1_movies: List of person 1's favorite movie titles
        person2_movies: List of person 2's favorite movie titles
        target_recommendations: Number of recommendations to return
    
    Returns:
        List of tuples (movie_title, score, explanation)
    """
    st.write("🎭 **Starting Couples Movie Recommendation Process**")
    
    # Extract features for both people
    st.write("👤 Analyzing Person 1's taste...")
    person1_features = extract_movie_features(person1_movies)
    
    st.write("👤 Analyzing Person 2's taste...")
    person2_features = extract_movie_features(person2_movies)

    # Find compatible expansion movies
    st.write("🎯 Finding compatible expansion movies...")
    tmdb = TMDb()

    # Find 6th movie for Person 1 (compatible with Person 2)
    person1_expansion_title, person1_expansion_details = find_compatible_expansion_movie(
        person1_features, person2_features, "Person 1", tmdb.api_key
    )

    # Find 7th movie for Person 2 (compatible with Person 1) 
    person2_expansion_title, person2_expansion_details = find_compatible_expansion_movie(
        person2_features, person1_features, "Person 2", tmdb.api_key
    )

    # Add expansion movies to the features if found
    if person1_expansion_title and person1_expansion_details:
        st.write(f"✅ Added expansion movie for Person 1: {person1_expansion_title}")
        
        # Extract features from expansion movie and add to person1_features
        try:
            overview = getattr(person1_expansion_details, 'overview', '') or ''
            if overview:
                # Add to movies_info
                movie_info = {"title": person1_expansion_title, "genres": [], "year": None}
                
                # Extract genres
                genres_list = getattr(person1_expansion_details, 'genres', [])
                for g in genres_list:
                    if isinstance(g, dict):
                        name = g.get('name', '')
                        genre_id = g.get('id', 0)
                    else:
                        name = getattr(g, 'name', '')
                        genre_id = getattr(g, 'id', 0)
                    
                    if name:
                        person1_features['genres'].add(name)
                        movie_info["genres"].append(name)
                    if genre_id:
                        person1_features['genre_ids'].add(genre_id)
                
                # Generate embedding
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode(overview, convert_to_tensor=True)
                person1_features['embeddings'].append(embedding)
                person1_features['movies_info'].append(movie_info)
                
        except Exception as e:
            st.warning(f"Error processing expansion movie for Person 1: {e}")

    if person2_expansion_title and person2_expansion_details:
        st.write(f"✅ Added expansion movie for Person 2: {person2_expansion_title}")
        
        # Extract features from expansion movie and add to person2_features  
        try:
            overview = getattr(person2_expansion_details, 'overview', '') or ''
            if overview:
                # Add to movies_info
                movie_info = {"title": person2_expansion_title, "genres": [], "year": None}
                
                # Extract genres
                genres_list = getattr(person2_expansion_details, 'genres', [])
                for g in genres_list:
                    if isinstance(g, dict):
                        name = g.get('name', '')
                        genre_id = g.get('id', 0)
                    else:
                        name = getattr(g, 'name', '')
                        genre_id = getattr(g, 'id', 0)
                    
                    if name:
                        person2_features['genres'].add(name)
                        movie_info["genres"].append(name)
                    if genre_id:
                        person2_features['genre_ids'].add(genre_id)
                
                # Generate embedding
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode(overview, convert_to_tensor=True)
                person2_features['embeddings'].append(embedding)
                person2_features['movies_info'].append(movie_info)
                
        except Exception as e:
            st.warning(f"Error processing expansion movie for Person 2: {e}")
    
    # Perform taste clustering
    clustering_results = perform_taste_clustering(person1_features, person2_features)
    
    # Build candidate pool using combined preferences
    st.write("🔍 Building couple candidate pool...")
    combined_genre_ids = person1_features['genre_ids'] | person2_features['genre_ids']
    combined_cast_ids = person1_features['cast_ids'] | person2_features['cast_ids']
    combined_director_ids = person1_features['director_ids'] | person2_features['director_ids']
    combined_years = person1_features['years'] + person2_features['years']
    
    tmdb = TMDb()
    candidate_movie_ids = build_custom_candidate_pool(  # ✅ Correct function name
        combined_genre_ids, combined_cast_ids, combined_director_ids,
        combined_years, tmdb.api_key
    )
    
    # Exclude movies already in their favorites
    all_favorite_titles = set([m.lower() for m in person1_movies + person2_movies])
    
    # Fetch candidate details
    candidate_movies = {}
    fetch_cache = st.session_state.get('fetch_cache', {})
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid, fetch_cache): mid 
                  for mid in list(candidate_movie_ids)[:100]}
        
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                st.write(f"🔍 Processing result: {result is not None}")
                
                if result is None:
                    st.write("❌ Result is None")
                    continue
                if result[1] is None:
                    st.write("❌ Result[1] (payload) is None") 
                    continue
                    
                mid, payload = result
                st.write(f"✅ Got movie ID: {mid}")
                
                m, embedding = payload
                if m is None:
                    st.write("❌ Movie object is None")
                    continue
                if embedding is None:
                    st.write("❌ Embedding is None")
                    continue
                    
                movie_title = getattr(m, 'title', 'Unknown')
                vote_count = getattr(m, 'vote_count', 0)
                st.write(f"🎬 Movie: {movie_title}, Votes: {vote_count}")
                
                # Skip if in favorites
                if movie_title.lower() in all_favorite_titles:
                    st.write(f"⚠️ Skipped {movie_title} - already in favorites")
                    continue
                    
                if vote_count < 30:
                    st.write(f"⚠️ Skipped {movie_title} - low vote count ({vote_count})")
                    continue
                    
                st.write(f"✅ Added to candidates: {movie_title}")
                candidate_movies[mid] = (m, embedding)
            except Exception as e:
                continue
    
    st.session_state['fetch_cache'] = fetch_cache
    
    if not candidate_movies:
        st.warning("No suitable candidate movies found!")
        return []
    
    st.write(f"🎬 Scoring {len(candidate_movies)} candidate movies...")
    
    # Score candidates with couple-specific logic
    scored_candidates = []
    
    for movie_obj, embedding in candidate_movies.values():
        try:
            # Compute couple compatibility score
            couple_score = compute_couple_compatibility_score(
                embedding, clustering_results['fusion_embeddings'], clustering_results['fusion_strategy']
            )
            
            # Get traditional metrics
            vote_average = getattr(movie_obj, 'vote_average', 0) or 0
            vote_count = getattr(movie_obj, 'vote_count', 0) or 0
            
            # Combined scoring
            final_score = (
                couple_score * 0.6 +           # Primary couple compatibility  
                (vote_average / 10) * 0.25 +   # Quality score
                min(vote_count / 1000, 1) * 0.15  # Popularity boost
            )
            
            # Generate explanation
            movie_title = getattr(movie_obj, 'title', 'Unknown')
            explanation = generate_couple_explanation(
                movie_title, person1_features, person2_features, clustering_results
            )
            
            scored_candidates.append((movie_title, final_score, explanation))
            
        except Exception as e:
            continue
    
    # Sort and return top recommendations
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Apply diversity filtering
    final_recommendations = []
    used_words = set()
    
    for title, score, explanation in scored_candidates:
        # Simple diversity check - avoid too many movies with same key words
        title_words = set(title.lower().split())
        if len(title_words & used_words) > 1 and len(final_recommendations) >= 2:
            continue
        
        final_recommendations.append((title, score, explanation))
        used_words.update(title_words)
        
        if len(final_recommendations) >= target_recommendations:
            break
    
    st.write(f"✅ Generated {len(final_recommendations)} couple recommendations!")
    
    return final_recommendations