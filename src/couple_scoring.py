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

# Import other modules from src
from movie_scoring import (
    build_enhanced_candidate_pool, 
    fetch_similar_movie_details
)
from utils import get_embedding_model
from narrative_analysis import infer_narrative_style
from movie_search import fuzzy_search_movies
from tmdbv3api import Movie, TMDb

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
    
    movie_api = Movie()
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
    
    # Perform taste clustering
    clustering_results = perform_taste_clustering(person1_features, person2_features)
    
    # Build candidate pool using combined preferences
    st.write("🔍 Building couple candidate pool...")
    combined_genre_ids = person1_features['genre_ids'] | person2_features['genre_ids']
    combined_cast_ids = person1_features['cast_ids'] | person2_features['cast_ids']
    combined_director_ids = person1_features['director_ids'] | person2_features['director_ids']
    combined_years = person1_features['years'] + person2_features['years']
    
    tmdb = TMDb()
    candidate_movie_ids = build_enhanced_candidate_pool(
        combined_genre_ids, combined_cast_ids, combined_director_ids,
        combined_years, tmdb.api_key, target_pool_size=200
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
                if result is None or result[1] is None:
                    continue
                mid, payload = result
                m, embedding = payload
                if m is None or embedding is None:
                    continue
                
                # Skip if in favorites
                movie_title = getattr(m, 'title', '').lower()
                if movie_title in all_favorite_titles:
                    continue
                
                vote_count = getattr(m, 'vote_count', 0)
                if vote_count < 30:  # Lower threshold for couples
                    continue
                
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