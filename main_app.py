"""
Couples Movie Recommendation App - Netflix-Style UI for Friend Feedback
Clean, mobile-responsive interface for collecting movie feedback
"""

import streamlit as st
import sys
import os

# =============================================================================
# COUPLE CONFIGURATION
# =============================================================================

COUPLE_NAME = "Sajal + Sneha"
PERSON1_NAME = "Sajal"
PERSON1_MOVIES = [
    "The Bourne Identity",
    "Knocked Up", 
    "Manchester by the Sea",
    "Miami Vice",
    "Gone Girl"
]

PERSON2_NAME = "Sneha"
PERSON2_MOVIES = [
    "How to Train Your Dragon",
    "3 Idiots",
    "Good Boys", 
    "The Lion King",
    "A Cinderella Story"
]

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules with error handling
try:
    from couple_scoring import recommend_movies_for_couple
    couple_scoring_available = True
except ImportError as e:
    couple_scoring_available = False

try:
    from feedback_system import (
        get_or_create_numeric_session_id,
        record_feedback_to_sheet,
        record_final_comments_to_sheet
    )
    feedback_available = True
except ImportError as e:
    feedback_available = False
    # Create dummy functions
    def get_or_create_numeric_session_id():
        return 1, "dummy-session"
    def record_feedback_to_sheet(*args, **kwargs):
        return False
    def record_final_comments_to_sheet(*args, **kwargs):
        return False

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize all required session state variables."""
    
    # Movie recommendations
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    
    # Feedback tracking
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}
    
    # UI state
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = None
    
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    
    # Caches
    if "fetch_cache" not in st.session_state:
        st.session_state.fetch_cache = {}
    
    if "movie_details_cache" not in st.session_state:
        st.session_state.movie_details_cache = {}
    
    if "movie_credits_cache" not in st.session_state:
        st.session_state.movie_credits_cache = {}
    
    if "recommendation_cache" not in st.session_state:
        st.session_state.recommendation_cache = {}
    
    if "couple_cache" not in st.session_state:
        st.session_state.couple_cache = {}
    
    # Session ID for feedback
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

# =============================================================================
# UI STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for Netflix-style carousel."""
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main-container {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Title styling */
    .couple-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #e50914;
    }
    
    /* Carousel container */
    .carousel-container {
        margin: 2rem 0;
    }
    
    .carousel-scroll {
        display: flex;
        overflow-x: auto;
        gap: 1rem;
        padding: 1rem 0;
        scroll-behavior: smooth;
    }
    
    .carousel-scroll::-webkit-scrollbar {
        height: 8px;
    }
    
    .carousel-scroll::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .carousel-scroll::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .carousel-scroll::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Movie card */
    .movie-card {
        flex: 0 0 200px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
    }
    
    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .movie-title {
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #333;
        line-height: 1.2;
        height: 2.4em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    /* Quick feedback buttons */
    .quick-feedback {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .feedback-btn {
        background: none;
        border: 2px solid #ddd;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        cursor: pointer;
        font-size: 1.2rem;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .feedback-btn:hover {
        transform: scale(1.1);
        border-color: #333;
    }
    
    .feedback-btn.selected {
        background-color: #e50914;
        color: white;
        border-color: #e50914;
    }
    
    /* Expanded details */
    .movie-details {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 4px solid #e50914;
    }
    
    .detail-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .detail-section {
        margin-bottom: 1.5rem;
    }
    
    .detail-label {
        font-weight: bold;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .detail-text {
        line-height: 1.6;
        color: #444;
    }
    
    /* Large feedback buttons */
    .large-feedback {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-top: 2rem;
    }
    
    .large-feedback-btn {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        min-width: 120px;
    }
    
    .btn-yes {
        background-color: #28a745;
        color: white;
    }
    
    .btn-yes:hover {
        background-color: #218838;
    }
    
    .btn-no {
        background-color: #dc3545;
        color: white;
    }
    
    .btn-no:hover {
        background-color: #c82333;
    }
    
    .btn-maybe {
        background-color: #ffc107;
        color: #212529;
    }
    
    .btn-maybe:hover {
        background-color: #e0a800;
    }
    
    /* Submit section */
    .submit-section {
        text-align: center;
        margin: 3rem 0;
        padding: 2rem;
        background: #f0f0f0;
        border-radius: 12px;
    }
    
    .feedback-count {
        font-size: 1.1rem;
        margin-bottom: 1rem;
        color: #666;
    }
    
    /* Thank you message */
    .thank-you {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin: 2rem 0;
    }
    
    .thank-you h2 {
        margin-bottom: 1rem;
        font-size: 2rem;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .couple-title {
            font-size: 2rem;
        }
        
        .movie-card {
            flex: 0 0 150px;
        }
        
        .movie-poster {
            height: 225px;
        }
        
        .movie-title {
            font-size: 0.8rem;
        }
        
        .feedback-btn {
            width: 35px;
            height: 35px;
            font-size: 1rem;
        }
        
        .movie-details {
            padding: 1rem;
        }
        
        .large-feedback {
            flex-direction: column;
            align-items: center;
        }
        
        .large-feedback-btn {
            width: 200px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_movie_poster_url(movie_title):
    """Get movie poster URL from TMDB."""
    try:
        from tmdbv3api import Movie
        movie_api = Movie()
        search_results = movie_api.search(movie_title)
        if search_results and search_results[0].poster_path:
            return f"https://image.tmdb.org/t/p/w500{search_results[0].poster_path}"
        return None
    except:
        return None

def get_movie_details(movie_title):
    """Get movie details from TMDB."""
    try:
        from tmdbv3api import Movie
        movie_api = Movie()
        search_results = movie_api.search(movie_title)
        if search_results:
            movie_id = search_results[0].id
            details = movie_api.details(movie_id)
            return {
                'overview': getattr(details, 'overview', 'No description available.'),
                'release_date': getattr(details, 'release_date', ''),
                'runtime': getattr(details, 'runtime', ''),
                'vote_average': getattr(details, 'vote_average', 0),
                'genres': [g.name for g in getattr(details, 'genres', [])]
            }
        return None
    except:
        return None

def generate_recommendations():
    """Generate movie recommendations for the couple."""
    if not couple_scoring_available:
        # Return dummy data for UI testing
        return [
            ("Inception", 0.85, "Perfect blend of action and complex storytelling that bridges both your tastes."),
            ("The Grand Budapest Hotel", 0.82, "Whimsical yet sophisticated, matching your appreciation for unique narratives."),
            ("Knives Out", 0.80, "Smart mystery with humor that appeals to both your preferences."),
            ("Spider-Man: Into the Spider-Verse", 0.78, "Innovative animation with heart, bridging adventure and emotional depth."),
            ("Parasite", 0.76, "Critically acclaimed thriller with social commentary you'll both appreciate."),
            ("The Princess Bride", 0.75, "Classic adventure-comedy that's both nostalgic and entertaining."),
            ("Mad Max: Fury Road", 0.73, "High-octane action with strong character development."),
            ("Moonrise Kingdom", 0.72, "Charming coming-of-age story with visual flair."),
            ("Baby Driver", 0.70, "Stylish action with great music and humor."),
            ("The Shape of Water", 0.68, "Unique fantasy romance with exceptional cinematography.")
        ]
    
    # Check cache first
    cache_key = f"couple_recs_{hash(tuple(PERSON1_MOVIES + PERSON2_MOVIES))}"
    if cache_key in st.session_state.couple_cache:
        return st.session_state.couple_cache[cache_key]
    
    try:
        recommendations = recommend_movies_for_couple(PERSON1_MOVIES, PERSON2_MOVIES)
        st.session_state.couple_cache[cache_key] = recommendations
        return recommendations
    except Exception as e:
        # Return dummy data on error
        return []

def record_feedback(movie_index, movie_title, feedback_type):
    """Record user feedback for a movie."""
    try:
        if feedback_available:
            numeric_id, session_id = get_or_create_numeric_session_id()
            combined_favorites = f"{PERSON1_NAME}: {', '.join(PERSON1_MOVIES)} | {PERSON2_NAME}: {', '.join(PERSON2_MOVIES)}"
            
            # Get recommendation details
            if movie_index < len(st.session_state.recommendations):
                _, score, explanation = st.session_state.recommendations[movie_index]
            else:
                score, explanation = 0.0, "No explanation available"
            
            record_feedback_to_sheet(
                numeric_session_id=numeric_id,
                uuid_session_id=session_id,
                user_top_5_movies=combined_favorites,
                user_taste_profile="couple_feedback",
                user_favorite_genres="mixed_preferences",
                recommendation_rank=movie_index + 1,
                movie_id=f"COUPLE_REC_{movie_index}",
                movie_title=movie_title,
                movie_genres="various",
                movie_year="N/A",
                recommendation_score=score,
                recommendation_reason=explanation,
                would_watch=feedback_type,
                liked_if_seen="N/A"
            )
        
        # Store feedback in session state
        st.session_state.feedback_given[movie_index] = feedback_type
        
    except Exception as e:
        # Still store locally even if remote fails
        st.session_state.feedback_given[movie_index] = feedback_type

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_movie_carousel():
    """Render the Netflix-style movie carousel using Streamlit columns."""
    if not st.session_state.recommendations:
        st.warning("Loading recommendations...")
        return
    
    st.markdown("### 🎬 Movie Recommendations")
    
    # Create two rows of 5 movies each
    for row in range(2):
        cols = st.columns(5)
        start_idx = row * 5
        end_idx = min(start_idx + 5, len(st.session_state.recommendations))
        
        for col_idx, movie_idx in enumerate(range(start_idx, end_idx)):
            if movie_idx >= len(st.session_state.recommendations):
                break
                
            movie_title, score, explanation = st.session_state.recommendations[movie_idx]
            
            with cols[col_idx]:
                # Movie poster
                poster_url = get_movie_poster_url(movie_title)
                
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="background-color: #ddd; height: 300px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #666;">🎬<br>No Poster</div>',
                        unsafe_allow_html=True
                    )
                
                # Movie title
                st.markdown(f"**{movie_title}**")
                
                # Quick feedback buttons
                feedback = st.session_state.feedback_given.get(movie_idx, None)
                
                button_cols = st.columns(3)
                
                with button_cols[0]:
                    button_type = "primary" if feedback == "Yes" else "secondary"
                    if st.button("👍", key=f"yes_{movie_idx}", type=button_type):
                        record_feedback(movie_idx, movie_title, "Yes")
                        st.rerun()
                
                with button_cols[1]:
                    button_type = "primary" if feedback == "Maybe" else "secondary"
                    if st.button("🤷", key=f"maybe_{movie_idx}", type=button_type):
                        record_feedback(movie_idx, movie_title, "Maybe")
                        st.rerun()
                
                with button_cols[2]:
                    button_type = "primary" if feedback == "No" else "secondary"
                    if st.button("👎", key=f"no_{movie_idx}", type=button_type):
                        record_feedback(movie_idx, movie_title, "No")
                        st.rerun()
                
                # Show detailed view button
                if st.button("ℹ️ Details", key=f"details_{movie_idx}"):
                    st.session_state.selected_movie = movie_idx
                    st.rerun()
        
        # Add some spacing between rows
        if row == 0:
            st.markdown("<br>", unsafe_allow_html=True)

def render_movie_details(movie_index):
    """Render expanded movie details."""
    if movie_index >= len(st.session_state.recommendations):
        return
    
    movie_title, score, explanation = st.session_state.recommendations[movie_index]
    details = get_movie_details(movie_title)
    feedback = st.session_state.feedback_given.get(movie_index, None)
    
    st.markdown(f'''
    <div class="movie-details">
        <div class="detail-title">{movie_title}</div>
        
        <div class="detail-section">
            <div class="detail-label">Why we recommend this:</div>
            <div class="detail-text">{explanation}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    if details:
        if details['overview']:
            st.markdown(f'''
            <div class="detail-section">
                <div class="detail-label">Plot:</div>
                <div class="detail-text">{details['overview']}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        if details['genres']:
            genres_text = ', '.join(details['genres'])
            st.markdown(f'''
            <div class="detail-section">
                <div class="detail-label">Genres:</div>
                <div class="detail-text">{genres_text}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        if details['vote_average']:
            st.markdown(f'''
            <div class="detail-section">
                <div class="detail-label">TMDB Rating:</div>
                <div class="detail-text">{details['vote_average']}/10</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Large feedback buttons
    st.markdown('<div class="large-feedback">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Yes!", key=f"large_yes_{movie_index}", 
                    type="primary" if feedback == "Yes" else "secondary"):
            record_feedback(movie_index, movie_title, "Yes")
            st.rerun()
    
    with col2:
        if st.button("🤷 Maybe", key=f"large_maybe_{movie_index}",
                    type="primary" if feedback == "Maybe" else "secondary"):
            record_feedback(movie_index, movie_title, "Maybe")
            st.rerun()
    
    with col3:
        if st.button("👎 No", key=f"large_no_{movie_index}",
                    type="primary" if feedback == "No" else "secondary"):
            record_feedback(movie_index, movie_title, "No")
            st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_submit_section():
    """Render the feedback submission section."""
    feedback_count = len(st.session_state.feedback_given)
    total_movies = len(st.session_state.recommendations)
    
    if feedback_count > 0:
        st.markdown(f'''
        <div class="submit-section">
            <div class="feedback-count">
                You've rated {feedback_count} out of {total_movies} movies
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        if feedback_count == total_movies:
            if st.button("✅ Submit All Feedback", type="primary", key="submit_all"):
                st.session_state.feedback_submitted = True
                st.rerun()

def render_thank_you():
    """Render thank you message after submission."""
    st.markdown('''
    <div class="thank-you">
        <h2>🎉 Thank You!</h2>
        <p>Your feedback has been submitted successfully.</p>
        <p>Sajal + Sneha will love seeing your movie recommendations!</p>
    </div>
    ''', unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    st.set_page_config(
        page_title=f"{COUPLE_NAME} Movie Night",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize
    initialize_session_state()
    inject_custom_css()
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title
    st.markdown(f'<h1 class="couple-title">🎬 {COUPLE_NAME}</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Help us pick our next movie night! Rate these recommendations:</p>', unsafe_allow_html=True)
    
    # Load recommendations if not already loaded
    if not st.session_state.recommendations:
        with st.spinner("🎯 Loading personalized recommendations..."):
            st.session_state.recommendations = generate_recommendations()
    
    # Show thank you page if feedback submitted
    if st.session_state.feedback_submitted:
        render_thank_you()
    else:
        # Render movie carousel
        render_movie_carousel()
        
        # Handle movie selection via URL params or session state
        query_params = st.query_params
        selected_index = query_params.get('movie', [None])[0]
        
        if selected_index is not None:
            try:
                movie_index = int(selected_index)
                if 0 <= movie_index < len(st.session_state.recommendations):
                    render_movie_details(movie_index)
            except ValueError:
                pass
        
        # JavaScript to handle carousel interactions
        st.markdown('''
        <script>
        window.addEventListener('message', function(event) {
            if (event.data.type === 'selectMovie') {
                const params = new URLSearchParams(window.location.search);
                params.set('movie', event.data.index);
                window.history.replaceState({}, '', `${window.location.pathname}?${params}`);
                window.location.reload();
            } else if (event.data.type === 'quickFeedback') {
                // This would need to be handled via Streamlit's session state
                // For now, we'll rely on the large buttons in the detail view
            }
        });
        </script>
        ''', unsafe_allow_html=True)
        
        # Submit section
        render_submit_section()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()