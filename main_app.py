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
    
    # NEW: Movie modal cache for cast/director info
    if "movie_modal_cache" not in st.session_state:
        st.session_state.movie_modal_cache = {}
    
    # Session ID for feedback
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

def get_movie_cast_director(movie_title):
    """Get cast and director info with caching for better performance."""
    # Check cache first
    if movie_title in st.session_state.movie_modal_cache:
        return st.session_state.movie_modal_cache[movie_title]
    
    try:
        from tmdbv3api import Movie
        movie_api = Movie()
        search_result = movie_api.search(movie_title)
        
        cast_info = {"cast": [], "director": ""}
        
        if search_result:
            credits = movie_api.credits(search_result[0].id)
            
            # Cast (top 3)
            if hasattr(credits, 'cast') and credits.cast:
                cast_names = []
                for actor in credits.cast[:3]:
                    if hasattr(actor, 'name'):
                        cast_names.append(actor.name)
                cast_info["cast"] = cast_names
            
            # Director
            if hasattr(credits, 'crew') and credits.crew:
                for person in credits.crew:
                    if hasattr(person, 'job') and person.job == 'Director' and hasattr(person, 'name'):
                        cast_info["director"] = person.name
                        break
        
        # Cache the result
        st.session_state.movie_modal_cache[movie_title] = cast_info
        return cast_info
    
    except Exception as e:
        # Return empty info on error
        empty_info = {"cast": [], "director": ""}
        st.session_state.movie_modal_cache[movie_title] = empty_info
        return empty_info

# =============================================================================
# UI STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for Netflix-style carousel and modal."""
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
    
    /* Modal Overlay */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        padding: 20px;
        box-sizing: border-box;
    }
    
    /* Modal Content */
    .modal-content {
        background: #141414;
        border-radius: 12px;
        width: 90vw;
        max-width: 1000px;
        max-height: 90vh;
        overflow-y: auto;
        position: relative;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7);
    }
    
    /* Modal Header */
    .modal-header {
        position: relative;
        padding: 2rem;
        border-bottom: 1px solid #333;
    }
    
    /* Close Button */
    .modal-close {
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(42, 42, 42, 0.8);
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        color: white;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-close:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.1);
    }
    
    /* Navigation Arrows */
    .nav-arrow {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(42, 42, 42, 0.8);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        color: white;
        font-size: 24px;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10001;
    }
    
    .nav-arrow:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-50%) scale(1.1);
    }
    
    .nav-arrow.prev {
        left: -30px;
    }
    
    .nav-arrow.next {
        right: -30px;
    }
    
    /* Modal Body */
    .modal-body {
        display: flex;
        padding: 2rem;
        gap: 2rem;
    }
    
    .modal-poster {
        flex: 0 0 300px;
    }
    
    .modal-poster img {
        width: 100%;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    .modal-details {
        flex: 1;
        color: #e5e5e5;
    }
    
    .modal-title {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .modal-section {
        margin-bottom: 1.5rem;
    }
    
    .modal-label {
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        display: block;
    }
    
    .modal-text {
        line-height: 1.6;
        color: #e5e5e5;
        margin-bottom: 1rem;
    }
    
    .modal-genres {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .genre-tag {
        background: #e50914;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Modal Feedback Buttons */
    .modal-feedback {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
        justify-content: center;
    }
    
    .modal-feedback-btn {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        border: 2px solid #333;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        min-width: 120px;
        background: transparent;
        color: white;
    }
    
    .modal-feedback-btn:hover {
        transform: scale(1.05);
        border-color: #e50914;
    }
    
    .modal-feedback-btn.selected {
        background: #e50914;
        border-color: #e50914;
        color: white;
    }
    
    .btn-yes {
        border-color: #28a745;
    }
    
    .btn-yes:hover,
    .btn-yes.selected {
        background: #28a745;
        border-color: #28a745;
    }
    
    .btn-maybe {
        border-color: #ffc107;
    }
    
    .btn-maybe:hover,
    .btn-maybe.selected {
        background: #ffc107;
        border-color: #ffc107;
        color: #000;
    }
    
    .btn-no {
        border-color: #dc3545;
    }
    
    .btn-no:hover,
    .btn-no.selected {
        background: #dc3545;
        border-color: #dc3545;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .modal-content {
            width: 95vw;
            max-height: 95vh;
        }
        
        .modal-body {
            flex-direction: column;
            padding: 1rem;
            gap: 1rem;
        }
        
        .modal-poster {
            flex: none;
            max-width: 200px;
            margin: 0 auto;
        }
        
        .modal-title {
            font-size: 1.5rem;
            text-align: center;
        }
        
        .nav-arrow {
            width: 40px;
            height: 40px;
            font-size: 20px;
        }
        
        .nav-arrow.prev {
            left: -20px;
        }
        
        .nav-arrow.next {
            right: -20px;
        }
        
        .modal-feedback {
            flex-direction: column;
            align-items: center;
        }
        
        .modal-feedback-btn {
            width: 200px;
        }
    }
    
    @media (max-width: 480px) {
        .modal-overlay {
            padding: 10px;
        }
        
        .modal-header,
        .modal-body {
            padding: 1rem;
        }
        
        .nav-arrow.prev {
            left: -15px;
        }
        
        .nav-arrow.next {
            right: -15px;
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
    """Render the Netflix-style movie carousel with feedback status indicators."""
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
            feedback = st.session_state.feedback_given.get(movie_idx, None)
            
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
                
                # Movie title with feedback status indicator
                if feedback:
                    feedback_icon = "✅" if feedback == "Yes" else "❓" if feedback == "Maybe" else "❌"
                    st.markdown(f"**{movie_title}** {feedback_icon}")
                else:
                    st.markdown(f"**{movie_title}**")
                
                # Quick feedback buttons
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

def render_movie_modal():
    """Render movie details with keyboard navigation and cached data."""
    if st.session_state.get('selected_movie') is None:
        return
    
    movie_idx = st.session_state.selected_movie
    if movie_idx >= len(st.session_state.recommendations):
        return
    
    movie_title, score, explanation = st.session_state.recommendations[movie_idx]
    
    # Add keyboard navigation JavaScript
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            window.parent.postMessage({type: 'closeModal'}, '*');
        } else if (event.key === 'ArrowLeft') {
            window.parent.postMessage({type: 'prevMovie'}, '*');
        } else if (event.key === 'ArrowRight') {
            window.parent.postMessage({type: 'nextMovie'}, '*');
        }
    });
    
    window.addEventListener('message', function(event) {
        if (event.data.type === 'closeModal') {
            // Trigger close
        } else if (event.data.type === 'prevMovie') {
            // Trigger previous
        } else if (event.data.type === 'nextMovie') {
            // Trigger next
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Compact styling
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for keyboard navigation via URL params (simple implementation)
    query_params = st.query_params
    if 'nav' in query_params:
        nav_action = query_params['nav']
        current_idx = st.session_state.selected_movie
        total_movies = len(st.session_state.recommendations)
        
        if nav_action == 'prev':
            new_idx = current_idx - 1 if current_idx > 0 else total_movies - 1
            st.session_state.selected_movie = new_idx
            del st.query_params['nav']
            st.rerun()
        elif nav_action == 'next':
            new_idx = current_idx + 1 if current_idx < total_movies - 1 else 0
            st.session_state.selected_movie = new_idx
            del st.query_params['nav']
            st.rerun()
        elif nav_action == 'close':
            st.session_state.selected_movie = None
            del st.query_params['nav']
            st.rerun()
    
    # Modal Header with navigation
    col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 2, 2])
    
    with col1:
        if st.button("\u25C0 Previous", key="modal_prev", use_container_width=True, help="Previous movie (← key)"):
            current_idx = st.session_state.selected_movie
            total_movies = len(st.session_state.recommendations)
            new_idx = current_idx - 1 if current_idx > 0 else total_movies - 1
            st.session_state.selected_movie = new_idx
            st.rerun()
    
    with col3:
        st.markdown(f"<h3 style='text-align: center; color: #e50914; margin: 0;'>Movie {movie_idx + 1} of {len(st.session_state.recommendations)}</h3>", 
                   unsafe_allow_html=True)
    
    with col5:
        if st.button("Next \u25B6", key="modal_next", use_container_width=True, help="Next movie (→ key)"):
            current_idx = st.session_state.selected_movie
            total_movies = len(st.session_state.recommendations)
            new_idx = current_idx + 1 if current_idx < total_movies - 1 else 0
            st.session_state.selected_movie = new_idx
            st.rerun()
    
    # Close button centered below
    col_spacer1, col_close, col_spacer2 = st.columns([4, 2, 4])
    with col_close:
        if st.button("✕ Close", key="modal_close", use_container_width=True, help="Close modal (Esc key)"):
            st.session_state.selected_movie = None
            st.rerun()
    
    # Compact divider
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Modal Body
    col_poster, col_details = st.columns([1, 2])
    
    with col_poster:
        # Movie poster
        poster_url = get_movie_poster_url(movie_title)
        if poster_url:
            st.image(poster_url, width=250)
        else:
            st.info("🎬 No Poster Available")
    
    with col_details:
        # Movie title
        st.markdown(f"<h2 style='margin-bottom: 0.5rem;'>{movie_title}</h2>", 
                   unsafe_allow_html=True)
        
        # Why we recommend this
        st.markdown("**🎯 Why we recommend this:**")
        st.write(explanation)
        
        # Get and display movie details
        details = get_movie_details(movie_title)
        
        if details:
            # Plot summary
            if details.get('overview'):
                st.markdown("**📖 Plot:**")
                st.write(details['overview'])
            
            # Genres
            if details.get('genres'):
                genres_text = " • ".join(details['genres'])
                st.markdown(f"**🎭 Genres:** {genres_text}")
            
            # Get cached cast and director info
            cast_director_info = get_movie_cast_director(movie_title)
            
            # Cast
            if cast_director_info["cast"]:
                st.markdown(f"**🎭 Starring:** {', '.join(cast_director_info['cast'])}")
            
            # Director
            if cast_director_info["director"]:
                st.markdown(f"**🎬 Director:** {cast_director_info['director']}")
        
        # Feedback Section
        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown("**Would you both watch this movie together?**")
        
        feedback = st.session_state.feedback_given.get(movie_idx, None)
        
        # Feedback buttons
        col_yes, col_maybe, col_no = st.columns(3)
        
        with col_yes:
            button_type = "primary" if feedback == "Yes" else "secondary"
            if st.button("👍 Yes!", key=f"modal_yes_{movie_idx}", type=button_type, use_container_width=True):
                record_feedback(movie_idx, movie_title, "Yes")
                st.success("✅ Marked as 'Yes'!")
                st.balloons()
                st.rerun()
        
        with col_maybe:
            button_type = "primary" if feedback == "Maybe" else "secondary"
            if st.button("🤷 Maybe", key=f"modal_maybe_{movie_idx}", type=button_type, use_container_width=True):
                record_feedback(movie_idx, movie_title, "Maybe")
                st.success("✅ Marked as 'Maybe'!")
                st.rerun()
        
        with col_no:
            button_type = "primary" if feedback == "No" else "secondary"
            if st.button("👎 No", key=f"modal_no_{movie_idx}", type=button_type, use_container_width=True):
                record_feedback(movie_idx, movie_title, "No")
                st.success("✅ Marked as 'No'!")
                st.rerun()

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
    
    # Title
    st.markdown(f'<h1 style="text-align: center; color: #e50914; font-size: 2.5rem; margin-bottom: 2rem;">🎬 {COUPLE_NAME}</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Help us pick our next movie night! Rate these recommendations:</p>', 
                unsafe_allow_html=True)
    
    # Load recommendations if not already loaded
    if not st.session_state.recommendations:
        with st.spinner("🎯 Loading personalized recommendations..."):
            st.session_state.recommendations = generate_recommendations()
    
    # Show thank you page if feedback submitted
    if st.session_state.feedback_submitted:
        render_thank_you()
    else:
        # Check if modal should be shown (selected_movie is not None)
        if st.session_state.get('selected_movie') is not None:
            render_movie_modal()
        else:
            # Show carousel only when modal is not open
            render_movie_carousel()
            render_submit_section()

if __name__ == "__main__":
    main()