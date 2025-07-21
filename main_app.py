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
                    st.query_params['modal'] = str(movie_idx)
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
    """Render the Netflix-style modal for movie details."""
    if st.session_state.get('selected_movie') is None:
        return
    
    movie_idx = st.session_state.selected_movie
    if movie_idx >= len(st.session_state.recommendations):
        return
    
    movie_title, score, explanation = st.session_state.recommendations[movie_idx]
    
    # Get movie details
    details = get_movie_details(movie_title)
    poster_url = get_movie_poster_url(movie_title)
    
    # Get feedback status
    feedback = st.session_state.feedback_given.get(movie_idx, None)
    
    # Create modal HTML
    modal_html = f'''
    <div class="modal-overlay" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <!-- Close Button -->
            <button class="modal-close" onclick="closeModal()">×</button>
            
            <!-- Navigation Arrows -->
            <button class="nav-arrow prev" onclick="navigateMovie({movie_idx}, 'prev')">‹</button>
            <button class="nav-arrow next" onclick="navigateMovie({movie_idx}, 'next')">›</button>
            
            <!-- Modal Body -->
            <div class="modal-body">
                <div class="modal-poster">
                    {"<img src='" + poster_url + "' alt='" + movie_title + "'>" if poster_url else "<div style='background:#333; height:450px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:#999;'>🎬<br>No Poster</div>"}
                </div>
                
                <div class="modal-details">
                    <h1 class="modal-title">{movie_title}</h1>
                    
                    <div class="modal-section">
                        <span class="modal-label">🎯 Why we recommend this:</span>
                        <div class="modal-text">{explanation}</div>
                    </div>
    '''
    
    # Add plot summary if available
    if details and details.get('overview'):
        modal_html += f'''
                    <div class="modal-section">
                        <span class="modal-label">📖 Plot:</span>
                        <div class="modal-text">{details['overview']}</div>
                    </div>
        '''
    
    # Add genres if available
    if details and details.get('genres'):
        genres_html = ''.join([f'<span class="genre-tag">{genre}</span>' for genre in details['genres']])
        modal_html += f'''
                    <div class="modal-section">
                        <span class="modal-label">🎭 Genres:</span>
                        <div class="modal-genres">{genres_html}</div>
                    </div>
        '''
    
    # Add cast and director if available (using TMDB API)
    try:
        from tmdbv3api import Movie
        movie_api = Movie()
        search_result = movie_api.search(movie_title)
        if search_result:
            credits = movie_api.credits(search_result[0].id)
            
            # Get cast (top 3)
            cast_list = credits.cast[:3] if hasattr(credits, 'cast') and credits.cast else []
            if cast_list:
                cast_names = [getattr(actor, 'name', '') for actor in cast_list if hasattr(actor, 'name')]
                if cast_names:
                    modal_html += f'''
                    <div class="modal-section">
                        <span class="modal-label">🎭 Starring:</span>
                        <div class="modal-text">{', '.join(cast_names)}</div>
                    </div>
                    '''
            
            # Get director
            crew_list = credits.crew if hasattr(credits, 'crew') and credits.crew else []
            directors = [getattr(person, 'name', '') for person in crew_list if getattr(person, 'job', '') == 'Director']
            if directors:
                modal_html += f'''
                    <div class="modal-section">
                        <span class="modal-label">🎬 Director:</span>
                        <div class="modal-text">{directors[0]}</div>
                    </div>
                '''
    except:
        pass
    
    # Add feedback section
    modal_html += f'''
                    <div class="modal-section">
                        <span class="modal-label">Would you both watch this movie together?</span>
                        <div class="modal-feedback">
                            <button class="modal-feedback-btn btn-yes {'selected' if feedback == 'Yes' else ''}" 
                                    onclick="giveFeedback({movie_idx}, 'Yes')">👍 Yes!</button>
                            <button class="modal-feedback-btn btn-maybe {'selected' if feedback == 'Maybe' else ''}" 
                                    onclick="giveFeedback({movie_idx}, 'Maybe')">🤷 Maybe</button>
                            <button class="modal-feedback-btn btn-no {'selected' if feedback == 'No' else ''}" 
                                    onclick="giveFeedback({movie_idx}, 'No')">👎 No</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function closeModal(event) {{
            // Close if clicking outside modal content or on close button
            if (!event || event.target.classList.contains('modal-overlay') || event.target.classList.contains('modal-close')) {{
                window.parent.postMessage({{type: 'closeModal'}}, '*');
            }}
        }}
        
        function navigateMovie(currentIdx, direction) {{
            const totalMovies = {len(st.session_state.recommendations)};
            let newIdx;
            
            if (direction === 'prev') {{
                newIdx = currentIdx > 0 ? currentIdx - 1 : totalMovies - 1;
            }} else {{
                newIdx = currentIdx < totalMovies - 1 ? currentIdx + 1 : 0;
            }}
            
            window.parent.postMessage({{type: 'navigateMovie', index: newIdx}}, '*');
        }}
        
        function giveFeedback(movieIdx, feedback) {{
            window.parent.postMessage({{type: 'giveFeedback', movieIdx: movieIdx, feedback: feedback}}, '*');
        }}
        
        // Handle keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }} else if (event.key === 'ArrowLeft') {{
                navigateMovie({st.session_state.selected_movie}, 'prev');
            }} else if (event.key === 'ArrowRight') {{
                navigateMovie({st.session_state.selected_movie}, 'next');
            }}
        }});
    </script>
    '''
    
    # Render the modal
    st.markdown(modal_html, unsafe_allow_html=True)

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
        
        # Handle JavaScript messages from modal
        st.markdown('''
        <script>
        window.addEventListener('message', function(event) {
            if (event.data.type === 'closeModal') {
                // This would trigger Streamlit to close modal
                const params = new URLSearchParams(window.location.search);
                params.delete('modal');
                window.history.replaceState({}, '', `${window.location.pathname}?${params}`);
                window.location.reload();
            } else if (event.data.type === 'navigateMovie') {
                const params = new URLSearchParams(window.location.search);
                params.set('modal', event.data.index);
                window.history.replaceState({}, '', `${window.location.pathname}?${params}`);
                window.location.reload();
            } else if (event.data.type === 'giveFeedback') {
                // Handle feedback submission
                const params = new URLSearchParams(window.location.search);
                params.set('feedback', `${event.data.movieIdx}-${event.data.feedback}`);
                window.history.replaceState({}, '', `${window.location.pathname}?${params}`);
                window.location.reload();
            }
        });
        </script>
        ''', unsafe_allow_html=True)
        
        # Handle URL parameters for modal and feedback
        query_params = st.query_params
        
        # Handle modal display
        if 'modal' in query_params:
            try:
                modal_idx = int(query_params['modal'])
                st.session_state.selected_movie = modal_idx
            except:
                pass
        
        # Handle feedback submission
        if 'feedback' in query_params:
            try:
                feedback_data = query_params['feedback'].split('-')
                if len(feedback_data) == 2:
                    movie_idx, feedback_type = int(feedback_data[0]), feedback_data[1]
                    movie_title = st.session_state.recommendations[movie_idx][0]
                    record_feedback(movie_idx, movie_title, feedback_type)
                    # Clear the feedback parameter
                    del st.query_params['feedback']
            except:
                pass
        
        # Render modal if movie is selected
        render_movie_modal()
        
        # Submit section
        render_submit_section()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()