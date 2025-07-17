"""
Couples Movie Recommendation App - Main UI and Orchestration
Streamlit interface that orchestrates the recommendation system
"""

import streamlit as st
import sys
import os

# Add this to your code temporarily to debug
from tmdbv3api import TMDb
tmdb = TMDb()
print(f"TMDB API Key: {tmdb.api_key}")

# =============================================================================
# DEBUG SECTION: Test movie_scoring imports
# =============================================================================
try:
    from movie_scoring import build_custom_candidate_pool
    st.write("✅ build_custom_candidate_pool import successful")
except Exception as e:
    st.write(f"❌ build_custom_candidate_pool import failed: {e}")

try:
    from movie_scoring import fetch_similar_movie_details
    st.write("✅ fetch_similar_movie_details import successful")
except Exception as e:
    st.write(f"❌ fetch_similar_movie_details import failed: {e}")
# =============================================================================
# END DEBUG SECTION
# =============================================================================

# =============================================================================
# COUPLE CONFIGURATION - EDIT THIS SECTION FOR EACH COUPLE
# =============================================================================

PERSON1_NAME = "Your"
PERSON1_MOVIES = [
    "The Bourne Identity",
    "Knocked Up", 
    "Manchester by the Sea",
    "Miami Vice",
    "Gone Girl"
]

PERSON2_NAME = "Sneha's"
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
    st.error(f"❌ Could not import couple_scoring: {e}")
    st.error("Make sure src/couple_scoring.py exists")
    couple_scoring_available = False

try:
    from feedback_system import (
        get_or_create_numeric_session_id,
        record_feedback_to_sheet,
        record_final_comments_to_sheet
    )
    feedback_available = True
except ImportError as e:
    st.warning(f"⚠️ Could not import feedback_system: {e}")
    st.warning("Feedback features will be disabled")
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

def initialize_couples_session_state():
    """Initialize all required session state variables."""
    
    # Couples-specific state
    if "couple_recommendations" not in st.session_state:
        st.session_state.couple_recommendations = []
    
    if "couple_feedback_given" not in st.session_state:
        st.session_state.couple_feedback_given = {}
    
    if "show_couple_feedback" not in st.session_state:
        st.session_state.show_couple_feedback = False
    
    if "couple_cache" not in st.session_state:
        st.session_state.couple_cache = {}
    
    # Shared cache from existing system
    if "fetch_cache" not in st.session_state:
        st.session_state.fetch_cache = {}
    
    if "movie_details_cache" not in st.session_state:
        st.session_state.movie_details_cache = {}
    
    if "movie_credits_cache" not in st.session_state:
        st.session_state.movie_credits_cache = {}
    
    if "recommendation_cache" not in st.session_state:
        st.session_state.recommendation_cache = {}
    
    # Session ID for feedback
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_favorite_movies():
    """Display both partners' favorite movies at the top."""
    st.title("🎬 Screen or Skip")
    st.markdown("---")
    
    # Create two columns for the couples' favorites
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🎯 {PERSON1_NAME} Core Favorites")
        for i, movie in enumerate(PERSON1_MOVIES, 1):
            st.write(f"{i}. **{movie}**")
        st.write("*+ AI will find a 6th compatible movie*")
    
    with col2:
        st.subheader(f"💕 {PERSON2_NAME} Core Favorites")
        for i, movie in enumerate(PERSON2_MOVIES, 1):
            st.write(f"{i}. **{movie}**")
        st.write("*+ AI will find a 7th compatible movie*")
    
    return PERSON1_MOVIES, PERSON2_MOVIES

def display_couple_recommendations(recommendations):
    """Display the couple recommendations with feedback options."""
    st.markdown("---")
    st.subheader("🎭 Movies You Both May Enjoy Together")
    
    # Show algorithm info
    with st.expander("🧠 How This Works", expanded=False):
        st.write("""
        **Taste Cluster Fusion via K-Means:**
        1. 🎬 Analyzes both your movie preferences using plot embeddings
        2. 🧠 Finds taste clusters using K-Means clustering  
        3. 🎯 Identifies overlapping preferences or bridges between different tastes
        4. ✨ Recommends movies that appeal to your fused taste profile
        
        **Three Fusion Strategies:**
        - **Overlap Centers**: When you have similar tastes
        - **Bridge Clusters**: When you have different but compatible tastes  
        - **Simple Average**: Balanced approach for all taste combinations
        """)
    
    if not recommendations:
        st.warning("No recommendations generated yet. Click 'Generate Recommendations' above!")
        return
    
    for i, (title, score, explanation) in enumerate(recommendations, 1):
        with st.expander(f"#{i} {title} (Score: {score:.3f})", expanded=i <= 3):
            st.write(f"**🎯 Why this movie:** {explanation}")
            
            # Feedback section
            feedback_key = f"couple_rec_{i}"
            
            if feedback_available and feedback_key not in st.session_state.couple_feedback_given:
                st.write("**Would you both watch this movie together?**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("👍 Yes!", key=f"yes_{feedback_key}"):
                        record_couple_feedback(i, title, score, explanation, "Yes", "N/A")
                        st.session_state.couple_feedback_given[feedback_key] = "Yes"
                        st.success("Thanks for the feedback!")
                        st.rerun()
                
                with col2:
                    if st.button("👎 No", key=f"no_{feedback_key}"):
                        record_couple_feedback(i, title, score, explanation, "No", "N/A")
                        st.session_state.couple_feedback_given[feedback_key] = "No"
                        st.success("Thanks for the feedback!")
                        st.rerun()
                
                with col3:
                    if st.button("🤔 Maybe", key=f"maybe_{feedback_key}"):
                        record_couple_feedback(i, title, score, explanation, "Maybe", "N/A")
                        st.session_state.couple_feedback_given[feedback_key] = "Maybe"
                        st.success("Thanks for the feedback!")
                        st.rerun()
            elif feedback_available:
                feedback = st.session_state.couple_feedback_given[feedback_key]
                st.success(f"✅ You both said: **{feedback}**")

# =============================================================================
# FEEDBACK FUNCTIONS
# =============================================================================

def record_couple_feedback(rank, title, score, explanation, would_watch, liked_if_seen):
    """Record feedback for couple recommendations."""
    try:
        numeric_id, session_id = get_or_create_numeric_session_id()
        
        combined_favorites = f"Person1: {', '.join(PERSON1_MOVIES)} | Person2: {', '.join(PERSON2_MOVIES)}"
        
        success = record_feedback_to_sheet(
            numeric_session_id=numeric_id,
            uuid_session_id=session_id,
            user_top_5_movies=combined_favorites,
            user_taste_profile="couple_fusion",
            user_favorite_genres="mixed_couple_preferences",
            recommendation_rank=rank,
            movie_id="COUPLE_REC",
            movie_title=title,
            movie_genres="various",
            movie_year="N/A",
            recommendation_score=score,
            recommendation_reason=explanation,
            would_watch=would_watch,
            liked_if_seen=liked_if_seen
        )
        
        if not success:
            st.warning("Could not save to Google Sheets, but feedback recorded locally.")
            
    except Exception as e:
        st.error(f"Error recording feedback: {e}")

def generate_couple_recommendations(person1_movies, person2_movies):
    """Generate recommendations for the couple using K-Means clustering."""
    if not couple_scoring_available:
        st.error("Couple scoring module not available. Cannot generate recommendations.")
        return []
    
    # Check cache first
    cache_key = f"couple_recs_{hash(tuple(person1_movies + person2_movies))}"
    if cache_key in st.session_state.couple_cache:
        return st.session_state.couple_cache[cache_key]
    
    with st.spinner("🎯 Analyzing both your tastes and finding perfect matches..."):
        try:
            # Call the recommendation function from couple_scoring module
            recommendations = recommend_movies_for_couple(person1_movies, person2_movies)
            
            # Cache the results
            st.session_state.couple_cache[cache_key] = recommendations
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.error(f"Error details: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return []

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Couples Movie Matcher",
        page_icon="🎬",
        layout="wide"
    )
    
    initialize_couples_session_state()
    
    # Debug info (expandable)
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        st.write(f"Python path includes: {src_dir}")
        st.write(f"Feedback available: {feedback_available}")
        st.write(f"Couple scoring available: {couple_scoring_available}")
    
    # Display favorite movies
    person1_movies, person2_movies = display_favorite_movies()
    
    # Generate recommendations button
    st.markdown("---")
    if st.button("🎯 Generate Couple Recommendations", type="primary"):
        recommendations = generate_couple_recommendations(person1_movies, person2_movies)
        st.session_state.couple_recommendations = recommendations
        st.session_state.show_couple_feedback = True
    
    # Display recommendations if available
    if st.session_state.couple_recommendations and st.session_state.show_couple_feedback:
        display_couple_recommendations(st.session_state.couple_recommendations)
    
    # Final feedback section
    if st.session_state.couple_recommendations and feedback_available:
        st.markdown("---")
        st.subheader("💬 Overall Feedback")
        
        final_comments = st.text_area(
            "Any thoughts on these recommendations? What worked well or didn't?",
            placeholder="These recommendations were great because... / I wish there were more...",
            height=100
        )
        
        if st.button("Submit Final Feedback"):
            if final_comments.strip():
                numeric_id, session_id = get_or_create_numeric_session_id()
                combined_favorites = f"Person1: {', '.join(PERSON1_MOVIES)} | Person2: {', '.join(PERSON2_MOVIES)}"
                
                success = record_final_comments_to_sheet(
                    numeric_session_id=numeric_id,
                    uuid_session_id=session_id,
                    user_top_5_movies=combined_favorites,
                    user_taste_profile="couple_fusion",
                    user_favorite_genres="mixed_couple_preferences",
                    final_comments=final_comments
                )
                
                if success:
                    st.success("Thanks for your feedback! This will help improve our couple recommendations.")
                else:
                    st.warning("Feedback saved locally (couldn't reach Google Sheets)")
            else:
                st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()

    #test status
