"""
User feedback collection and Google Sheets integration.
"""

import os
import csv
import uuid
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# File constants
FEEDBACK_FILE = "user_feedback.csv"
SESSION_MAP_FILE = "session_map.csv"

def initialize_feedback_csv():
    """Initialize the feedback CSV file with headers if it doesn't exist."""
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "numeric_session_id",
                "session_id",
                "user_top_5_movies",
                "user_taste_profile",
                "user_favorite_genres",
                "recommendation_rank",
                "movie_id",
                "movie_title",
                "movie_genres",
                "movie_year",
                "recommendation_score",
                "recommendation_reason",
                "would_watch",
                "liked_if_seen",
                "timestamp"
            ])

def get_or_create_numeric_session_id():
    """
    Get or create a numeric session ID for the current user session.
    
    Returns:
        Tuple of (numeric_id, session_id)
    """
    if not os.path.exists(SESSION_MAP_FILE):
        pd.DataFrame(columns=["numeric_session_id", "session_id"]).to_csv(SESSION_MAP_FILE, index=False)

    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state["session_id"] = session_id

    df = pd.read_csv(SESSION_MAP_FILE)
    if session_id in df["session_id"].values:
        numeric_id = df[df["session_id"] == session_id]["numeric_session_id"].values[0]
    else:
        numeric_id = df["numeric_session_id"].max() + 1 if not df.empty else 1
        new_entry = pd.DataFrame([[numeric_id, session_id]], columns=["numeric_session_id", "session_id"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(SESSION_MAP_FILE, index=False)

    return numeric_id, session_id

def save_feedback(numeric_id, session_id, user_top_5_movies, user_taste_profile, 
                  user_favorite_genres, recommendation_rank, movie_id, movie_title, 
                  movie_genres, movie_year, recommendation_score, recommendation_reason, 
                  would_watch, liked_if_seen):
    """
    Save feedback to local CSV file.
    
    Args:
        All the feedback parameters to save
    """
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            numeric_id,
            session_id,
            user_top_5_movies,
            user_taste_profile,
            user_favorite_genres,
            recommendation_rank,
            movie_id,
            movie_title,
            movie_genres,
            movie_year,
            recommendation_score,
            recommendation_reason,
            would_watch,
            liked_if_seen,
            datetime.utcnow().isoformat()
        ])

def get_gsheet_client():
    """
    Set up Google Sheets client using Streamlit secrets.
    
    Returns:
        Google Sheets client or None if setup fails
    """
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Check if the secret exists
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Google Cloud service account credentials not found in Streamlit secrets")
            return None
            
        # Build credentials dictionary from individual TOML fields
        creds_dict = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
        }
        
        # Fix private key format - convert literal \n to actual newlines
        private_key = creds_dict.get("private_key", "")
        if "\\n" in private_key:
            private_key = private_key.replace("\\n", "\n")
            creds_dict["private_key"] = private_key
        
        # Additional cleanup
        private_key = private_key.strip()
        if private_key.startswith('"') and private_key.endswith('"'):
            private_key = private_key[1:-1]
        creds_dict["private_key"] = private_key
        
        # Check private key format
        if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
            st.error("❌ Private key is not in correct PEM format")
            st.info("💡 Make sure your private key starts with '-----BEGIN PRIVATE KEY-----'")
            return None
            
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
        
    except Exception as e:
        st.error(f"❌ Error setting up Google Sheets client: {str(e)}")
        st.info("💡 This usually means your service account JSON is corrupted or improperly formatted.")
        return None

def record_feedback_to_sheet(numeric_session_id, uuid_session_id, user_top_5_movies, 
                           user_taste_profile, user_favorite_genres, recommendation_rank, 
                           movie_id, movie_title, movie_genres, movie_year, 
                           recommendation_score, recommendation_reason, would_watch, 
                           liked_if_seen, user_email=""):
    """
    Record user feedback to Google Sheets.
    
    Args:
        All feedback parameters plus user_email
    
    Returns:
        Boolean indicating success
    """
    try:
        sheet_name = "user_feedback"
        client = get_gsheet_client()
        if client is None:
            st.error("❌ Could not connect to Google Sheets. Please check your credentials.")
            return False

        sheet = client.open(sheet_name).sheet1

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Convert all values to safe Python built-ins
        row = [
            int(numeric_session_id),
            str(uuid_session_id),
            str(user_top_5_movies),
            str(user_taste_profile),
            str(user_favorite_genres),
            int(recommendation_rank),
            str(movie_id),
            str(movie_title),
            str(movie_genres),
            str(movie_year),
            float(recommendation_score),
            str(recommendation_reason),
            str(would_watch),
            str(liked_if_seen),
            str(user_email),
            str(timestamp)
        ]

        sheet.append_row(row)
        return True

    except Exception as e:
        st.error(f"❌ Error saving to Google Sheets: {str(e)}")
        return False

def record_final_comments_to_sheet(numeric_session_id, uuid_session_id, user_top_5_movies, 
                                 user_taste_profile, user_favorite_genres, final_comments, 
                                 user_email=""):
    """
    Record user's final comments to Google Sheets.
    
    Args:
        Session and user data plus final comments
    
    Returns:
        Boolean indicating success
    """
    try:
        sheet_name = "user_feedback"
        client = get_gsheet_client()
        if client is None:
            st.error("❌ Could not connect to Google Sheets. Please check your credentials.")
            return False

        sheet = client.open(sheet_name).sheet1

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Create row data matching existing headers structure
        row = [
            int(numeric_session_id),
            str(uuid_session_id),
            str(user_top_5_movies),
            str(user_taste_profile),
            str(user_favorite_genres),
            999,  # RecommendationRank - use 999 to indicate final comments
            "FINAL_COMMENTS",  # MovieID
            "Final User Comments",  # MovieTitle
            "",  # MovieGenres - empty
            "",  # MovieYear - empty
            0.0,  # RecommendationScore - 0 for comments
            final_comments,  # RecommendationReason - store comments here
            "N/A",  # WouldWatch
            "N/A",  # LikedIfSeen
            str(user_email),  # UserEmail
            str(timestamp)
        ]

        sheet.append_row(row)
        return True

    except Exception as e:
        st.error(f"❌ Error saving final comments to Google Sheets: {str(e)}")
        return False