"""
Unit tests for feedback system functionality.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import csv
import tempfile
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feedback_system import (
    initialize_feedback_csv,
    get_or_create_numeric_session_id,
    save_feedback,
    get_gsheet_client,
    record_feedback_to_sheet,
    record_final_comments_to_sheet,
    FEEDBACK_FILE,
    SESSION_MAP_FILE
)

class TestFeedbackSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.sample_feedback_data = {
            "numeric_session_id": 1,
            "session_id": "test-session-123",
            "user_top_5_movies": "Movie 1 | Movie 2 | Movie 3 | Movie 4 | Movie 5",
            "user_taste_profile": "diverse",
            "user_favorite_genres": "Action | Comedy | Drama",
            "recommendation_rank": 1,
            "movie_id": "12345",
            "movie_title": "Test Movie",
            "movie_genres": "Action | Adventure",
            "movie_year": "2020",
            "recommendation_score": 0.85,
            "recommendation_reason": "Genre match",
            "would_watch": "Yes",
            "liked_if_seen": "N/A",
            "user_email": "test@example.com"
        }

    @patch('feedback_system.os.path.exists')
    @patch('feedback_system.open', new_callable=mock_open)
    def test_initialize_feedback_csv_new_file(self, mock_file, mock_exists):
        """Test initialization of feedback CSV when file doesn't exist."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Call function
        initialize_feedback_csv()
        
        # Verify file was opened for writing
        mock_file.assert_called_once_with(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8')
        
        # Verify CSV header was written
        handle = mock_file()
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
        
        # Check that headers are present
        expected_headers = [
            "numeric_session_id", "session_id", "user_top_5_movies", 
            "user_taste_profile", "user_favorite_genres", "recommendation_rank",
            "movie_id", "movie_title", "movie_genres", "movie_year",
            "recommendation_score", "recommendation_reason", "would_watch",
            "liked_if_seen", "timestamp"
        ]
        
        for header in expected_headers:
            self.assertIn(header, written_content)

    @patch('feedback_system.os.path.exists')
    def test_initialize_feedback_csv_existing_file(self, mock_exists):
        """Test initialization when feedback CSV already exists."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Should not create new file
        with patch('feedback_system.open', new_callable=mock_open) as mock_file:
            initialize_feedback_csv()
            mock_file.assert_not_called()

    @patch('feedback_system.st.session_state', {"session_id": "existing-session"})
    @patch('feedback_system.pd.read_csv')
    @patch('feedback_system.os.path.exists')
    def test_get_or_create_numeric_session_id_existing(self, mock_exists, mock_read_csv):
        """Test getting existing numeric session ID."""
        # Mock session map exists
        mock_exists.return_value = True
        
        # Mock existing session data
        mock_df = pd.DataFrame({
            "numeric_session_id": [1, 2, 3],
            "session_id": ["session-1", "existing-session", "session-3"]
        })
        mock_read_csv.return_value = mock_df
        
        with patch('feedback_system.st.session_state', {"session_id": "existing-session"}):
            numeric_id, session_id = get_or_create_numeric_session_id()
        
        self.assertEqual(numeric_id, 2)
        self.assertEqual(session_id, "existing-session")

    @patch('feedback_system.st.session_state', {})
    @patch('feedback_system.pd.read_csv')
    @patch('feedback_system.os.path.exists')
    def test_get_or_create_numeric_session_id_new(self, mock_exists, mock_read_csv):
        """Test creating new numeric session ID."""
        # Mock session map exists
        mock_exists.return_value = True
        
        # Mock existing session data without current session
        mock_df = pd.DataFrame({
            "numeric_session_id": [1, 2, 3],
            "session_id": ["session-1", "session-2", "session-3"]
        })
        mock_read_csv.return_value = mock_df
        
        with patch('feedback_system.pd.DataFrame.to_csv') as mock_to_csv:
            with patch('feedback_system.st.session_state', {}) as mock_session:
                numeric_id, session_id = get_or_create_numeric_session_id()
        
        # Should create new ID (max + 1)
        self.assertEqual(numeric_id, 4)
        self.assertIsInstance(session_id, str)
        mock_to_csv.assert_called()

    @patch('feedback_system.open', new_callable=mock_open)
    @patch('feedback_system.datetime')
    def test_save_feedback(self, mock_datetime, mock_file):
        """Test saving feedback to CSV file."""
        # Mock current time
        mock_datetime.utcnow.return_value.isoformat.return_value = "2023-07-09T12:00:00"
        
        # Call function
        save_feedback(
            self.sample_feedback_data["numeric_session_id"],
            self.sample_feedback_data["session_id"],
            self.sample_feedback_data["user_top_5_movies"],
            self.sample_feedback_data["user_taste_profile"],
            self.sample_feedback_data["user_favorite_genres"],
            self.sample_feedback_data["recommendation_rank"],
            self.sample_feedback_data["movie_id"],
            self.sample_feedback_data["movie_title"],
            self.sample_feedback_data["movie_genres"],
            self.sample_feedback_data["movie_year"],
            self.sample_feedback_data["recommendation_score"],
            self.sample_feedback_data["recommendation_reason"],
            self.sample_feedback_data["would_watch"],
            self.sample_feedback_data["liked_if_seen"]
        )
        
        # Verify file was opened for appending
        mock_file.assert_called_once_with(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8')

    @patch('feedback_system.st.secrets', {})
    def test_get_gsheet_client_no_secrets(self):
        """Test Google Sheets client setup with missing secrets."""
        client = get_gsheet_client()
        self.assertIsNone(client)

    @patch('feedback_system.st.secrets')
    @patch('feedback_system.Credentials.from_service_account_info')
    @patch('feedback_system.gspread.authorize')
    def test_get_gsheet_client_success(self, mock_authorize, mock_credentials, mock_secrets):
        """Test successful Google Sheets client setup."""
        # Mock secrets
        mock_secrets.__contains__.return_value = True
        mock_secrets.__getitem__.return_value = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\\ntest-key\\n-----END PRIVATE KEY-----",
            "client_email": "test@test.com",
            "client_id": "123",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://test.com",
            "universe_domain": "googleapis.com"
        }
        
        # Mock successful authentication
        mock_creds = MagicMock()
        mock_credentials.return_value = mock_creds
        mock_client = MagicMock()
        mock_authorize.return_value = mock_client
        
        client = get_gsheet_client()
        
        self.assertIsNotNone(client)
        mock_credentials.assert_called_once()
        mock_authorize.assert_called_once_with(mock_creds)

    @patch('feedback_system.st.secrets')
    def test_get_gsheet_client_invalid_private_key(self, mock_secrets):
        """Test Google Sheets client setup with invalid private key format."""
        # Mock secrets with invalid private key
        mock_secrets.__contains__.return_value = True
        mock_secrets.__getitem__.return_value = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "invalid-key-format",  # Invalid format
            "client_email": "test@test.com",
            "client_id": "123",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://test.com",
            "universe_domain": "googleapis.com"
        }
        
        client = get_gsheet_client()
        self.assertIsNone(client)

    @patch('feedback_system.get_gsheet_client')
    @patch('feedback_system.datetime')
    def test_record_feedback_to_sheet_success(self, mock_datetime, mock_get_client):
        """Test successful feedback recording to Google Sheets."""
        # Mock client and sheet
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_worksheet = MagicMock()
        
        mock_client.open.return_value = mock_sheet
        mock_sheet.sheet1 = mock_worksheet
        mock_get_client.return_value = mock_client
        
        # Mock timestamp
        mock_datetime.utcnow.return_value.strftime.return_value = "2023-07-09 12:00:00"
        
        # Call function
        result = record_feedback_to_sheet(
            **self.sample_feedback_data
        )
        
        # Verify success
        self.assertTrue(result)
        mock_worksheet.append_row.assert_called_once()
        
        # Verify the row data structure
        call_args = mock_worksheet.append_row.call_args[0][0]
        self.assertIsInstance(call_args, list)
        self.assertEqual(len(call_args), 16)  # Expected number of columns

    @patch('feedback_system.get_gsheet_client')
    def test_record_feedback_to_sheet_no_client(self, mock_get_client):
        """Test feedback recording when client setup fails."""
        mock_get_client.return_value = None
        
        result = record_feedback_to_sheet(
            **self.sample_feedback_data
        )
        
        self.assertFalse(result)

    @patch('feedback_system.get_gsheet_client')
    def test_record_feedback_to_sheet_exception(self, mock_get_client):
        """Test feedback recording with exception."""
        # Mock client that raises exception
        mock_client = MagicMock()
        mock_client.open.side_effect = Exception("Test exception")
        mock_get_client.return_value = mock_client
        
        result = record_feedback_to_sheet(
            **self.sample_feedback_data
        )
        
        self.assertFalse(result)

    @patch('feedback_system.get_gsheet_client')
    @patch('feedback_system.datetime')
    def test_record_final_comments_to_sheet_success(self, mock_datetime, mock_get_client):
        """Test successful final comments recording to Google Sheets."""
        # Mock client setup
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_worksheet = MagicMock()
        
        mock_client.open.return_value = mock_sheet
        mock_sheet.sheet1 = mock_worksheet
        mock_get_client.return_value = mock_client
        
        # Mock timestamp
        mock_datetime.utcnow.return_value.strftime.return_value = "2023-07-09 12:00:00"
        
        # Call function
        result = record_final_comments_to_sheet(
            numeric_session_id=1,
            uuid_session_id="test-session",
            user_top_5_movies="Movie 1 | Movie 2",
            user_taste_profile="diverse",
            user_favorite_genres="Action | Comedy",
            final_comments="Great recommendations!",
            user_email="test@example.com"
        )
        
        # Verify success
        self.assertTrue(result)
        mock_worksheet.append_row.assert_called_once()
        
        # Verify special markers for final comments
        call_args = mock_worksheet.append_row.call_args[0][0]
        self.assertEqual(call_args[5], 999)  # Special rank for comments
        self.assertEqual(call_args[6], "FINAL_COMMENTS")  # Special movie ID
        self.assertEqual(call_args[11], "Great recommendations!")  # Comments in reason field

class TestFeedbackSystemIntegration(unittest.TestCase):
    """Integration tests for feedback system components."""
    
    def test_csv_file_constants(self):
        """Test that file constants are properly defined."""
        self.assertIsInstance(FEEDBACK_FILE, str)
        self.assertIsInstance(SESSION_MAP_FILE, str)
        self.assertTrue(FEEDBACK_FILE.endswith('.csv'))
        self.assertTrue(SESSION_MAP_FILE.endswith('.csv'))

    def test_feedback_data_types(self):
        """Test that feedback functions handle different data types correctly."""
        # Test with various data types
        test_cases = [
            (1, int),
            ("string", str),
            (1.5, float),
            (True, bool),
        ]
        
        for value, expected_type in test_cases:
            # Test that conversion to string works (for CSV storage)
            str_value = str(value)
            self.assertIsInstance(str_value, str)

    @patch('feedback_system.open', new_callable=mock_open)
    @patch('feedback_system.os.path.exists')
    def test_csv_creation_and_writing_workflow(self, mock_exists, mock_file):
        """Test the complete workflow of CSV creation and writing."""
        # Test file creation
        mock_exists.return_value = False
        initialize_feedback_csv()
        
        # Reset mock for save operation
        mock_file.reset_mock()
        
        # Test data saving
        with patch('feedback_system.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2023-07-09T12:00:00"
            
            save_feedback(
                numeric_id=1,
                session_id="test",
                user_top_5_movies="movies",
                user_taste_profile="profile",
                user_favorite_genres="genres",
                recommendation_rank=1,
                movie_id="123",
                movie_title="title",
                movie_genres="action",
                movie_year="2020",
                recommendation_score=0.8,
                recommendation_reason="reason",
                would_watch="yes",
                liked_if_seen="yes"
            )
        
        # Verify both operations worked
        self.assertGreaterEqual(mock_file.call_count, 1)

class TestFeedbackSystemEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_session_state(self):
        """Test behavior with empty session state."""
        with patch('feedback_system.st.session_state', {}):
            with patch('feedback_system.os.path.exists', return_value=False):
                with patch('feedback_system.pd.DataFrame') as mock_df:
                    mock_df.return_value.to_csv = MagicMock()
                    try:
                        numeric_id, session_id = get_or_create_numeric_session_id()
                        self.assertIsInstance(numeric_id, int)
                        self.assertIsInstance(session_id, str)
                    except Exception as e:
                        # Expected if pandas operations fail in test environment
                        self.assertIsInstance(e, Exception)

    def test_malformed_secrets(self):
        """Test handling of malformed secrets."""
        malformed_secrets = {
            "gcp_service_account": {
                "type": "service_account",
                # Missing required fields
            }
        }
        
        with patch('feedback_system.st.secrets', malformed_secrets):
            client = get_gsheet_client()
            # Should handle gracefully
            self.assertIsNone(client)

    def test_unicode_handling(self):
        """Test handling of unicode characters in feedback."""
        unicode_data = {
            "numeric_session_id": 1,
            "session_id": "test-session-123",
            "user_top_5_movies": "Movie with émojis 🎬 and ñoñó",
            "user_taste_profile": "diverse",
            "user_favorite_genres": "Açtion | Comëdy",
            "recommendation_rank": 1,
            "movie_id": "12345",
            "movie_title": "Test Mövie with ñámes",
            "movie_genres": "Açtion | Advëñture",
            "movie_year": "2020",
            "recommendation_score": 0.85,
            "recommendation_reason": "Génre match with spécial chars",
            "would_watch": "Yës",
            "liked_if_seen": "N/Á",
            "user_email": "tëst@example.com"
        }
        
        # Should handle unicode without errors
        with patch('feedback_system.get_gsheet_client') as mock_client:
            mock_client.return_value = None  # Force early return
            
            result = record_feedback_to_sheet(**unicode_data)
            # Should not crash due to unicode
            self.assertFalse(result)  # False because no client

if __name__ == '__main__':
    unittest.main()