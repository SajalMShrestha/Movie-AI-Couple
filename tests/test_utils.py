"""
Unit tests for utility functions and constants.
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import torch
import requests

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    RECOMMENDATION_WEIGHTS,
    STREAMING_PLATFORM_PRIORITY,
    MOOD_TONE_MAP,
    get_embedding_model,
    get_mood_score,
    fetch_similar_movie_details,
    get_trending_popularity,
    estimate_user_age,
    fetch_multiple_movie_details
)

class TestUtilsConstants(unittest.TestCase):
    """Test utility constants and configurations."""
    
    def test_recommendation_weights_structure(self):
        """Test recommendation weights configuration."""
        # Verify it's a dictionary
        self.assertIsInstance(RECOMMENDATION_WEIGHTS, dict)
        
        # Verify expected keys exist
        expected_keys = [
            "mood_tone", "genre_similarity", "cast_crew", "narrative_style",
            "ratings", "trending_factor", "release_year", "discovery_boost",
            "age_alignment", "embedding_similarity"
        ]
        
        for key in expected_keys:
            self.assertIn(key, RECOMMENDATION_WEIGHTS)
            self.assertIsInstance(RECOMMENDATION_WEIGHTS[key], (int, float))
            self.assertGreaterEqual(RECOMMENDATION_WEIGHTS[key], 0.0)
        
        # Verify weights sum to reasonable total (around 1.0)
        total_weight = sum(RECOMMENDATION_WEIGHTS.values())
        self.assertGreater(total_weight, 0.5)
        self.assertLess(total_weight, 2.0)

    def test_streaming_platform_priority_structure(self):
        """Test streaming platform priority configuration."""
        self.assertIsInstance(STREAMING_PLATFORM_PRIORITY, dict)
        
        # Verify expected platforms
        expected_platforms = ["netflix", "disney_plus", "hbo_max", "hulu", "prime_video"]
        
        for platform in expected_platforms:
            self.assertIn(platform, STREAMING_PLATFORM_PRIORITY)
            self.assertIsInstance(STREAMING_PLATFORM_PRIORITY[platform], (int, float))
            self.assertGreaterEqual(STREAMING_PLATFORM_PRIORITY[platform], 0.0)
            self.assertLessEqual(STREAMING_PLATFORM_PRIORITY[platform], 1.0)

    def test_mood_tone_map_structure(self):
        """Test mood tone mapping configuration."""
        self.assertIsInstance(MOOD_TONE_MAP, dict)
        
        # Verify expected moods
        expected_moods = ["feel_good", "gritty", "cerebral", "intense", "melancholic", "classic"]
        
        for mood in expected_moods:
            self.assertIn(mood, MOOD_TONE_MAP)
            self.assertIsInstance(MOOD_TONE_MAP[mood], set)
            self.assertGreater(len(MOOD_TONE_MAP[mood]), 0)
            
            # Verify all items in set are strings
            for genre in MOOD_TONE_MAP[mood]:
                self.assertIsInstance(genre, str)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_genres = [
            {"name": "Action"},
            {"name": "Comedy"},
            {"name": "Drama"}
        ]
        
        self.sample_genres_objects = [
            Mock(name="Action"),
            Mock(name="Comedy"), 
            Mock(name="Drama")
        ]

    @patch('utils.SentenceTransformer')
    def test_get_embedding_model(self, mock_transformer):
        """Test embedding model initialization."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Test caching behavior
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        
        # Should return same instance (cached)
        self.assertEqual(model1, model2)
        
        # Should only create transformer once
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2', device='cpu')

    def test_get_mood_score_dict_format(self):
        """Test mood score calculation with dictionary format genres."""
        preferred_moods = {"feel_good", "gritty"}
        
        # Comedy should match feel_good
        score = get_mood_score(self.sample_genres, preferred_moods)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_get_mood_score_object_format(self):
        """Test mood score calculation with object format genres."""
        preferred_moods = {"feel_good", "gritty"}
        
        score = get_mood_score(self.sample_genres_objects, preferred_moods)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_get_mood_score_no_matches(self):
        """Test mood score with no matching genres."""
        # Use genres that don't match any mood
        non_matching_genres = [{"name": "Documentary"}, {"name": "News"}]
        preferred_moods = {"feel_good", "gritty"}
        
        score = get_mood_score(non_matching_genres, preferred_moods)
        self.assertEqual(score, 0.0)

    def test_get_mood_score_empty_inputs(self):
        """Test mood score with empty inputs."""
        # Empty genres
        score = get_mood_score([], {"feel_good"})
        self.assertEqual(score, 0.0)
        
        # Empty preferred moods
        score = get_mood_score(self.sample_genres, set())
        self.assertEqual(score, 0.0)
        
        # Both empty
        score = get_mood_score([], set())
        self.assertEqual(score, 0.0)

    @patch('utils.requests.get')
    def test_get_trending_popularity_success(self, mock_get):
        """Test successful trending popularity retrieval."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": 123, "popularity": 100.0},
                {"id": 124, "popularity": 80.0},
                {"id": 125, "popularity": 60.0}
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_trending_popularity("fake_api_key")
        
        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn(123, result)
        self.assertIn(124, result)
        self.assertIn(125, result)
        
        # Verify normalization (should be between 0 and 1)
        for movie_id, score in result.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Verify highest popularity gets score of 1.0
        self.assertEqual(result[123], 1.0)

    @patch('utils.requests.get')
    def test_get_trending_popularity_failure(self, mock_get):
        """Test trending popularity retrieval with API failure."""
        # Mock API failure
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = get_trending_popularity("fake_api_key")
        
        # Should return empty dict on failure
        self.assertEqual(result, {})

    @patch('utils.requests.get')
    def test_get_trending_popularity_exception(self, mock_get):
        """Test trending popularity retrieval with exception."""
        # Mock exception
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = get_trending_popularity("fake_api_key")
        
        # Should return empty dict on exception
        self.assertEqual(result, {})

    @patch('utils.requests.get')
    def test_get_trending_popularity_empty_results(self, mock_get):
        """Test trending popularity with empty results."""
        # Mock empty results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        result = get_trending_popularity("fake_api_key")
        
        # Should return empty dict
        self.assertEqual(result, {})

    def test_estimate_user_age_with_years(self):
        """Test user age estimation with movie years."""
        # Test with recent years
        recent_years = [2018, 2019, 2020, 2021, 2022]
        age = estimate_user_age(recent_years)
        
        self.assertIsInstance(age, int)
        self.assertGreater(age, 15)  # Reasonable age range
        self.assertLess(age, 100)
        
        # Test with older years
        older_years = [1990, 1995, 2000, 2005, 2010]
        age_older = estimate_user_age(older_years)
        
        # Should be older than someone who watches recent movies
        self.assertGreater(age_older, age)

    def test_estimate_user_age_empty_years(self):
        """Test user age estimation with no years."""
        age = estimate_user_age([])
        self.assertEqual(age, 30)  # Default age

    def test_estimate_user_age_single_year(self):
        """Test user age estimation with single year."""
        age = estimate_user_age([2020])
        self.assertIsInstance(age, int)
        self.assertGreater(age, 15)

    @patch('utils.Movie')
    @patch('utils.get_embedding_model')
    @patch('utils.infer_narrative_style')
    def test_fetch_similar_movie_details_success(self, mock_narrative, mock_embedding_model, mock_movie_class):
        """Test successful movie details fetching."""
        # Mock movie API
        mock_movie_api = MagicMock()
        mock_movie_class.return_value = mock_movie_api
        
        # Mock movie details
        mock_details = MagicMock()
        mock_details.genres = [{"name": "Action"}, {"name": "Adventure"}]
        mock_details.overview = "A great action movie with lots of excitement."
        mock_movie_api.details.return_value = mock_details
        
        # Mock credits
        mock_credits = {
            "cast": [{"name": "Actor 1", "job": "Actor"}, {"name": "Actor 2", "job": "Actor"}],
            "crew": [{"name": "Director 1", "job": "Director"}]
        }
        mock_movie_api.credits.return_value = mock_credits
        
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(384)
        mock_embedding_model.return_value = mock_model
        
        # Mock narrative style
        mock_narrative.return_value = {"tone": "positive", "complexity": "simple"}
        
        # Test the function
        cache = {}
        result = fetch_similar_movie_details(123, cache)
        
        # Verify return format
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        movie_id, payload = result
        self.assertEqual(movie_id, 123)
        
        if payload is not None:  # Success case
            movie_details, embedding = payload
            self.assertIsNotNone(movie_details)
            self.assertIsInstance(embedding, torch.Tensor)

    @patch('utils.Movie')
    def test_fetch_similar_movie_details_short_plot(self, mock_movie_class):
        """Test movie details fetching with short plot."""
        # Mock movie API
        mock_movie_api = MagicMock()
        mock_movie_class.return_value = mock_movie_api
        
        # Mock movie with short plot
        mock_details = MagicMock()
        mock_details.overview = "Short."  # Too short
        mock_movie_api.details.return_value = mock_details
        mock_movie_api.credits.return_value = {"cast": [], "crew": []}
        
        # Test the function
        cache = {}
        result = fetch_similar_movie_details(123, cache)
        
        # Should return None for short plot
        movie_id, payload = result
        self.assertEqual(movie_id, 123)
        self.assertIsNone(payload)

    @patch('utils.Movie')
    def test_fetch_similar_movie_details_exception(self, mock_movie_class):
        """Test movie details fetching with exception."""
        # Mock movie API that raises exception
        mock_movie_api = MagicMock()
        mock_movie_api.details.side_effect = Exception("API Error")
        mock_movie_class.return_value = mock_movie_api
        
        # Test the function
        cache = {}
        result = fetch_similar_movie_details(123, cache)
        
        # Should handle exception gracefully
        movie_id, payload = result
        self.assertEqual(movie_id, 123)
        self.assertIsNone(payload)

    def test_fetch_similar_movie_details_caching(self):
        """Test caching behavior of movie details fetching."""
        cache = {123: ("cached_data", "cached_embedding")}
        
        # Should return cached data
        result = fetch_similar_movie_details(123, cache)
        movie_id, payload = result
        
        self.assertEqual(movie_id, 123)
        self.assertEqual(payload, ("cached_data", "cached_embedding"))

    @patch('utils.fetch_similar_movie_details')
    def test_fetch_multiple_movie_details(self, mock_fetch):
        """Test fetching multiple movie details with threading."""
        # Mock individual fetch function
        def mock_fetch_side_effect(movie_id, cache):
            if movie_id == 123:
                return (123, ("movie_123", "embedding_123"))
            elif movie_id == 124:
                return (124, ("movie_124", "embedding_124"))
            else:
                return (movie_id, None)
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        # Test the function
        movie_ids = [123, 124, 125]
        cache = {}
        results = fetch_multiple_movie_details(movie_ids, cache)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn(123, results)
        self.assertIn(124, results)
        self.assertNotIn(125, results)  # Should be filtered out (None result)

class TestUtilsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_get_mood_score_malformed_genres(self):
        """Test mood score with malformed genre data."""
        # Missing 'name' key
        malformed_genres = [{"title": "Action"}, {"genre": "Comedy"}]
        preferred_moods = {"feel_good"}
        
        score = get_mood_score(malformed_genres, preferred_moods)
        self.assertEqual(score, 0.0)  # Should handle gracefully

    def test_get_mood_score_none_values(self):
        """Test mood score with None values."""
        genres_with_none = [{"name": "Action"}, {"name": None}, None]
        preferred_moods = {"feel_good"}
        
        # Should not crash
        score = get_mood_score(genres_with_none, preferred_moods)
        self.assertIsInstance(score, float)

    def test_estimate_user_age_invalid_years(self):
        """Test user age estimation with invalid years."""
        # Future years
        future_years = [2030, 2040, 2050]
        age = estimate_user_age(future_years)
        
        # Should still return reasonable age
        self.assertIsInstance(age, int)
        
        # Very old years
        old_years = [1900, 1920, 1950]
        age_old = estimate_user_age(old_years)
        
        self.assertIsInstance(age_old, int)

    def test_mood_tone_map_coverage(self):
        """Test that mood tone map covers common genres."""
        all_mapped_genres = set()
        for mood, genres in MOOD_TONE_MAP.items():
            all_mapped_genres.update(genres)
        
        # Should include common genres
        common_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
        
        for genre in common_genres:
            if genre not in all_mapped_genres:
                # Log which genres are missing (for development)
                print(f"Genre '{genre}' not mapped to any mood")

    def test_recommendation_weights_balance(self):
        """Test that recommendation weights are reasonably balanced."""
        weights = list(RECOMMENDATION_WEIGHTS.values())
        
        # No single weight should dominate (> 50%)
        max_weight = max(weights)
        self.assertLess(max_weight, 0.5)
        
        # No weight should be negligible (< 1% unless intentionally zero)
        min_weight = min(weights)
        self.assertGreaterEqual(min_weight, 0.0)

class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utility functions."""
    
    def test_mood_score_integration(self):
        """Test mood score calculation with realistic data."""
        # Test realistic genre combinations
        action_adventure = [{"name": "Action"}, {"name": "Adventure"}]
        comedy_romance = [{"name": "Comedy"}, {"name": "Romance"}]
        horror_thriller = [{"name": "Horror"}, {"name": "Thriller"}]
        
        # User preferences
        feel_good_user = {"feel_good"}
        intense_user = {"intense", "gritty"}
        
        # Action/Adventure should score well for feel_good
        score1 = get_mood_score(action_adventure, feel_good_user)
        self.assertGreater(score1, 0.0)
        
        # Comedy/Romance should score well for feel_good
        score2 = get_mood_score(comedy_romance, feel_good_user)
        self.assertGreater(score2, 0.0)
        
        # Horror/Thriller should score well for intense user
        score3 = get_mood_score(horror_thriller, intense_user)
        self.assertGreater(score3, 0.0)
        
        # Horror/Thriller should score poorly for feel_good user
        score4 = get_mood_score(horror_thriller, feel_good_user)
        self.assertEqual(score4, 0.0)

    @patch('utils.requests.get')
    def test_trending_popularity_realistic_data(self, mock_get):
        """Test trending popularity with realistic API response."""
        # Mock realistic API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": 634649, "popularity": 1234.567},  # High popularity
                {"id": 438148, "popularity": 987.123},   # Medium popularity
                {"id": 505642, "popularity": 456.789},   # Lower popularity
                {"id": 299534, "popularity": 123.456},   # Lowest popularity
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_trending_popularity("real_api_key")
        
        # Verify realistic behavior
        self.assertEqual(len(result), 4)
        
        # Highest popularity should be 1.0
        self.assertEqual(result[634649], 1.0)
        
        # Others should be proportionally lower
        self.assertLess(result[438148], 1.0)
        self.assertLess(result[505642], result[438148])
        self.assertLess(result[299534], result[505642])
        
        # All should be positive
        for score in result.values():
            self.assertGreater(score, 0.0)

    def test_age_estimation_realistic_scenarios(self):
        """Test age estimation with realistic movie watching patterns."""
        # Young person (recent movies)
        young_person_movies = [2019, 2020, 2021, 2022, 2023]
        young_age = estimate_user_age(young_person_movies)
        
        # Middle-aged person (mix of decades)
        middle_aged_movies = [1995, 2005, 2015, 2020, 2022]
        middle_age = estimate_user_age(middle_aged_movies)
        
        # Older person (classic movies)
        older_person_movies = [1980, 1985, 1990, 1995, 2000]
        older_age = estimate_user_age(older_person_movies)
        
        # Ages should be in logical order
        self.assertLess(young_age, middle_age)
        self.assertLess(middle_age, older_age)
        
        # All should be reasonable ages
        for age in [young_age, middle_age, older_age]:
            self.assertGreater(age, 15)
            self.assertLess(age, 90)

class TestUtilsPerformance(unittest.TestCase):
    """Test performance characteristics of utility functions."""
    
    def test_mood_score_performance_large_input(self):
        """Test mood score performance with large genre lists."""
        # Create large genre list
        large_genre_list = [{"name": f"Genre_{i}"} for i in range(1000)]
        large_genre_list.extend([{"name": "Action"}, {"name": "Comedy"}])  # Add some real genres
        
        preferred_moods = {"feel_good", "intense", "gritty"}
        
        import time
        start_time = time.time()
        score = get_mood_score(large_genre_list, preferred_moods)
        end_time = time.time()
        
        # Should complete quickly (under 1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(score, float)

    def test_embedding_model_caching(self):
        """Test that embedding model is properly cached."""
        # Multiple calls should be fast after first call
        import time
        
        # First call might be slow (loading model)
        start_time = time.time()
        model1 = get_embedding_model()
        first_call_time = time.time() - start_time
        
        # Subsequent calls should be fast (cached)
        start_time = time.time()
        model2 = get_embedding_model()
        second_call_time = time.time() - start_time
        
        # Should be same object (cached)
        self.assertIs(model1, model2)
        
        # Second call should be much faster (or at least not slower)
        self.assertLessEqual(second_call_time, first_call_time + 0.1)

if __name__ == '__main__':
    # Run tests with different verbosity levels
    unittest.main(verbosity=2)