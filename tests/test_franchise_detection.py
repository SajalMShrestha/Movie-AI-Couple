"""
Unit tests for franchise detection functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from franchise_detection import (
    extract_base_title_simple,
    get_franchise_key_robust,
    apply_final_franchise_limit,
    debug_franchise_keys
)

class TestFranchiseDetection(unittest.TestCase):
    
    def test_extract_base_title_simple(self):
        """Test base title extraction."""
        test_cases = [
            # (input, expected_output)
            ("The Dark Knight", "the dark knight"),
            ("Iron Man 2", "iron man"),
            ("Avengers: Endgame", "avengers"),
            ("Star Wars: Episode IV - A New Hope", "star wars"),
            ("The Godfather (1972)", "the godfather"),
            ("Rocky II", "rocky"),
            ("Mission: Impossible", "mission"),
            ("Spider-Man: Into the Spider-Verse", "spider-man"),
            ("The Matrix", "the matrix"),
            ("Fast & Furious 6", "fast & furious"),
            ("John Wick: Chapter 2", "john wick"),
            ("Pirates of the Caribbean: The Curse of the Black Pearl", "pirates of the caribbean"),
        ]
        
        for input_title, expected in test_cases:
            with self.subTest(input_title=input_title):
                result = extract_base_title_simple(input_title)
                self.assertEqual(result, expected)

    def test_extract_base_title_edge_cases(self):
        """Test edge cases for base title extraction."""
        # Empty string
        result = extract_base_title_simple("")
        self.assertEqual(result, "")
        
        # Single word
        result = extract_base_title_simple("Inception")
        self.assertEqual(result, "inception")
        
        # Numbers at the end
        result = extract_base_title_simple("Movie 123")
        self.assertEqual(result, "movie")
        
        # Roman numerals
        result = extract_base_title_simple("Rocky III")
        self.assertEqual(result, "rocky")
        
        # Multiple patterns
        result = extract_base_title_simple("The Dark Knight: Chapter 2 (2019)")
        self.assertEqual(result, "the dark knight")

    def test_get_franchise_key_robust_simple_fallback(self):
        """Test franchise key generation with simple fallback."""
        # Test when movie object is not found (fallback to simple)
        candidates = {}
        
        result = get_franchise_key_robust("The Dark Knight", candidates)
        expected = extract_base_title_simple("The Dark Knight")
        self.assertEqual(result, expected)

    def test_get_franchise_key_robust_with_cast(self):
        """Test franchise key generation with cast information."""
        # Create mock movie object
        mock_movie = MagicMock()
        mock_movie.title = "Iron Man"
        mock_movie.cast = [
            {"name": "Robert Downey Jr."},
            {"name": "Gwyneth Paltrow"},
            {"name": "Jeff Bridges"}
        ]
        
        # Create candidates dictionary
        candidates = {
            123: (mock_movie, None)
        }
        
        result = get_franchise_key_robust("Iron Man", candidates)
        expected = "iron man_Robert Downey Jr."
        self.assertEqual(result, expected)

    def test_get_franchise_key_robust_httyd_special_case(self):
        """Test special case for How to Train Your Dragon franchise."""
        # Create mock movie object with Jay Baruchel (Hiccup's voice)
        mock_movie = MagicMock()
        mock_movie.title = "How to Train Your Dragon"
        mock_movie.cast = [
            {"name": "Jay Baruchel"},
            {"name": "Gerard Butler"},
            {"name": "America Ferrera"}
        ]
        
        candidates = {
            123: (mock_movie, None)
        }
        
        result = get_franchise_key_robust("How to Train Your Dragon", candidates)
        expected = "httyd_jay_baruchel_dragon"
        self.assertEqual(result, expected)

    def test_get_franchise_key_robust_cast_object_format(self):
        """Test franchise key generation with object-format cast."""
        # Create mock movie object with object-style cast
        mock_actor = MagicMock()
        mock_actor.name = "Tom Hanks"
        
        mock_movie = MagicMock()
        mock_movie.title = "Toy Story"
        mock_movie.cast = [mock_actor]
        
        candidates = {
            123: (mock_movie, None)
        }
        
        result = get_franchise_key_robust("Toy Story", candidates)
        expected = "toy story_Tom Hanks"
        self.assertEqual(result, expected)

    def test_apply_final_franchise_limit_basic(self):
        """Test basic franchise limiting functionality."""
        # Create test recommendations
        recommendations = [
            ("Iron Man", 0.9),
            ("Iron Man 2", 0.8),
            ("The Dark Knight", 0.85),
            ("The Dark Knight Rises", 0.75),
            ("Inception", 0.7),
            ("Interstellar", 0.65),
        ]
        
        # Create mock candidates
        candidates = {}
        for title, score in recommendations:
            mock_movie = MagicMock()
            mock_movie.title = title
            mock_movie.cast = [{"name": "Actor Name"}]
            candidates[hash(title)] = (mock_movie, None)
        
        # Apply franchise limiting
        result = apply_final_franchise_limit(recommendations, candidates, max_per_franchise=1)
        
        # Should return limited results
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), len(recommendations))
        
        # Verify it's a list of (title, score) tuples
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)  # title
            self.assertIsInstance(item[1], (int, float))  # score

    def test_apply_final_franchise_limit_empty_input(self):
        """Test franchise limiting with empty input."""
        result = apply_final_franchise_limit([], {}, max_per_franchise=1)
        self.assertEqual(result, [])

    def test_apply_final_franchise_limit_no_candidates(self):
        """Test franchise limiting with no candidate data."""
        recommendations = [("Movie 1", 0.9), ("Movie 2", 0.8)]
        candidates = {}
        
        result = apply_final_franchise_limit(recommendations, candidates, max_per_franchise=1)
        # Should handle gracefully
        self.assertIsInstance(result, list)

    def test_apply_final_franchise_limit_max_ten_results(self):
        """Test that franchise limiting returns maximum 10 results."""
        # Create 15 recommendations
        recommendations = [(f"Movie {i}", 0.9 - i*0.01) for i in range(15)]
        
        # Create mock candidates
        candidates = {}
        for title, score in recommendations:
            mock_movie = MagicMock()
            mock_movie.title = title
            mock_movie.cast = [{"name": f"Actor {title}"}]  # Different actors for different franchises
            candidates[hash(title)] = (mock_movie, None)
        
        result = apply_final_franchise_limit(recommendations, candidates, max_per_franchise=1)
        
        # Should return maximum 10 results
        self.assertLessEqual(len(result), 10)

    def test_apply_final_franchise_limit_franchise_grouping(self):
        """Test that movies from same franchise are properly grouped."""
        # Create recommendations with clear franchise patterns
        recommendations = [
            ("Iron Man", 0.95),
            ("Iron Man 2", 0.90),
            ("Iron Man 3", 0.85),
            ("Batman Begins", 0.80),
            ("The Dark Knight", 0.88),
            ("Inception", 0.75),
        ]
        
        # Create candidates with same main cast for franchise movies
        candidates = {}
        
        # Iron Man movies share Robert Downey Jr.
        for title in ["Iron Man", "Iron Man 2", "Iron Man 3"]:
            mock_movie = MagicMock()
            mock_movie.title = title
            mock_movie.cast = [{"name": "Robert Downey Jr."}, {"name": "Other Actor"}]
            candidates[hash(title)] = (mock_movie, None)
        
        # Batman movies share Christian Bale
        for title in ["Batman Begins", "The Dark Knight"]:
            mock_movie = MagicMock()
            mock_movie.title = title
            mock_movie.cast = [{"name": "Christian Bale"}, {"name": "Other Actor"}]
            candidates[hash(title)] = (mock_movie, None)
        
        # Inception is standalone
        mock_movie = MagicMock()
        mock_movie.title = "Inception"
        mock_movie.cast = [{"name": "Leonardo DiCaprio"}]
        candidates[hash("Inception")] = (mock_movie, None)
        
        result = apply_final_franchise_limit(recommendations, candidates, max_per_franchise=1)
        
        # Should have maximum one movie per franchise
        result_titles = [title for title, score in result]
        
        # Should not have multiple Iron Man movies
        iron_man_count = sum(1 for title in result_titles if "iron man" in title.lower())
        self.assertLessEqual(iron_man_count, 1)
        
        # Should not have multiple Batman movies
        batman_count = sum(1 for title in result_titles if "batman" in title.lower() or "dark knight" in title.lower())
        self.assertLessEqual(batman_count, 1)

    def test_debug_franchise_keys(self):
        """Test debug function for franchise keys."""
        recommendations = [
            ("Iron Man", 0.9),
            ("The Dark Knight", 0.8),
            ("Inception", 0.7)
        ]
        
        candidates = {}
        for title, score in recommendations:
            mock_movie = MagicMock()
            mock_movie.title = title
            mock_movie.cast = [{"name": "Actor Name"}]
            candidates[hash(title)] = (mock_movie, None)
        
        # Should not raise an error
        try:
            debug_franchise_keys(recommendations, candidates)
        except Exception as e:
            self.fail(f"debug_franchise_keys raised an exception: {e}")

class TestFranchiseDetectionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_get_franchise_key_empty_cast(self):
        """Test franchise key generation with empty cast."""
        mock_movie = MagicMock()
        mock_movie.title = "Movie Title"
        mock_movie.cast = []
        
        candidates = {123: (mock_movie, None)}
        
        result = get_franchise_key_robust("Movie Title", candidates)
        expected = "movie title_unknown"
        self.assertEqual(result, expected)

    def test_get_franchise_key_none_cast(self):
        """Test franchise key generation with None cast."""
        mock_movie = MagicMock()
        mock_movie.title = "Movie Title"
        mock_movie.cast = None
        
        candidates = {123: (mock_movie, None)}
        
        # Should handle gracefully
        result = get_franchise_key_robust("Movie Title", candidates)
        self.assertIsInstance(result, str)

    def test_get_franchise_key_missing_name_attribute(self):
        """Test franchise key generation with missing name attributes."""
        # Cast with missing name
        mock_movie = MagicMock()
        mock_movie.title = "Movie Title"
        mock_movie.cast = [{"not_name": "Actor"}]  # Missing 'name' key
        
        candidates = {123: (mock_movie, None)}
        
        result = get_franchise_key_robust("Movie Title", candidates)
        expected = "movie title_unknown"
        self.assertEqual(result, expected)

    def test_extract_base_title_special_characters(self):
        """Test base title extraction with special characters."""
        test_cases = [
            ("Movie: Title & Subtitle", "movie"),
            ("Movie - The Beginning", "movie"),
            ("Movie (Director's Cut)", "movie"),
            ("Movie!!! 2", "movie"),
            ("@Movie# Title$", "@movie# title$"),  # Non-pattern characters preserved
        ]
        
        for input_title, expected in test_cases:
            with self.subTest(input_title=input_title):
                result = extract_base_title_simple(input_title)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()