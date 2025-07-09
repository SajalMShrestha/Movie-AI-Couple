"""
Unit tests for movie search functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from movie_search import (
    generate_search_variations, 
    calculate_title_similarity,
    fuzzy_search_movies
)

class TestMovieSearch(unittest.TestCase):
    
    def test_generate_search_variations(self):
        """Test search variation generation."""
        # Test basic functionality
        variations = generate_search_variations("3 idiots")
        self.assertIn("3 idiots", variations)
        self.assertIn("three idiots", variations)
        
        # Test number conversion
        variations = generate_search_variations("three idiots")
        self.assertIn("3 idiots", variations)
        
        # Test single word handling
        variations = generate_search_variations("inception")
        self.assertIn("inception", variations)
        
        # Test limit on variations
        variations = generate_search_variations("very long movie title")
        self.assertLessEqual(len(variations), 5)

    def test_calculate_title_similarity(self):
        """Test title similarity calculation."""
        # Test exact match
        similarity = calculate_title_similarity("Inception", "Inception")
        self.assertEqual(similarity, 1.0)
        
        # Test case insensitive
        similarity = calculate_title_similarity("inception", "Inception")
        self.assertEqual(similarity, 1.0)
        
        # Test substring matching
        similarity = calculate_title_similarity("Inception", "Inception (2010)")
        self.assertGreater(similarity, 0.9)
        
        # Test "3 idiots" specific case
        similarity = calculate_title_similarity("3 idoits", "3 Idiots")
        self.assertGreater(similarity, 0.8)
        
        # Test typo tolerance
        similarity = calculate_title_similarity("godfater", "The Godfather")
        self.assertGreater(similarity, 0.5)
        
        # Test completely different titles
        similarity = calculate_title_similarity("Inception", "The Godfather")
        self.assertLess(similarity, 0.3)

    def test_title_similarity_edge_cases(self):
        """Test edge cases for title similarity."""
        # Empty strings
        similarity = calculate_title_similarity("", "")
        self.assertGreaterEqual(similarity, 0)
        
        # One empty string
        similarity = calculate_title_similarity("Inception", "")
        self.assertGreaterEqual(similarity, 0)
        
        # Very short strings
        similarity = calculate_title_similarity("It", "It")
        self.assertEqual(similarity, 1.0)
        
        # Numbers and words
        similarity = calculate_title_similarity("2001", "2001: A Space Odyssey")
        self.assertGreater(similarity, 0.5)

    def test_fuzzy_search_variations(self):
        """Test various fuzzy search scenarios."""
        test_cases = [
            ("thre idoits", "3 Idiots"),
            ("godfater", "The Godfather"), 
            ("jurrasic park", "Jurassic Park"),
            ("avengrs", "Avengers"),
            ("intersteler", "Interstellar"),
            ("dark knght", "The Dark Knight"),
            ("iron man", "Iron Man"),
            ("spiderman", "Spider-Man"),
            ("batman begins", "Batman Begins"),
            ("fast furious", "Fast & Furious"),
            ("john wick", "John Wick"),
            ("star wars", "Star Wars"),
            ("matrix", "The Matrix"),
            ("titanic", "Titanic"),
            ("toy story", "Toy Story"),
            ("finding nemo", "Finding Nemo"),
        ]
        
        for query, expected in test_cases:
            with self.subTest(query=query, expected=expected):
                similarity = calculate_title_similarity(query, expected)
                # Should have reasonable similarity for these known cases
                self.assertGreater(similarity, 0.3, 
                    f"'{query}' vs '{expected}' similarity too low: {similarity}")

    @patch('movie_search.requests.get')
    @patch('movie_search.st.secrets')
    def test_fuzzy_search_movies_api_call(self, mock_secrets, mock_get):
        """Test fuzzy search with mocked API calls."""
        # Mock API key
        mock_secrets.__getitem__.return_value = "fake_api_key"
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "3 Idiots",
                    "release_date": "2009-12-25",
                    "id": 20453,
                    "poster_path": "/poster.jpg"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the function
        results = fuzzy_search_movies("3 idiots", max_results=5)
        
        # Verify results
        self.assertIsInstance(results, list)
        if results:  # If we get results
            self.assertIn("title", results[0])
            self.assertIn("id", results[0])
            self.assertIn("similarity", results[0])

    def test_word_based_similarity(self):
        """Test word-based similarity calculations."""
        # Test word overlap
        similarity = calculate_title_similarity("lord rings", "The Lord of the Rings")
        self.assertGreater(similarity, 0.5)
        
        # Test with stop words
        similarity = calculate_title_similarity("lord of rings", "The Lord of the Rings")
        self.assertGreater(similarity, 0.7)
        
        # Test partial word matching
        similarity = calculate_title_similarity("avengrs", "The Avengers")
        self.assertGreater(similarity, 0.5)

    def test_character_level_similarity(self):
        """Test character-level similarity as fallback."""
        # Similar characters
        similarity = calculate_title_similarity("abc", "abd")
        self.assertGreater(similarity, 0.5)
        
        # Very different
        similarity = calculate_title_similarity("abc", "xyz")
        self.assertLess(similarity, 0.5)

if __name__ == '__main__':
    unittest.main()