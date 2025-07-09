"""
Unit tests for narrative analysis functionality.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from narrative_analysis import (
    infer_tone, 
    infer_narrative_complexity,
    infer_genre_indicator,
    infer_setting_context,
    infer_narrative_style,
    infer_mood_from_plot,
    compute_narrative_similarity
)

class TestNarrativeAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.happy_plot = "A young man finds love and happiness in a beautiful small town. Everyone laughs and celebrates together in this heartwarming comedy."
        self.sad_plot = "The protagonist faces tragedy and loss as war destroys everything he holds dear. Death and suffering pervade this dark drama."
        self.neutral_plot = "A detective investigates a case in the city. He follows clues and interviews witnesses to solve the mystery."
        self.complex_plot = "An epistemological investigation into the phenomenological aspects of consciousness reveals the metaphysical implications of quantum mechanical interpretations."
        self.simple_plot = "John goes to the store. He buys milk. Then he goes home."
        self.action_plot = "The hero fights enemies in an epic battle. He escapes from explosions and saves the day through war and combat."
        self.romance_plot = "Two people fall in love despite obstacles. Their relationship grows through friendship and emotional connection."
        self.fantasy_plot = "In a magical kingdom far away, dragons and wizards battle in space using supernatural powers in an alternate universe."
        self.realistic_plot = "Set in modern day Chicago, a suburban family deals with everyday life in their ordinary town neighborhood."

    def test_infer_tone(self):
        """Test tone inference from plot text."""
        # Test positive tone
        tone = infer_tone(self.happy_plot)
        self.assertEqual(tone, "positive")
        
        # Test negative tone
        tone = infer_tone(self.sad_plot)
        self.assertEqual(tone, "negative")
        
        # Test neutral tone
        tone = infer_tone(self.neutral_plot)
        self.assertEqual(tone, "neutral")
        
        # Test empty plot
        tone = infer_tone("")
        self.assertIn(tone, ["positive", "negative", "neutral"])

    def test_infer_narrative_complexity(self):
        """Test narrative complexity assessment."""
        # Test complex narrative
        complexity = infer_narrative_complexity(self.complex_plot)
        self.assertEqual(complexity, "complex")
        
        # Test simple narrative
        complexity = infer_narrative_complexity(self.simple_plot)
        self.assertEqual(complexity, "simple")
        
        # Test empty plot
        complexity = infer_narrative_complexity("")
        self.assertIn(complexity, ["complex", "simple"])

    def test_infer_genre_indicator(self):
        """Test genre indicator classification."""
        # Test action-oriented
        genre = infer_genre_indicator(self.action_plot)
        self.assertEqual(genre, "action-oriented")
        
        # Test emotion/character-oriented
        genre = infer_genre_indicator(self.romance_plot)
        self.assertEqual(genre, "emotion/character-oriented")
        
        # Test general (no specific keywords)
        genre = infer_genre_indicator(self.neutral_plot)
        self.assertEqual(genre, "general")

    def test_infer_setting_context(self):
        """Test setting context classification."""
        # Test fantastical/surreal
        setting = infer_setting_context(self.fantasy_plot)
        self.assertEqual(setting, "fantastical/surreal")
        
        # Test realistic
        setting = infer_setting_context(self.realistic_plot)
        self.assertEqual(setting, "realistic")
        
        # Test neutral
        setting = infer_setting_context(self.neutral_plot)
        self.assertEqual(setting, "neutral")

    def test_infer_narrative_style(self):
        """Test comprehensive narrative style analysis."""
        # Test with valid plot
        style = infer_narrative_style(self.happy_plot)
        self.assertIsInstance(style, dict)
        self.assertIn("tone", style)
        self.assertIn("complexity", style)
        self.assertIn("genre_indicator", style)
        self.assertIn("setting_context", style)
        
        # Test with empty plot
        style = infer_narrative_style("")
        self.assertEqual(style["tone"], "neutral")
        self.assertEqual(style["complexity"], "simple")
        self.assertEqual(style["genre_indicator"], "general")
        self.assertEqual(style["setting_context"], "neutral")

    def test_infer_mood_from_plot(self):
        """Test mood inference from plot."""
        # Test feel-good mood
        mood = infer_mood_from_plot(self.happy_plot)
        self.assertEqual(mood, "feel_good")
        
        # Test melancholic mood
        mood = infer_mood_from_plot(self.sad_plot)
        self.assertEqual(mood, "melancholic")
        
        # Test cerebral mood
        mood = infer_mood_from_plot(self.neutral_plot)
        self.assertEqual(mood, "cerebral")

    def test_compute_narrative_similarity(self):
        """Test narrative similarity computation."""
        candidate_style = {
            "tone": "positive",
            "complexity": "simple",
            "genre_indicator": "action-oriented",
            "setting_context": "realistic"
        }
        
        # Perfect match
        reference_styles = {
            "tone": ["positive", "positive", "positive"],
            "complexity": ["simple", "simple", "simple"],
            "genre_indicator": ["action-oriented", "action-oriented", "action-oriented"],
            "setting_context": ["realistic", "realistic", "realistic"]
        }
        
        similarity = compute_narrative_similarity(candidate_style, reference_styles)
        self.assertEqual(similarity, 1.0)
        
        # No match
        reference_styles = {
            "tone": ["negative", "negative", "negative"],
            "complexity": ["complex", "complex", "complex"],
            "genre_indicator": ["emotion/character-oriented", "emotion/character-oriented", "emotion/character-oriented"],
            "setting_context": ["fantastical/surreal", "fantastical/surreal", "fantastical/surreal"]
        }
        
        similarity = compute_narrative_similarity(candidate_style, reference_styles)
        self.assertEqual(similarity, 0.0)
        
        # Partial match
        reference_styles = {
            "tone": ["positive", "negative", "neutral"],
            "complexity": ["simple", "complex", "simple"],
            "genre_indicator": ["general", "general", "general"],
            "setting_context": ["neutral", "neutral", "neutral"]
        }
        
        similarity = compute_narrative_similarity(candidate_style, reference_styles)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)

    def test_keyword_detection(self):
        """Test that keywords are properly detected."""
        # Action keywords
        action_plot = "The hero fights in battle and escapes from the mission"
        genre = infer_genre_indicator(action_plot)
        self.assertEqual(genre, "action-oriented")
        
        # Emotion keywords
        emotion_plot = "A story of love and friendship overcoming betrayal"
        genre = infer_genre_indicator(emotion_plot)
        self.assertEqual(genre, "emotion/character-oriented")
        
        # Concept keywords
        concept_plot = "An exploration of reality and consciousness through dreams"
        genre = infer_genre_indicator(concept_plot)
        self.assertEqual(genre, "idea/concept-oriented")
        
        # Fantasy keywords
        fantasy_plot = "In a magical kingdom with dragons in space"
        setting = infer_setting_context(fantasy_plot)
        self.assertEqual(setting, "fantastical/surreal")
        
        # Realistic keywords
        realistic_plot = "Set in a modern city suburb with everyday people"
        setting = infer_setting_context(realistic_plot)
        self.assertEqual(setting, "realistic")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very short text
        style = infer_narrative_style("Hi")
        self.assertIsInstance(style, dict)
        
        # Only punctuation
        style = infer_narrative_style("!@#$%")
        self.assertIsInstance(style, dict)
        
        # Very long text (shouldn't break)
        long_text = "word " * 1000
        style = infer_narrative_style(long_text)
        self.assertIsInstance(style, dict)

if __name__ == '__main__':
    unittest.main()