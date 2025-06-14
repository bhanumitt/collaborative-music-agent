import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, music_data: Dict):
        """Initialize the playlist generator with music database"""
        self.music_data = music_data
        self.scaler = StandardScaler()
        self._prepare_feature_matrix()
    
    def _prepare_feature_matrix(self):
        """Prepare feature matrix for ML algorithms"""
        if not self.music_data:
            logger.warning("No music data provided")
            return
        
        # Extract features for all songs
        songs = []
        features_list = []
        
        for song_name, features in self.music_data.items():
            songs.append(song_name)
            feature_vector = [
                features.get('tempo', 120) / 200.0,  # Normalize tempo
                features.get('danceability', 0.5),
                features.get('energy', 0.5),
                features.get('valence', 0.5),
                features.get('acousticness', 0.5),
                features.get('instrumentalness', 0.5),
                (features.get('loudness', -10) + 30) / 30.0,  # Normalize loudness
                features.get('speechiness', 0.1)
            ]
            features_list.append(feature_vector)
        
        self.songs = songs
        self.feature_matrix = np.array(features_list)
        
        # Fit scaler
        if len(features_list) > 0:
            self.scaled_features = self.scaler.fit_transform(self.feature_matrix)
        else:
            self.scaled_features = np.array([])
    
    def generate_collaborative_playlist(self, user_preferences: str, playlist_length: int = 10) -> Dict:
        """Generate a collaborative playlist based on user preferences"""
        try:
            # Parse user preferences (simple implementation)
            genres = self._extract_genres_from_text(user_preferences)
            mood = self._extract_mood_from_text(user_preferences)
            
            # Get songs that match the criteria
            candidate_songs = self._filter_songs_by_criteria(genres, mood)
            
            # If we have specific song examples, use them for similarity
            target_features = self._create_target_features(genres, mood)
            
            # Generate playlist using collaborative filtering approach
            playlist = self._select_diverse_playlist(candidate_songs, target_features, playlist_length)
            
            return {
                'playlist': playlist,
                'explanation': self._generate_playlist_explanation(playlist, genres, mood),
                'total_songs': len(playlist),
                'genres_covered': list(set([song.get('genre', 'unknown') for song in playlist])),
                'avg_energy': np.mean([song.get('energy', 0.5) for song in playlist]),
                'avg_danceability': np.mean([song.get('danceability', 0.5) for song in playlist])
            }
        
        except Exception as e:
            logger.error(f"Error generating playlist: {str(e)}")
            return self._generate_fallback_playlist(playlist_length)
    
    def _extract_genres_from_text(self, text: str) -> List[str]:
        """Extract mentioned genres from text"""
        genre_keywords = {
            'jazz': ['jazz', 'bebop', 'swing', 'blues'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'electro'],
            'rock': ['rock', 'metal', 'punk', 'alternative'],
            'pop': ['pop', 'mainstream', 'commercial'],
            'classical': ['classical', 'orchestral', 'symphony'],
            'hip-hop': ['hip-hop', 'rap', 'hip hop'],
            'country': ['country', 'folk', 'bluegrass'],
            'reggae': ['reggae', 'ska', 'dub']
        }
        
        text_lower = text.lower()
        found_genres = []
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_genres.append(genre)
        
        return found_genres if found_genres else ['pop']  # Default to pop if no genres found
    
    def _extract_mood_from_text(self, text: str) -> str:
        """Extract mood/energy from text"""
        mood_keywords = {
            'energetic': ['energetic', 'upbeat', 'high energy', 'party', 'dance', 'pump up'],
            'chill': ['chill', 'relaxed', 'calm', 'mellow', 'smooth', 'laid back'],
            'happy': ['happy', 'joyful', 'cheerful', 'positive', 'uplifting'],
            'melancholy': ['sad', 'melancholy', 'emotional', 'moody', 'dark'],
            'focus': ['study', 'focus', 'concentration', 'work', 'background']
        }
        
        text_lower = text.lower()
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return mood
        
        return 'balanced'  # Default mood
    
    def _create_target_features(self, genres: List[str], mood: str) -> Dict:
        """Create target features based on genres and mood"""
        # Base features for different genres
        genre_profiles = {
            'jazz': {'tempo': 0.6, 'danceability': 0.4, 'energy': 0.5, 'acousticness': 0.8, 'instrumentalness': 0.7},
            'electronic': {'tempo': 0.7, 'danceability': 0.8, 'energy': 0.8, 'acousticness': 0.1, 'instrumentalness': 0.6},
            'rock': {'tempo': 0.7, 'danceability': 0.5, 'energy': 0.8, 'acousticness': 0.2, 'instrumentalness': 0.3},
            'pop': {'tempo': 0.6, 'danceability': 0.7, 'energy': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.1},
            'classical': {'tempo': 0.5, 'danceability': 0.2, 'energy': 0.4, 'acousticness': 0.9, 'instrumentalness': 0.9}
        }
        
        # Mood adjustments
        mood_adjustments = {
            'energetic': {'energy': 0.3, 'danceability': 0.2, 'tempo': 0.2},
            'chill': {'energy': -0.3, 'danceability': -0.2, 'tempo': -0.2, 'valence': 0.1},
            'happy': {'valence': 0.3, 'energy': 0.1},
            'melancholy': {'valence': -0.3, 'energy': -0.1},
            'focus': {'energy': -0.1, 'speechiness': -0.1, 'instrumentalness': 0.2}
        }
        
        # Combine genre preferences
        if genres:
            target = {}
            for feature in ['tempo', 'danceability', 'energy', 'acousticness', 'instrumentalness']:
                values = [genre_profiles.get(genre, {}).get(feature, 0.5) for genre in genres if genre in genre_profiles]
                target[feature] = np.mean(values) if values else 0.5
        else:
            target = {'tempo': 0.5, 'danceability': 0.5, 'energy': 0.5, 'acousticness': 0.5, 'instrumentalness': 0.5}
        
        # Apply mood adjustments
        if mood in mood_adjustments:
            for feature, adjustment in mood_adjustments[mood].items():
                target[feature] = np.clip(target.get(feature, 0.5) + adjustment, 0.0, 1.0)
        
        # Add other default features
        target.update({
            'valence': target.get('valence', 0.6),
            'loudness': 0.7,
            'speechiness': 0.1
        })
        
        return target
    
    def _filter_songs_by_criteria(self, genres: List[str], mood: str) -> List[Dict]:
        """Filter songs that match the given criteria"""
        candidates = []
        
        for song_name, features in self.music_data.items():
            song_genre = features.get('genre', 'unknown')
            
            # Check genre match
            genre_match = not genres or song_genre in genres or any(g in song_genre for g in genres)
            
            # Check mood compatibility
            mood_match = self._check_mood_compatibility(features, mood)
            
            if genre_match or mood_match:  # Include if either criteria matches
                song_dict = {'name': song_name, **features}
                candidates.append(song_dict)
        
        return candidates if candidates else list({'name': name, **features} for name, features in self.music_data.items())
    
    def _check_mood_compatibility(self, features: Dict, mood: str) -> bool:
        """Check if song features match the desired mood"""
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        danceability = features.get('danceability', 0.5)
        
        if mood == 'energetic':
            return energy > 0.6 and danceability > 0.6
        elif mood == 'chill':
            return energy < 0.6 and features.get('acousticness', 0.5) > 0.3
        elif mood == 'happy':
            return valence > 0.6
        elif mood == 'melancholy':
            return valence < 0.4
        elif mood == 'focus':
            return features.get('speechiness', 0.1) < 0.1 and features.get('instrumentalness', 0.5) > 0.3
        
        return True  # Default: all songs are compatible
    
    def _select_diverse_playlist(self, candidates: List[Dict], target_features: Dict, length: int) -> List[Dict]:
        """Select a diverse playlist from candidates"""
        if not candidates:
            return []
        
        if len(candidates) <= length:
            return candidates
        
        # Calculate similarity scores to target
        scored_songs = []
        for song in candidates:
            similarity = self._calculate_feature_similarity(song, target_features)
            scored_songs.append((song, similarity))
        
        # Sort by similarity
        scored_songs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top songs while ensuring diversity
        selected = []
        selected_features = []
        
        for song, score in scored_songs:
            if len(selected) >= length:
                break
            
            # Check diversity
            if not selected_features or self._is_diverse_enough(song, selected_features):
                selected.append(song)
                selected_features.append(self._extract_feature_vector(song))
        
        # If we don't have enough diverse songs, fill with highest scoring remaining
        while len(selected) < length and len(selected) < len(scored_songs):
            for song, score in scored_songs:
                if song not in selected:
                    selected.append(song)
                    break
        
        return selected
    
    def _calculate_feature_similarity(self, song: Dict, target_features: Dict) -> float:
        """Calculate similarity between song and target features"""
        similarities = []
        
        for feature in ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']:
            if feature in target_features:
                song_val = song.get(feature, 0.5)
                target_val = target_features[feature]
                
                # Normalize tempo
                if feature == 'tempo':
                    song_val = min(song_val / 200.0, 1.0)
                
                similarity = 1.0 - abs(song_val - target_val)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _extract_feature_vector(self, song: Dict) -> List[float]:
        """Extract feature vector from song"""
        return [
            song.get('tempo', 120) / 200.0,
            song.get('danceability', 0.5),
            song.get('energy', 0.5),
            song.get('valence', 0.5),
            song.get('acousticness', 0.5),
            song.get('instrumentalness', 0.5)
        ]
    
    def _is_diverse_enough(self, new_song: Dict, existing_features: List[List[float]], threshold: float = 0.8) -> bool:
        """Check if new song is diverse enough from existing selections"""
        new_features = self._extract_feature_vector(new_song)
        
        for existing in existing_features:
            similarity = cosine_similarity([new_features], [existing])[0][0]
            if similarity > threshold:
                return False
        
        return True
    
    def _generate_playlist_explanation(self, playlist: List[Dict], genres: List[str], mood: str) -> str:
        """Generate explanation for the playlist choices"""
        if not playlist:
            return "I couldn't generate a playlist with the given preferences."
        
        explanation_parts = []
        
        # Overall approach
        if genres and mood:
            explanation_parts.append(f"I created a {mood} playlist blending {', '.join(genres)} styles.")
        elif genres:
            explanation_parts.append(f"I focused on {', '.join(genres)} music for this playlist.")
        elif mood:
            explanation_parts.append(f"I curated a {mood} playlist across different genres.")
        
        # Musical characteristics
        avg_energy = np.mean([song.get('energy', 0.5) for song in playlist])
        avg_danceability = np.mean([song.get('danceability', 0.5) for song in playlist])
        
        energy_desc = "high-energy" if avg_energy > 0.7 else "mellow" if avg_energy < 0.4 else "balanced-energy"
        dance_desc = "danceable" if avg_danceability > 0.7 else "contemplative" if avg_danceability < 0.4 else "moderately rhythmic"
        
        explanation_parts.append(f"The playlist features {energy_desc}, {dance_desc} tracks.")
        
        # Highlight diversity
        genres_in_playlist = set([song.get('genre', 'unknown') for song in playlist])
        if len(genres_in_playlist) > 1:
            explanation_parts.append(f"I included variety across {len(genres_in_playlist)} genres to keep it interesting.")
        
        return " ".join(explanation_parts)
    
    def _generate_fallback_playlist(self, length: int) -> Dict:
        """Generate a fallback playlist when main algorithm fails"""
        # Select random diverse songs
        all_songs = list({'name': name, **features} for name, features in self.music_data.items())
        selected = random.sample(all_songs, min(length, len(all_songs)))
        
        return {
            'playlist': selected,
            'explanation': "I've created a diverse mix of popular songs across different genres.",
            'total_songs': len(selected),
            'genres_covered': list(set([song.get('genre', 'unknown') for song in selected])),
            'avg_energy': np.mean([song.get('energy', 0.5) for song in selected]),
            'avg_danceability': np.mean([song.get('danceability', 0.5) for song in selected])
        }
    
    def create_transition_playlist(self, start_song: Dict, end_song: Dict, steps: int = 5) -> List[Dict]:
        """Create a playlist that transitions smoothly from one song to another"""
        transition_songs = []
        
        # Calculate feature differences
        start_features = self._extract_feature_vector(start_song)
        end_features = self._extract_feature_vector(end_song)
        
        # Create intermediate target features
        for i in range(1, steps + 1):
            ratio = i / (steps + 1)
            intermediate_features = [
                start_val + (end_val - start_val) * ratio
                for start_val, end_val in zip(start_features, end_features)
            ]
            
            # Find songs closest to intermediate features
            target_dict = {
                'tempo': intermediate_features[0] * 200,
                'danceability': intermediate_features[1],
                'energy': intermediate_features[2],
                'valence': intermediate_features[3],
                'acousticness': intermediate_features[4],
                'instrumentalness': intermediate_features[5]
            }
            
            best_match = None
            best_similarity = -1
            
            for song_name, features in self.music_data.items():
                similarity = self._calculate_feature_similarity({'name': song_name, **features}, target_dict)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {'name': song_name, **features}
            
            if best_match:
                transition_songs.append(best_match)
        
        return transition_songs