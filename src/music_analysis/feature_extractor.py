import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)

class MusicFeatureExtractor:
    def __init__(self):
        """Initialize the music feature extractor"""
        self.sample_rate = 22050
        self.n_mfcc = 13
        
        # Load pre-computed features database
        self.features_db = self._load_features_database()
    
    def _load_features_database(self) -> Dict:
        """Load pre-computed music features for demo songs"""
        # For demo purposes, we'll create a sample database
        # In a real implementation, this would load from a file or API
        
        sample_features = {
            # Jazz examples
            "Kind of Blue - Miles Davis": {
                "tempo": 120.0,
                "key": "D",
                "mode": "major",
                "danceability": 0.4,
                "energy": 0.5,
                "valence": 0.6,
                "acousticness": 0.9,
                "instrumentalness": 0.8,
                "speechiness": 0.05,
                "liveness": 0.3,
                "loudness": -15.0,
                "genre": "jazz",
                "time_signature": 4
            },
            "Take Five - Dave Brubeck": {
                "tempo": 170.0,
                "key": "Eb",
                "mode": "minor",
                "danceability": 0.5,
                "energy": 0.6,
                "valence": 0.7,
                "acousticness": 0.8,
                "instrumentalness": 0.9,
                "speechiness": 0.03,
                "liveness": 0.4,
                "loudness": -12.0,
                "genre": "jazz",
                "time_signature": 5
            },
            
            # EDM examples
            "Strobe - Deadmau5": {
                "tempo": 128.0,
                "key": "C#",
                "mode": "minor",
                "danceability": 0.8,
                "energy": 0.9,
                "valence": 0.6,
                "acousticness": 0.0,
                "instrumentalness": 0.9,
                "speechiness": 0.04,
                "liveness": 0.1,
                "loudness": -8.0,
                "genre": "electronic",
                "time_signature": 4
            },
            "Clarity - Zedd": {
                "tempo": 128.0,
                "key": "G",
                "mode": "major",
                "danceability": 0.9,
                "energy": 0.95,
                "valence": 0.8,
                "acousticness": 0.02,
                "instrumentalness": 0.0,
                "speechiness": 0.15,
                "liveness": 0.15,
                "loudness": -5.0,
                "genre": "electronic",
                "time_signature": 4
            },
            
            # Jazz-Electronic fusion examples
            "Kiara - Bonobo": {
                "tempo": 110.0,
                "key": "Am",
                "mode": "minor",
                "danceability": 0.7,
                "energy": 0.6,
                "valence": 0.5,
                "acousticness": 0.3,
                "instrumentalness": 0.8,
                "speechiness": 0.02,
                "liveness": 0.2,
                "loudness": -10.0,
                "genre": "electronic-jazz",
                "time_signature": 4
            },
            "Teardrop - Massive Attack": {
                "tempo": 73.0,
                "key": "F#",
                "mode": "minor",
                "danceability": 0.5,
                "energy": 0.4,
                "valence": 0.3,
                "acousticness": 0.4,
                "instrumentalness": 0.3,
                "speechiness": 0.25,
                "liveness": 0.15,
                "loudness": -12.0,
                "genre": "trip-hop",
                "time_signature": 4
            },
            
            # Rock examples
            "Bohemian Rhapsody - Queen": {
                "tempo": 144.0,
                "key": "Bb",
                "mode": "major",
                "danceability": 0.3,
                "energy": 0.8,
                "valence": 0.6,
                "acousticness": 0.1,
                "instrumentalness": 0.2,
                "speechiness": 0.1,
                "liveness": 0.4,
                "loudness": -6.0,
                "genre": "rock",
                "time_signature": 4
            },
            
            # Pop examples
            "Blinding Lights - The Weeknd": {
                "tempo": 171.0,
                "key": "F#",
                "mode": "minor",
                "danceability": 0.8,
                "energy": 0.8,
                "valence": 0.7,
                "acousticness": 0.0,
                "instrumentalness": 0.0,
                "speechiness": 0.06,
                "liveness": 0.1,
                "loudness": -5.0,
                "genre": "pop",
                "time_signature": 4
            }
        }
        
        return sample_features
    
    def analyze_song(self, song_name: str, artist: str = "") -> Dict:
        """Analyze a song and return its musical features"""
        song_key = f"{song_name} - {artist}".strip(" -")
        
        # First, check if we have pre-computed features
        if song_key in self.features_db:
            return self.features_db[song_key]
        
        # If not found, try partial matching
        for key in self.features_db.keys():
            if song_name.lower() in key.lower() or (artist and artist.lower() in key.lower()):
                logger.info(f"Found partial match: {key} for query: {song_key}")
                return self.features_db[key]
        
        # If still not found, return a generic analysis
        logger.warning(f"Song not found in database: {song_key}")
        return self._generate_placeholder_features(song_name, artist)
    
    def _generate_placeholder_features(self, song_name: str, artist: str) -> Dict:
        """Generate placeholder features for unknown songs"""
        # Simple heuristics based on song/artist name
        features = {
            "tempo": 120.0,
            "key": "C",
            "mode": "major",
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "acousticness": 0.5,
            "instrumentalness": 0.5,
            "speechiness": 0.1,
            "liveness": 0.2,
            "loudness": -10.0,
            "genre": "unknown",
            "time_signature": 4,
            "note": f"Placeholder features for unknown song: {song_name} - {artist}"
        }
        
        # Adjust based on keywords in song/artist name
        name_lower = f"{song_name} {artist}".lower()
        
        if any(word in name_lower for word in ['dance', 'beat', 'club', 'party']):
            features.update({"danceability": 0.8, "energy": 0.8, "tempo": 128.0})
        
        if any(word in name_lower for word in ['slow', 'ballad', 'soft', 'quiet']):
            features.update({"danceability": 0.3, "energy": 0.3, "tempo": 70.0})
        
        if any(word in name_lower for word in ['rock', 'metal', 'punk']):
            features.update({"energy": 0.9, "loudness": -5.0, "acousticness": 0.1})
        
        if any(word in name_lower for word in ['jazz', 'blues']):
            features.update({"acousticness": 0.8, "instrumentalness": 0.7})
        
        if any(word in name_lower for word in ['electronic', 'techno', 'house', 'edm']):
            features.update({"acousticness": 0.0, "instrumentalness": 0.6, "danceability": 0.8})
        
        return features
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """Extract features from an audio file using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract basic features
            features = {}
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            return {}
    
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two sets of musical features"""
        # Important features for similarity calculation
        important_features = [
            'tempo', 'danceability', 'energy', 'valence', 
            'acousticness', 'instrumentalness', 'loudness'
        ]
        
        similarity_scores = []
        
        for feature in important_features:
            if feature in features1 and feature in features2:
                val1, val2 = features1[feature], features2[feature]
                
                # Normalize tempo to 0-1 scale
                if feature == 'tempo':
                    val1 = min(val1 / 200.0, 1.0)
                    val2 = min(val2 / 200.0, 1.0)
                elif feature == 'loudness':
                    # Convert loudness to 0-1 scale (typical range -30 to 0 dB)
                    val1 = (val1 + 30) / 30.0
                    val2 = (val2 + 30) / 30.0
                
                # Calculate similarity (1 - normalized difference)
                diff = abs(val1 - val2)
                similarity = 1.0 - diff
                similarity_scores.append(similarity)
        
        # Return average similarity
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def get_genre_compatibility(self, genre1: str, genre2: str) -> float:
        """Calculate compatibility between two genres"""
        # Define genre compatibility matrix
        compatibility_matrix = {
            'jazz': {'jazz': 1.0, 'blues': 0.8, 'electronic-jazz': 0.9, 'trip-hop': 0.7, 'rock': 0.4, 'pop': 0.5, 'electronic': 0.6},
            'electronic': {'electronic': 1.0, 'pop': 0.8, 'electronic-jazz': 0.8, 'trip-hop': 0.9, 'rock': 0.6, 'jazz': 0.6, 'blues': 0.4},
            'rock': {'rock': 1.0, 'pop': 0.7, 'blues': 0.6, 'jazz': 0.4, 'electronic': 0.6, 'electronic-jazz': 0.5, 'trip-hop': 0.5},
            'pop': {'pop': 1.0, 'rock': 0.7, 'electronic': 0.8, 'jazz': 0.5, 'blues': 0.5, 'electronic-jazz': 0.6, 'trip-hop': 0.6},
            'blues': {'blues': 1.0, 'jazz': 0.8, 'rock': 0.6, 'pop': 0.5, 'electronic': 0.4, 'electronic-jazz': 0.6, 'trip-hop': 0.5},
            'electronic-jazz': {'electronic-jazz': 1.0, 'jazz': 0.9, 'electronic': 0.8, 'trip-hop': 0.8, 'pop': 0.6, 'rock': 0.5, 'blues': 0.6},
            'trip-hop': {'trip-hop': 1.0, 'electronic': 0.9, 'electronic-jazz': 0.8, 'jazz': 0.7, 'pop': 0.6, 'rock': 0.5, 'blues': 0.5}
        }
        
        return compatibility_matrix.get(genre1, {}).get(genre2, 0.5)
    
    def analyze_group_preferences(self, song_list: List[Tuple[str, str]]) -> Dict:
        """Analyze preferences for a group based on their favorite songs"""
        group_features = []
        genres = []
        
        for song_name, artist in song_list:
            features = self.analyze_song(song_name, artist)
            group_features.append(features)
            genres.append(features.get('genre', 'unknown'))
        
        if not group_features:
            return {}
        
        # Calculate average features
        avg_features = {}
        feature_keys = ['tempo', 'danceability', 'energy', 'valence', 
                       'acousticness', 'instrumentalness', 'loudness']
        
        for key in feature_keys:
            values = [f.get(key, 0) for f in group_features if key in f]
            if values:
                avg_features[key] = np.mean(values)
                avg_features[f'{key}_std'] = np.std(values)
        
        # Analyze genre distribution
        genre_counts = {}
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Find most compatible genres
        genre_compatibility = self._calculate_group_genre_compatibility(genres)
        
        return {
            'average_features': avg_features,
            'genre_distribution': genre_counts,
            'genre_compatibility': genre_compatibility,
            'diversity_score': np.mean([avg_features.get(f'{key}_std', 0) for key in feature_keys]),
            'dominant_genres': sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _calculate_group_genre_compatibility(self, genres: List[str]) -> float:
        """Calculate overall genre compatibility for a group"""
        if len(genres) <= 1:
            return 1.0
        
        compatibility_scores = []
        for i in range(len(genres)):
            for j in range(i + 1, len(genres)):
                score = self.get_genre_compatibility(genres[i], genres[j])
                compatibility_scores.append(score)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.5
    
    def find_bridge_songs(self, preferences1: Dict, preferences2: Dict) -> List[str]:
        """Find songs that could bridge different musical preferences"""
        bridge_candidates = []
        
        # Look for songs in our database that could work as bridges
        for song_key, features in self.features_db.items():
            # Calculate how well this song fits both preferences
            sim1 = self.calculate_similarity(features, preferences1.get('average_features', {}))
            sim2 = self.calculate_similarity(features, preferences2.get('average_features', {}))
            
            # A good bridge song should have decent similarity to both
            bridge_score = min(sim1, sim2) * 0.7 + (sim1 + sim2) / 2 * 0.3
            
            if bridge_score > 0.6:  # Threshold for bridge songs
                bridge_candidates.append((song_key, bridge_score))
        
        # Sort by bridge score and return top candidates
        bridge_candidates.sort(key=lambda x: x[1], reverse=True)
        return [song for song, score in bridge_candidates[:5]]
    
    def get_all_songs(self) -> List[Dict]:
        """Get all songs in the database with their features"""
        return [
            {'name': name, **features} 
            for name, features in self.features_db.items()
        ]
    
    def search_songs_by_features(self, target_features: Dict, limit: int = 10) -> List[Tuple[str, float]]:
        """Search for songs similar to target features"""
        song_scores = []
        
        for song_name, features in self.features_db.items():
            similarity = self.calculate_similarity(features, target_features)
            song_scores.append((song_name, similarity))
        
        # Sort by similarity and return top results
        song_scores.sort(key=lambda x: x[1], reverse=True)
        return song_scores[:limit]