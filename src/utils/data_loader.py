import json
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def load_sample_data() -> Dict:
    """Load sample music data for the demo"""
    
    # Extended sample music database for demo
    sample_music_data = {
        # Jazz Classics
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
            "time_signature": 4,
            "year": 1959,
            "popularity": 85
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
            "time_signature": 5,
            "year": 1959,
            "popularity": 78
        },
        "Blue Train - John Coltrane": {
            "tempo": 140.0,
            "key": "Bb",
            "mode": "major",
            "danceability": 0.3,
            "energy": 0.7,
            "valence": 0.6,
            "acousticness": 0.85,
            "instrumentalness": 0.9,
            "speechiness": 0.02,
            "liveness": 0.5,
            "loudness": -10.0,
            "genre": "jazz",
            "time_signature": 4,
            "year": 1957,
            "popularity": 72
        },
        "A Love Supreme - John Coltrane": {
            "tempo": 110.0,
            "key": "F",
            "mode": "major",
            "danceability": 0.2,
            "energy": 0.8,
            "valence": 0.8,
            "acousticness": 0.9,
            "instrumentalness": 0.95,
            "speechiness": 0.01,
            "liveness": 0.6,
            "loudness": -8.0,
            "genre": "jazz",
            "time_signature": 4,
            "year": 1965,
            "popularity": 80
        },
        
        # Electronic/EDM
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
            "time_signature": 4,
            "year": 2009,
            "popularity": 88
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
            "time_signature": 4,
            "year": 2012,
            "popularity": 92
        },
        "Levels - Avicii": {
            "tempo": 126.0,
            "key": "C",
            "mode": "major",
            "danceability": 0.85,
            "energy": 0.92,
            "valence": 0.85,
            "acousticness": 0.01,
            "instrumentalness": 0.7,
            "speechiness": 0.08,
            "liveness": 0.2,
            "loudness": -4.0,
            "genre": "electronic",
            "time_signature": 4,
            "year": 2011,
            "popularity": 95
        },
        "One More Time - Daft Punk": {
            "tempo": 123.0,
            "key": "D",
            "mode": "major",
            "danceability": 0.9,
            "energy": 0.85,
            "valence": 0.9,
            "acousticness": 0.0,
            "instrumentalness": 0.3,
            "speechiness": 0.2,
            "liveness": 0.1,
            "loudness": -6.0,
            "genre": "electronic",
            "time_signature": 4,
            "year": 2000,
            "popularity": 90
        },
        
        # Electronic-Jazz Fusion
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
            "time_signature": 4,
            "year": 2010,
            "popularity": 75
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
            "time_signature": 4,
            "year": 1998,
            "popularity": 82
        },
        "Midnight City - M83": {
            "tempo": 105.0,
            "key": "F",
            "mode": "major",
            "danceability": 0.6,
            "energy": 0.7,
            "valence": 0.6,
            "acousticness": 0.1,
            "instrumentalness": 0.4,
            "speechiness": 0.1,
            "liveness": 0.2,
            "loudness": -7.0,
            "genre": "electronic-rock",
            "time_signature": 4,
            "year": 2011,
            "popularity": 85
        },
        
        # Rock
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
            "time_signature": 4,
            "year": 1975,
            "popularity": 96
        },
        "Hotel California - Eagles": {
            "tempo": 75.0,
            "key": "Bm",
            "mode": "minor",
            "danceability": 0.4,
            "energy": 0.6,
            "valence": 0.4,
            "acousticness": 0.3,
            "instrumentalness": 0.1,
            "speechiness": 0.08,
            "liveness": 0.3,
            "loudness": -8.0,
            "genre": "rock",
            "time_signature": 4,
            "year": 1976,
            "popularity": 94
        },
        "Stairway to Heaven - Led Zeppelin": {
            "tempo": 82.0,
            "key": "Am",
            "mode": "minor",
            "danceability": 0.2,
            "energy": 0.7,
            "valence": 0.5,
            "acousticness": 0.4,
            "instrumentalness": 0.3,
            "speechiness": 0.05,
            "liveness": 0.4,
            "loudness": -9.0,
            "genre": "rock",
            "time_signature": 4,
            "year": 1971,
            "popularity": 97
        },
        
        # Pop
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
            "time_signature": 4,
            "year": 2019,
            "popularity": 98
        },
        "Shape of You - Ed Sheeran": {
            "tempo": 96.0,
            "key": "C#",
            "mode": "minor",
            "danceability": 0.825,
            "energy": 0.652,
            "valence": 0.931,
            "acousticness": 0.581,
            "instrumentalness": 0.0,
            "speechiness": 0.0802,
            "liveness": 0.0931,
            "loudness": -3.183,
            "genre": "pop",
            "time_signature": 4,
            "year": 2017,
            "popularity": 95
        },
        "Uptown Funk - Mark Ronson ft. Bruno Mars": {
            "tempo": 115.0,
            "key": "D",
            "mode": "minor",
            "danceability": 0.896,
            "energy": 0.842,
            "valence": 0.957,
            "acousticness": 0.00775,
            "instrumentalness": 0.0,
            "speechiness": 0.198,
            "liveness": 0.0849,
            "loudness": -4.552,
            "genre": "pop-funk",
            "time_signature": 4,
            "year": 2014,
            "popularity": 93
        },
        
        # Hip-Hop
        "Good Kid - Kendrick Lamar": {
            "tempo": 90.0,
            "key": "A",
            "mode": "minor",
            "danceability": 0.6,
            "energy": 0.7,
            "valence": 0.4,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "speechiness": 0.3,
            "liveness": 0.1,
            "loudness": -6.0,
            "genre": "hip-hop",
            "time_signature": 4,
            "year": 2012,
            "popularity": 88
        },
        "HUMBLE. - Kendrick Lamar": {
            "tempo": 150.0,
            "key": "F#",
            "mode": "minor",
            "danceability": 0.7,
            "energy": 0.8,
            "valence": 0.4,
            "acousticness": 0.02,
            "instrumentalness": 0.0,
            "speechiness": 0.25,
            "liveness": 0.1,
            "loudness": -4.0,
            "genre": "hip-hop",
            "time_signature": 4,
            "year": 2017,
            "popularity": 90
        },
        
        # Classical Crossover
        "Canon in D - Pachelbel": {
            "tempo": 100.0,
            "key": "D",
            "mode": "major",
            "danceability": 0.2,
            "energy": 0.3,
            "valence": 0.8,
            "acousticness": 0.95,
            "instrumentalness": 0.98,
            "speechiness": 0.0,
            "liveness": 0.3,
            "loudness": -20.0,
            "genre": "classical",
            "time_signature": 4,
            "year": 1680,
            "popularity": 85
        },
        "Four Seasons: Spring - Vivaldi": {
            "tempo": 120.0,
            "key": "E",
            "mode": "major",
            "danceability": 0.3,
            "energy": 0.6,
            "valence": 0.9,
            "acousticness": 0.98,
            "instrumentalness": 0.99,
            "speechiness": 0.0,
            "liveness": 0.4,
            "loudness": -18.0,
            "genre": "classical",
            "time_signature": 4,
            "year": 1725,
            "popularity": 88
        },
        
        # World Music
        "Bambaataa - Shpongle": {
            "tempo": 95.0,
            "key": "G",
            "mode": "minor",
            "danceability": 0.6,
            "energy": 0.7,
            "valence": 0.6,
            "acousticness": 0.2,
            "instrumentalness": 0.8,
            "speechiness": 0.05,
            "liveness": 0.3,
            "loudness": -8.0,
            "genre": "world-electronic",
            "time_signature": 4,
            "year": 1998,
            "popularity": 70
        },
        "Bom Bom - Sam and the Womp": {
            "tempo": 110.0,
            "key": "C",
            "mode": "major",
            "danceability": 0.8,
            "energy": 0.9,
            "valence": 0.9,
            "acousticness": 0.1,
            "instrumentalness": 0.2,
            "speechiness": 0.15,
            "liveness": 0.4,
            "loudness": -5.0,
            "genre": "world-pop",
            "time_signature": 4,
            "year": 2012,
            "popularity": 75
        },
        
        # Ambient/Chill
        "Weightless - Marconi Union": {
            "tempo": 60.0,
            "key": "C",
            "mode": "major",
            "danceability": 0.1,
            "energy": 0.1,
            "valence": 0.5,
            "acousticness": 0.8,
            "instrumentalness": 0.95,
            "speechiness": 0.0,
            "liveness": 0.1,
            "loudness": -25.0,
            "genre": "ambient",
            "time_signature": 4,
            "year": 2011,
            "popularity": 68
        },
        "An Ending (Ascent) - Brian Eno": {
            "tempo": 70.0,
            "key": "F",
            "mode": "major",
            "danceability": 0.15,
            "energy": 0.2,
            "valence": 0.6,
            "acousticness": 0.9,
            "instrumentalness": 0.98,
            "speechiness": 0.0,
            "liveness": 0.15,
            "loudness": -22.0,
            "genre": "ambient",
            "time_signature": 4,
            "year": 1983,
            "popularity": 72
        },
        
        # Funk
        "Superstition - Stevie Wonder": {
            "tempo": 100.0,
            "key": "Eb",
            "mode": "minor",
            "danceability": 0.85,
            "energy": 0.8,
            "valence": 0.7,
            "acousticness": 0.1,
            "instrumentalness": 0.3,
            "speechiness": 0.1,
            "liveness": 0.3,
            "loudness": -7.0,
            "genre": "funk",
            "time_signature": 4,
            "year": 1972,
            "popularity": 89
        },
        "Give Up the Funk - Parliament": {
            "tempo": 105.0,
            "key": "G",
            "mode": "minor",
            "danceability": 0.9,
            "energy": 0.85,
            "valence": 0.8,
            "acousticness": 0.05,
            "instrumentalness": 0.1,
            "speechiness": 0.2,
            "liveness": 0.4,
            "loudness": -6.0,
            "genre": "funk",
            "time_signature": 4,
            "year": 1975,
            "popularity": 82
        }
    }
    
    logger.info(f"Loaded {len(sample_music_data)} sample songs for demo")
    return sample_music_data


def load_user_profiles() -> List[Dict]:
    """Load sample user profiles for demo purposes"""
    
    user_profiles = [
        {
            "name": "Jazz Enthusiast",
            "favorite_songs": [
                ("Kind of Blue", "Miles Davis"),
                ("Take Five", "Dave Brubeck"),
                ("Blue Train", "John Coltrane")
            ],
            "preferences": {
                "genres": ["jazz", "blues"],
                "energy_level": "medium",
                "preferred_tempo": "moderate",
                "mood": "sophisticated"
            }
        },
        {
            "name": "EDM Lover",
            "favorite_songs": [
                ("Strobe", "Deadmau5"),
                ("Clarity", "Zedd"),
                ("Levels", "Avicii")
            ],
            "preferences": {
                "genres": ["electronic", "house", "techno"],
                "energy_level": "high",
                "preferred_tempo": "fast",
                "mood": "energetic"
            }
        },
        {
            "name": "Rock Fan",
            "favorite_songs": [
                ("Bohemian Rhapsody", "Queen"),
                ("Hotel California", "Eagles"),
                ("Stairway to Heaven", "Led Zeppelin")
            ],
            "preferences": {
                "genres": ["rock", "classic rock"],
                "energy_level": "high",
                "preferred_tempo": "varied",
                "mood": "powerful"
            }
        },
        {
            "name": "Pop Music Fan",
            "favorite_songs": [
                ("Blinding Lights", "The Weeknd"),
                ("Shape of You", "Ed Sheeran"),
                ("Uptown Funk", "Mark Ronson ft. Bruno Mars")
            ],
            "preferences": {
                "genres": ["pop", "mainstream"],
                "energy_level": "medium-high",
                "preferred_tempo": "catchy",
                "mood": "uplifting"
            }
        },
        {
            "name": "Chill Music Lover",
            "favorite_songs": [
                ("Kiara", "Bonobo"),
                ("Teardrop", "Massive Attack"),
                ("Weightless", "Marconi Union")
            ],
            "preferences": {
                "genres": ["ambient", "trip-hop", "electronic-jazz"],
                "energy_level": "low",
                "preferred_tempo": "slow",
                "mood": "relaxed"
            }
        },
        {
            "name": "Genre Explorer",
            "favorite_songs": [
                ("Midnight City", "M83"),
                ("Superstition", "Stevie Wonder"),
                ("One More Time", "Daft Punk")
            ],
            "preferences": {
                "genres": ["varied", "fusion", "experimental"],
                "energy_level": "varied",
                "preferred_tempo": "diverse",
                "mood": "adventurous"
            }
        }
    ]
    
    logger.info(f"Loaded {len(user_profiles)} sample user profiles")
    return user_profiles


def get_genre_statistics() -> Dict:
    """Get statistics about genres in the demo database"""
    music_data = load_sample_data()
    
    genre_counts = {}
    total_songs = len(music_data)
    
    for song_data in music_data.values():
        genre = song_data.get('genre', 'unknown')
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Calculate percentages
    genre_percentages = {
        genre: (count / total_songs) * 100 
        for genre, count in genre_counts.items()
    }
    
    return {
        'total_songs': total_songs,
        'genre_counts': genre_counts,
        'genre_percentages': genre_percentages,
        'unique_genres': len(genre_counts)
    }


def get_audio_feature_ranges() -> Dict:
    """Get the ranges of audio features in the demo database"""
    music_data = load_sample_data()
    
    features = ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 
               'instrumentalness', 'speechiness', 'liveness', 'loudness']
    
    feature_ranges = {}
    
    for feature in features:
        values = [song_data.get(feature, 0) for song_data in music_data.values() if feature in song_data]
        if values:
            feature_ranges[feature] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'range': max(values) - min(values)
            }
    
    return feature_ranges


def search_songs_by_keyword(keyword: str) -> List[Dict]:
    """Search songs by keyword in title or artist"""
    music_data = load_sample_data()
    keyword_lower = keyword.lower()
    
    matching_songs = []
    
    for song_name, song_data in music_data.items():
        if keyword_lower in song_name.lower():
            matching_songs.append({
                'name': song_name,
                **song_data
            })
    
    return matching_songs


def get_songs_by_genre(genre: str) -> List[Dict]:
    """Get all songs of a specific genre"""
    music_data = load_sample_data()
    genre_lower = genre.lower()
    
    genre_songs = []
    
    for song_name, song_data in music_data.items():
        song_genre = song_data.get('genre', '').lower()
        if genre_lower in song_genre or song_genre in genre_lower:
            genre_songs.append({
                'name': song_name,
                **song_data
            })
    
    return genre_songs


def get_songs_by_year_range(start_year: int, end_year: int) -> List[Dict]:
    """Get songs within a specific year range"""
    music_data = load_sample_data()
    
    year_songs = []
    
    for song_name, song_data in music_data.items():
        song_year = song_data.get('year', 0)
        if start_year <= song_year <= end_year:
            year_songs.append({
                'name': song_name,
                **song_data
            })
    
    return sorted(year_songs, key=lambda x: x.get('year', 0))


def save_user_session(session_data: Dict, session_id: str) -> bool:
    """Save user session data (demo implementation)"""
    try:
        # In a real implementation, this would save to a database
        # For demo, we'll just log the action
        logger.info(f"Session {session_id} data would be saved: {len(session_data)} items")
        return True
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {str(e)}")
        return False


def load_user_session(session_id: str) -> Dict:
    """Load user session data (demo implementation)"""
    try:
        # In a real implementation, this would load from a database
        # For demo, return empty session
        logger.info(f"Loading session {session_id} (demo: returning empty session)")
        return {}
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {str(e)}")
        return {}


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    
    # Load and display sample data
    music_data = load_sample_data()
    print(f"Loaded {len(music_data)} songs")
    
    # Show genre statistics
    stats = get_genre_statistics()
    print(f"Genre distribution: {stats['genre_counts']}")
    
    # Show feature ranges
    ranges = get_audio_feature_ranges()
    print(f"Tempo range: {ranges['tempo']['min']:.1f} - {ranges['tempo']['max']:.1f} BPM")
    
    # Test search
    jazz_songs = get_songs_by_genre('jazz')
    print(f"Found {len(jazz_songs)} jazz songs")
    
    # Test user profiles
    profiles = load_user_profiles()
    print(f"Loaded {len(profiles)} user profiles")
    
    print("Data loader test completed successfully!")