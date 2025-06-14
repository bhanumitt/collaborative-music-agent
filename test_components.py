#!/usr/bin/env python3
"""
Test script for Collaborative Music Creation Agent components
Run this to verify all components are working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loader():
    """Test the data loader"""
    print("🔍 Testing data loader...")
    try:
        from utils.data_loader import load_sample_data, get_genre_statistics
        
        # Load data
        music_data = load_sample_data()
        print(f"✓ Loaded {len(music_data)} songs")
        
        # Test statistics
        stats = get_genre_statistics()
        print(f"✓ Found {stats['unique_genres']} genres")
        
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_feature_extractor():
    """Test the music feature extractor"""
    print("🔍 Testing feature extractor...")
    try:
        from music_analysis.feature_extractor import MusicFeatureExtractor
        
        extractor = MusicFeatureExtractor()
        
        # Test song analysis
        features = extractor.analyze_song("Take Five", "Dave Brubeck")
        print(f"✓ Analyzed song features: {len(features)} attributes")
        
        # Test similarity calculation
        features2 = extractor.analyze_song("Strobe", "Deadmau5")
        similarity = extractor.calculate_similarity(features, features2)
        print(f"✓ Calculated similarity: {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Feature extractor test failed: {e}")
        return False

def test_playlist_generator():
    """Test the playlist generator"""
    print("🔍 Testing playlist generator...")
    try:
        from recommendation.playlist_generator import PlaylistGenerator
        from utils.data_loader import load_sample_data
        
        music_data = load_sample_data()
        generator = PlaylistGenerator(music_data)
        
        # Test playlist generation
        playlist_result = generator.generate_collaborative_playlist(
            "I love jazz and my friend loves electronic music", 
            playlist_length=5
        )
        
        print(f"✓ Generated playlist with {len(playlist_result['playlist'])} songs")
        print(f"✓ Explanation: {playlist_result['explanation'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Playlist generator test failed: {e}")
        return False

def test_music_creator():
    """Test the music creator"""
    print("🔍 Testing music creator...")
    try:
        from generation.music_creator import MusicCreator
        
        creator = MusicCreator()
        
        # Test music creation
        composition = creator.create_music("blend jazz and electronic styles")
        
        print(f"✓ Created composition in key: {composition['musical_elements']['key']}")
        print(f"✓ Tempo: {composition['musical_elements']['tempo']} BPM")
        print(f"✓ Generated {composition['musical_elements']['melody_notes']} melody notes")
        
        return True
    except Exception as e:
        print(f"❌ Music creator test failed: {e}")
        return False

def test_conversation_manager():
    """Test the conversation manager (without full LLM loading)"""
    print("🔍 Testing conversation manager...")
    try:
        from llm_interface.conversation_manager import ConversationManager
        
        # Initialize without loading the full model for testing
        manager = ConversationManager.__new__(ConversationManager)
        manager.conversation_state = {
            'user_preferences': {},
            'current_task': None,
            'song_examples': [],
            'group_preferences': []
        }
        
        # Test state management
        manager._update_conversation_state(
            "I love jazz music", 
            "Great! Jazz is a wonderful genre."
        )
        
        print("✓ Conversation state management working")
        
        return True
    except Exception as e:
        print(f"❌ Conversation manager test failed: {e}")
        return False

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing critical imports...")
    
    critical_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'), 
        ('librosa', 'librosa'),
        ('music21', 'music21'),
        ('sklearn', 'scikit-learn'),
        ('gradio', 'Gradio'),
        ('numpy', 'NumPy'),
        ('pandas', 'pandas')
    ]
    
    failed_imports = []
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def run_all_tests():
    """Run all component tests"""
    print("🎵 Running Collaborative Music Creation Agent Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loader", test_data_loader),
        ("Feature Extractor", test_feature_extractor),
        ("Playlist Generator", test_playlist_generator),
        ("Music Creator", test_music_creator),
        ("Conversation Manager", test_conversation_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the application")
        print("2. Open http://localhost:7860 in your browser")
        print("3. Start creating collaborative playlists!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)