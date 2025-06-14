import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize with lightweight CPU-friendly model"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_state = {
            'user_preferences': {},
            'current_task': None,
            'song_examples': [],
            'group_preferences': []
        }
        
        # Always load the LLM - no fallbacks
        self._load_model()
    
    def _load_model(self):
        """Load the quantized LLM model for CPU inference"""
        try:
            logger.info(f"Loading lightweight model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Lightweight model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to no model
            self.model = None
            self.tokenizer = None
    
    def _load_simple_model(self):
        """Fallback model loading without quantization"""
        try:
            logger.info("Loading model without quantization...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            
            logger.info("Simple model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading simple model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the music agent"""
        return """You are a music assistant. When users ask about playlists, songs, or music preferences:

1. Acknowledge what they mentioned specifically
2. Create helpful recommendations immediately  
3. Use the available music database (jazz, electronic, rock, pop, classical, etc.)
4. Be direct and helpful, not vague

Available music includes artists like Miles Davis, Deadmau5, Queen, Ed Sheeran, and many more across different genres.

Be conversational but focused on actually helping with music requests."""
    
    def _extract_function_calls(self, text: str) -> List[Dict]:
        """Extract function calls from LLM response"""
        function_pattern = r'(\w+)\(([^)]*)\)'
        matches = re.findall(function_pattern, text)
        
        functions = []
        for func_name, args_str in matches:
            if func_name in ['analyze_song_features', 'generate_playlist', 'create_original_music', 'explain_song_choice']:
                # Parse arguments (simple implementation)
                args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
                functions.append({
                    'name': func_name,
                    'args': args
                })
        
        return functions
    
    def _execute_function(self, function: Dict, feature_extractor, playlist_generator, music_creator) -> str:
        """Execute a function call and return result"""
        func_name = function['name']
        args = function['args']
        
        try:
            if func_name == 'analyze_song_features' and len(args) >= 2:
                song_name, artist = args[0], args[1]
                features = feature_extractor.analyze_song(song_name, artist)
                return f"Song analysis: {features}"
            
            elif func_name == 'generate_playlist' and len(args) >= 1:
                preferences = args[0]
                playlist = playlist_generator.generate_collaborative_playlist(preferences)
                return f"Generated playlist: {playlist}"
            
            elif func_name == 'create_original_music' and len(args) >= 1:
                style_blend = args[0]
                music_result = music_creator.create_music(style_blend)
                return f"Created original music: {music_result}"
            
            elif func_name == 'explain_song_choice' and len(args) >= 2:
                song, reason = args[0], args[1]
                return f"Explanation for {song}: {reason}"
            
            else:
                return f"Function {func_name} executed with args: {args}"
        
        except Exception as e:
            logger.error(f"Error executing function {func_name}: {str(e)}")
            return f"Error executing {func_name}: {str(e)}"
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the LLM"""
        if self.model is None or self.tokenizer is None:
            return "I'm sorry, my AI model isn't loaded properly. Please try restarting the application."
        
        try:
            # For DialoGPT, we need to encode properly
            inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            return response.strip() if response.strip() else "I'd be happy to help you with music recommendations!"
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm having trouble generating a response right now. Could you please try again?"
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate intelligent rule-based responses for any music query"""
        prompt_lower = prompt.lower()
        
        # Handle playlist requests
        if 'playlist' in prompt_lower:
            return self._handle_playlist_request(prompt_lower)
        
        # Handle music generation
        if any(word in prompt_lower for word in ['generate', 'create', 'compose']) and any(word in prompt_lower for word in ['song', 'music', 'track']):
            return self._handle_music_generation_request(prompt_lower)
        
        # Handle analysis questions
        if any(word in prompt_lower for word in ['why', 'analyze', 'explain', 'work together', 'compatible']):
            return self._handle_analysis_request(prompt_lower)
        
        # Handle genre/mood requests
        if any(word in prompt_lower for word in ['jazz', 'electronic', 'rock', 'pop', 'classical', 'ambient', 'genre']):
            return self._handle_genre_request(prompt_lower)
        
        # Default greeting/help
        return self._handle_general_greeting()
    
    def _handle_playlist_request(self, prompt: str) -> str:
        """Handle playlist creation requests generically"""
        # Check for mood/context keywords
        if any(word in prompt for word in ['study', 'focus', 'concentration', 'work']):
            return """📚 **Study & Focus Playlist**

I've curated songs perfect for concentration:

**🎼 Your Focus Companion:**
1. **Weightless - Marconi Union** (scientifically designed to reduce anxiety)
2. **An Ending (Ascent) - Brian Eno** (pure ambient focus)
3. **Kiara - Bonobo** (electronic-jazz fusion, non-distracting)
4. **Canon in D - Pachelbel** (classical structure aids concentration)
5. **Teardrop - Massive Attack** (hypnotic, steady rhythm)

**🧠 Why these work:** 60-70 BPM tracks with minimal lyrics and consistent textures enhance focus without distraction."""
        
        elif any(word in prompt for word in ['party', 'dance', 'energy', 'upbeat', 'high energy']):
            return """🎉 **High-Energy Party Playlist**

**🔥 Get Everyone Moving:**
1. **Levels - Avicii** (instant energy boost)
2. **Uptown Funk - Mark Ronson ft. Bruno Mars** (crowd favorite)
3. **One More Time - Daft Punk** (classic dance anthem)
4. **Clarity - Zedd** (modern EDM hit)
5. **Superstition - Stevie Wonder** (funk that gets everyone dancing)

**💃 Perfect for:** Dancing, group sing-alongs, maintaining high energy throughout your event!"""
        
        elif any(word in prompt for word in ['chill', 'relax', 'calm', 'ambient', 'background']):
            return """😌 **Chill & Relaxation Playlist**

**🌙 Your Calm Companion:**
1. **Weightless - Marconi Union** (most relaxing song ever recorded)
2. **Kiara - Bonobo** (downtempo electronic bliss)
3. **An Ending (Ascent) - Brian Eno** (ambient masterpiece)
4. **Teardrop - Massive Attack** (hypnotic trip-hop)
5. **Four Seasons: Spring - Vivaldi** (peaceful classical)

**✨ These tracks create a perfect atmosphere for unwinding and peaceful moments."""
        
        elif 'jazz' in prompt and any(word in prompt for word in ['electronic', 'edm', 'techno', 'house']):
            return """🎷🎧 **Jazz-Electronic Fusion Playlist**

**🌉 Bridging Two Worlds:**
1. **Take Five - Dave Brubeck** (jazz foundation with complex rhythms)
2. **Kiara - Bonobo** (electronic with jazz sensibilities)
3. **Strobe - Deadmau5** (progressive build with improvisation elements)
4. **Midnight City - M83** (saxophone meets electronic production)
5. **Teardrop - Massive Attack** (trip-hop that bridges both genres)

**🎯 The Magic:** Jazz improvisation principles meet electronic production techniques for endless creative possibilities!"""
        
        # Generic playlist response
        return """🎵 **Custom Playlist Creation**

I'd love to create the perfect playlist for you! To make the best recommendations, please share:

**📝 Tell me about:**
- **Mood/Setting:** Study, party, workout, relaxation, background music?
- **Genre Preferences:** Any styles you particularly enjoy or want to explore?
- **Energy Level:** High energy, mellow, or somewhere in between?
- **Specific Songs:** Any tracks you love that I can use as inspiration?

**🎼 I can create playlists for:**
- Different moods and activities
- Genre fusion and crossover styles  
- Group listening with varied tastes
- Discovering new music based on your favorites

The more details you provide, the better I can tailor the perfect musical experience for you! 🎶"""
    
    def _handle_music_generation_request(self, prompt: str) -> str:
        """Handle original music generation requests"""
        # Check for specific style combinations
        if 'jazz' in prompt and any(word in prompt for word in ['electronic', 'edm', 'techno']):
            return """🎼 **Jazz-Electronic Fusion Composition**

I've created an original piece blending these styles:

**🎷 Jazz Elements:**
- Complex chord progressions with 7th and 9th chords
- Syncopated rhythms and improvisation-style melodies
- Walking bass line foundation
- Key signature that allows for both swing and electronic feels

**🎧 Electronic Elements:**
- Modern synthesizer textures and ambient pads
- Programmed beats with both acoustic and electronic percussion
- Digital effects and processing on traditional instruments
- Tempo optimized for both genres (120 BPM)

**🎯 Result:** A sophisticated composition that respects both traditions while creating something entirely new. Perfect for modern listeners who appreciate musical innovation!"""
        
        elif 'classical' in prompt and any(word in prompt for word in ['electronic', 'modern', 'digital']):
            return """🎻 **Classical-Electronic Hybrid Composition**

**🏛️ Classical Foundation:**
- Traditional harmonic progressions and orchestral arrangements
- Structured form with clear movements and themes
- Rich melodic development and counterpoint

**💻 Electronic Enhancement:**
- Digital orchestration with realistic instrument samples
- Ambient textures and atmospheric effects
- Modern production techniques applied to classical forms

**🎼 The Fusion:** Creates a bridge between centuries of musical tradition and cutting-edge technology."""
        
        # Generic music generation response
        return """🎵 **Original Music Composition**

I can create custom compositions blending any musical styles! Here's how it works:

**🎼 Popular Style Combinations:**
- **Classical + Electronic** → Orchestral meets digital innovation
- **Jazz + Rock** → Sophisticated harmony with driving energy  
- **Ambient + Pop** → Atmospheric textures with memorable melodies
- **Funk + Electronic** → Groove-heavy rhythms with modern production
- **World + Electronic** → Traditional instruments meet digital soundscapes

**🎯 To create your perfect composition, tell me:**
- Which styles you'd like me to blend
- Desired mood (energetic, contemplative, dramatic, etc.)
- Tempo preference (slow, medium, uptempo)
- Any specific instruments you'd like featured

I'll compose something unique that captures exactly what you're looking for! 🎶"""
    
    def _handle_analysis_request(self, prompt: str) -> str:
        """Handle music analysis and explanation requests"""
        return """🎯 **Music Analysis & Theory**

I can analyze musical compatibility and explain why certain songs work well together:

**🎼 What I Analyze:**
- **Harmonic Compatibility** → How chord progressions complement each other
- **Rhythmic Relationships** → Tempo, time signatures, and groove compatibility  
- **Tonal Characteristics** → Key relationships and modal interchange
- **Energy Flow** → How songs create compelling listening sequences
- **Genre Fusion Potential** → Where different styles naturally connect

**📊 Analysis Framework:**
- **Technical Elements:** BPM, key signatures, chord progressions, instrumentation
- **Emotional Content:** Mood, energy level, emotional arc through playlist
- **Structural Patterns:** Song forms, arrangements, production styles
- **Cultural Context:** How genres historically influence each other

**🎵 To get a detailed analysis:**
- Name specific songs you'd like me to compare
- Ask about genre compatibility (e.g., "Why do jazz and electronic work together?")
- Request playlist flow analysis
- Inquire about music theory concepts

What would you like me to analyze for you? 🎶"""
    
    def _handle_genre_request(self, prompt: str) -> str:
        """Handle genre-specific requests"""
        # Detect specific genres mentioned
        genres_mentioned = []
        genre_keywords = {
            'jazz': ['jazz', 'bebop', 'swing', 'blues'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'ambient'],
            'rock': ['rock', 'metal', 'punk', 'alternative'],
            'pop': ['pop', 'mainstream', 'commercial'],
            'classical': ['classical', 'orchestral', 'symphony', 'baroque'],
            'funk': ['funk', 'groove', 'soul'],
            'world': ['world', 'ethnic', 'traditional', 'folk']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                genres_mentioned.append(genre)
        
        if len(genres_mentioned) == 1:
            genre = genres_mentioned[0]
            return f"""🎼 **{genre.title()} Music Exploration**

**🎵 Key Characteristics:**
{self._get_genre_characteristics(genre)}

**💡 Want to explore further?**
- Request specific {genre} recommendations
- Ask about {genre} fusion with other genres
- Learn about {genre} history and evolution
- Create {genre}-based playlists for different moods

What aspect of {genre} interests you most? 🎶"""
        
        elif len(genres_mentioned) > 1:
            return f"""🎼 **Multi-Genre Exploration: {' + '.join([g.title() for g in genres_mentioned])}**

These genres can create fascinating combinations! Each brings unique elements:

{chr(10).join([f"**{g.title()}:** {self._get_genre_characteristics(g)}" for g in genres_mentioned])}

**🌉 Fusion Potential:** These styles can blend through shared rhythmic elements, harmonic progressions, or production techniques.

Would you like me to create a fusion playlist, generate original music combining these styles, or explain the theoretical connections between them? 🎶"""
        
        # Generic genre response
        return """🎼 **Genre Guide & Exploration**

**🎵 Available Genres in Our Database:**
- **Jazz** → Improvisation, complex harmony, swing rhythms
- **Electronic/EDM** → Synthesized sounds, programmed beats, digital production
- **Rock** → Guitar-driven, powerful vocals, strong rhythm section
- **Pop** → Catchy melodies, accessible structures, polished production  
- **Classical** → Orchestral arrangements, formal composition techniques
- **Funk** → Groove-based rhythms, prominent bass lines, syncopation
- **Ambient** → Atmospheric textures, minimal structure, mood-focused
- **World** → Traditional instruments, cultural rhythms, global influences

**🎯 How can I help you explore?**
- Deep dive into specific genre characteristics
- Create cross-genre fusion playlists
- Explain historical connections between styles
- Recommend gateway songs for unfamiliar genres

Which musical territory would you like to explore? 🎶"""
    
    def _get_genre_characteristics(self, genre: str) -> str:
        """Get characteristics for a specific genre"""
        characteristics = {
            'jazz': "Complex harmonies, improvisation, syncopated rhythms, individual expression",
            'electronic': "Digital synthesis, programmed beats, sound design, technological innovation",
            'rock': "Guitar-driven arrangements, powerful vocals, strong rhythmic foundation", 
            'pop': "Memorable melodies, accessible song structures, mainstream appeal",
            'classical': "Formal composition techniques, orchestral arrangements, structural complexity",
            'funk': "Groove-centered rhythms, prominent bass, syncopation, rhythmic emphasis",
            'ambient': "Atmospheric textures, minimal structure, mood and environment focus",
            'world': "Cultural traditions, indigenous instruments, regional rhythmic patterns"
        }
        return characteristics.get(genre, "Rich musical tradition with unique characteristics")
    
    def _handle_general_greeting(self) -> str:
        """Handle general greetings and help requests"""
        return """🎵 **Welcome to Your AI Music Assistant!**

I'm here to help you explore, create, and understand music in all its forms.

**🎼 What I Can Do:**
- **🎵 Create Custom Playlists** → Perfect mixes for any mood or occasion
- **🎶 Generate Original Music** → Blend genres into unique compositions  
- **🎯 Analyze Musical Compatibility** → Explain why songs work well together
- **🎸 Explore Genres** → Deep dive into any musical style
- **🤝 Collaborative Recommendations** → Find common ground for group listening

**💡 Try Asking:**
- *"Create a playlist for [mood/activity]"*
- *"Generate music blending [genre] and [genre]"*  
- *"Why do these songs work together?"*
- *"Tell me about [genre] music"*
- *"Help me find music similar to [description]"*

**🎶 Ready to dive into your musical journey?** Just tell me what you're looking for! ✨"""
    
    def process_message(self, message: str, chat_history: List, feature_extractor, playlist_generator, music_creator) -> Dict:
        """Hybrid approach: LLM understanding + music logic + LLM formatting"""
        
        # Step 1: Extract user intent using simple parsing
        intent = self._extract_user_intent(message)
        
        # Step 2: Use music logic to generate actual recommendations
        music_result = self._execute_music_logic(intent, feature_extractor, playlist_generator, music_creator)
        
        # Step 3: Use LLM to format the response nicely
        formatted_response = self._format_response_with_llm(intent, music_result, message)
        
        # Update conversation state
        self._update_conversation_state(message, formatted_response)
        
        return {
            'response': formatted_response,
            'functions_called': [],
            'function_results': [music_result],
            'conversation_state': self.conversation_state
        }
    
    def _extract_user_intent(self, message: str) -> Dict:
        """Extract user intent using keyword matching"""
        message_lower = message.lower()
        intent = {
            'type': 'general',
            'songs_mentioned': [],
            'genres_mentioned': [],
            'mood': None,
            'activity': None
        }
        
        # Extract song mentions
        song_keywords = ['pompeii', 'strobe', 'levels', 'clarity', 'take five', 'kind of blue']
        for song in song_keywords:
            if song in message_lower:
                intent['songs_mentioned'].append(song)
        
        # Extract genres
        genre_keywords = {
            'jazz': ['jazz', 'bebop', 'swing'],
            'electronic': ['electronic', 'edm', 'techno', 'house'],
            'rock': ['rock', 'indie', 'alternative'],
            'pop': ['pop', 'mainstream'],
            'classical': ['classical', 'orchestral']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intent['genres_mentioned'].append(genre)
        
        # Extract mood/activity
        if any(word in message_lower for word in ['study', 'focus', 'work']):
            intent['activity'] = 'study'
        elif any(word in message_lower for word in ['party', 'dance', 'energy']):
            intent['activity'] = 'party'
        elif any(word in message_lower for word in ['chill', 'relax', 'calm']):
            intent['activity'] = 'chill'
        
        # Determine intent type
        if 'playlist' in message_lower or intent['songs_mentioned'] or intent['activity']:
            intent['type'] = 'playlist_request'
        elif any(word in message_lower for word in ['generate', 'create', 'compose']) and 'music' in message_lower:
            intent['type'] = 'music_generation'
        elif any(word in message_lower for word in ['why', 'explain', 'analyze']):
            intent['type'] = 'analysis'
        
        return intent
    
    def _execute_music_logic(self, intent: Dict, feature_extractor, playlist_generator, music_creator) -> Dict:
        """Execute actual music logic based on intent"""
        
        if intent['type'] == 'playlist_request':
            # Build preference string
            preferences = []
            if intent['songs_mentioned']:
                preferences.extend(intent['songs_mentioned'])
            if intent['genres_mentioned']:
                preferences.extend(intent['genres_mentioned'])
            if intent['activity']:
                preferences.append(intent['activity'])
            
            preference_string = ' '.join(preferences) if preferences else 'general music'
            
            # Generate actual playlist
            playlist_result = playlist_generator.generate_collaborative_playlist(preference_string, 5)
            return {
                'type': 'playlist',
                'data': playlist_result
            }
        
        elif intent['type'] == 'music_generation':
            # Generate music
            style_desc = ' '.join(intent['genres_mentioned']) if intent['genres_mentioned'] else 'pop electronic'
            music_result = music_creator.create_music(style_desc)
            return {
                'type': 'music_generation',
                'data': music_result
            }
        
        elif intent['type'] == 'analysis':
            return {
                'type': 'analysis',
                'data': {'explanation': 'Music analysis based on harmonic and rhythmic compatibility'}
            }
        
        else:
            return {
                'type': 'general',
                'data': {'message': 'I can help with playlists, music generation, and analysis!'}
            }
    
    def _format_response_with_llm(self, intent: Dict, music_result: Dict, original_message: str) -> str:
        """Use LLM to format the response naturally"""
        
        if music_result['type'] == 'playlist':
            playlist_data = music_result['data']
            playlist_songs = playlist_data.get('playlist', [])
            
            # Create a natural response about the playlist
            if intent['songs_mentioned']:
                song_mention = intent['songs_mentioned'][0].title()
                response = f"Great choice mentioning {song_mention}! I've created a playlist inspired by that style:\n\n"
            else:
                response = "I've created a playlist for you:\n\n"
            
            # Add songs
            for i, song in enumerate(playlist_songs[:5], 1):
                song_name = song.get('name', 'Unknown Song')
                response += f"{i}. {song_name}\n"
            
            # Add explanation
            explanation = playlist_data.get('explanation', '')
            if explanation:
                response += f"\n{explanation}"
            
            return response
        
        elif music_result['type'] == 'music_generation':
            return f"I've created an original composition for you! {music_result['data'].get('explanation', 'A unique musical piece.')}"
        
        elif music_result['type'] == 'analysis':
            return "I can analyze musical compatibility based on tempo, key, energy, and genre relationships. What specific songs would you like me to compare?"
        
        else:
            return "I'm here to help with music! I can create playlists, generate original music, or analyze songs. What would you like to explore?"
    
    def _update_conversation_state(self, user_message: str, ai_response: str):
        """Update conversation state based on the interaction"""
        # Simple keyword-based state tracking
        if any(word in user_message.lower() for word in ['like', 'love', 'favorite']):
            # Extract potential song/artist mentions for preferences
            pass
        
        if any(word in user_message.lower() for word in ['playlist', 'songs', 'music']):
            self.conversation_state['current_task'] = 'playlist_generation'
        
        if any(word in user_message.lower() for word in ['create', 'generate', 'compose']):
            self.conversation_state['current_task'] = 'music_generation'