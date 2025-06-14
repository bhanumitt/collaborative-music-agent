import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """Initialize the conversation manager with LLM"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_state = {
            'user_preferences': {},
            'current_task': None,
            'song_examples': [],
            'group_preferences': []
        }
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the quantized LLM model for CPU inference"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Configure quantization for CPU efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simpler approach without quantization
            self._load_simple_model()
    
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
        return """You are an AI music assistant specializing in collaborative playlist creation and music generation. Your capabilities include:

1. ANALYZE user music preferences from song examples
2. CREATE collaborative playlists that find common ground between different tastes
3. GENERATE original music by blending different styles
4. EXPLAIN your musical choices and recommendations

Guidelines:
- Always ask for specific song examples to better understand preferences
- Explain your recommendations in musical terms (tempo, key, energy, etc.)
- Offer to create original music when appropriate
- Be conversational and enthusiastic about music
- Use function calls to analyze songs, generate playlists, or create music

Available functions:
- analyze_song_features(song_name, artist): Get musical features of a song
- generate_playlist(preferences, group_size): Create collaborative playlist
- create_original_music(style_blend, tempo, key): Generate new music
- explain_song_choice(song, reason): Explain why a song fits

Remember: You're helping people discover music and create together!"""
    
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
            return "I'm having trouble with my AI model. Let me try to help you with a basic response about music recommendations."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. Could you please try again?"
    
    def process_message(self, message: str, chat_history: List, feature_extractor, playlist_generator, music_creator) -> Dict:
        """Process user message and return response with potential function calls"""
        
        # Create conversation context
        conversation_context = self._create_system_prompt() + "\n\n"
        
        # Add chat history
        for user_msg, ai_msg in chat_history[-3:]:  # Last 3 exchanges for context
            conversation_context += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"
        
        # Add current message
        conversation_context += f"User: {message}\nAssistant:"
        
        # Generate initial response
        response = self._generate_response(conversation_context)
        
        # Extract and execute function calls
        functions = self._extract_function_calls(response)
        function_results = []
        
        for function in functions:
            result = self._execute_function(function, feature_extractor, playlist_generator, music_creator)
            function_results.append(result)
        
        # If there were function calls, generate a final response incorporating the results
        if function_results:
            results_context = conversation_context + response + "\n\nFunction results:\n" + "\n".join(function_results) + "\n\nFinal response based on the above:"
            final_response = self._generate_response(results_context)
            response = final_response if final_response.strip() else response
        
        # Update conversation state
        self._update_conversation_state(message, response)
        
        return {
            'response': response,
            'functions_called': functions,
            'function_results': function_results,
            'conversation_state': self.conversation_state
        }
    
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