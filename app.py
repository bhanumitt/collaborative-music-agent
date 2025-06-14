import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os
from typing import List, Dict, Tuple
import logging

# Import our custom modules
from src.llm_interface.conversation_manager import ConversationManager
from src.music_analysis.feature_extractor import MusicFeatureExtractor
from src.recommendation.playlist_generator import PlaylistGenerator
from src.generation.music_creator import MusicCreator
from src.utils.data_loader import load_sample_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicAgent:
    def __init__(self):
        """Initialize the Collaborative Music Creation Agent"""
        logger.info("Initializing Music Agent...")
        
        # Load sample music data
        self.music_data = load_sample_data()
        
        # Initialize components
        self.conversation_manager = ConversationManager()
        self.feature_extractor = MusicFeatureExtractor()
        self.playlist_generator = PlaylistGenerator(self.music_data)
        self.music_creator = MusicCreator()
        
        # Chat history for the session
        self.chat_history = []
        
        logger.info("Music Agent initialized successfully!")
    
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Process user message and return AI response"""
        try:
            # Update chat history
            self.chat_history = history
            
            # Process message through conversation manager
            response_data = self.conversation_manager.process_message(
                message, 
                self.chat_history,
                self.feature_extractor,
                self.playlist_generator,
                self.music_creator
            )
            
            # Extract response text
            response_text = response_data.get('response', 'I apologize, but I encountered an error processing your request.')
            
            # Update history
            new_history = history + [(message, response_text)]
            
            return response_text, new_history
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = "I'm sorry, I encountered an error. Please try again or rephrase your request."
            new_history = history + [(message, error_response)]
            return error_response, new_history
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_history = []
        return [], []

def create_interface():
    """Create the Gradio interface"""
    
    # Initialize the music agent
    agent = MusicAgent()
    
    # Create the interface
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Collaborative Music Creation Agent",
        css="""
        .container { max-width: 1200px; margin: auto; }
        .chat-container { height: 600px; }
        .header { text-align: center; padding: 20px; }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎵 Collaborative Music Creation Agent</h1>
            <p>AI-powered music assistant that creates collaborative playlists and generates original music</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Music Agent Chat",
                    height=500,
                    placeholder="Hi! I'm your AI music assistant. Tell me about your music preferences or ask me to create a collaborative playlist!",
                    elem_classes=["chat-container"]
                )
                
                # Message input
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here... (e.g., 'Create a playlist for me and my friends who love jazz and EDM')",
                    lines=2
                )
                
                # Buttons
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                # Information panel
                gr.HTML("""
                <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                    <h3>🎼 What I Can Do:</h3>
                    <ul>
                        <li><strong>Analyze Music</strong>: Understand your taste from song examples</li>
                        <li><strong>Create Playlists</strong>: Generate collaborative playlists for groups</li>
                        <li><strong>Generate Music</strong>: Create original songs blending different styles</li>
                        <li><strong>Explain Choices</strong>: Tell you why certain songs work well together</li>
                    </ul>
                    
                    <h3>🎯 Try These Examples:</h3>
                    <ul>
                        <li>"Create a playlist for studying"</li>
                        <li>"My friends like rock and I like jazz - help us find common ground"</li>
                        <li>"Generate an original song that blends classical and electronic"</li>
                        <li>"Why would these two songs work well together?"</li>
                    </ul>
                </div>
                """)
                
                # Sample data info
                gr.HTML("""
                <div style="padding: 15px; background: #e3f2fd; border-radius: 10px; margin-top: 20px;">
                    <h4>📊 Demo Data</h4>
                    <p>This demo uses pre-analyzed music features from popular songs across genres including:</p>
                    <ul>
                        <li>Jazz classics</li>
                        <li>Electronic/EDM hits</li>
                        <li>Rock anthems</li>
                        <li>Pop favorites</li>
                        <li>Classical pieces</li>
                    </ul>
                </div>
                """)
        
        # Event handlers
        def handle_send(message, history):
            if message.strip():
                return agent.process_message(message, history)
            return "", history
        
        def handle_clear():
            agent.clear_chat()
            return [], ""
        
        # Connect events
        send_btn.click(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        ).then(
            lambda: "",  # Clear input after sending
            outputs=msg_input
        )
        
        msg_input.submit(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        ).then(
            lambda: "",  # Clear input after sending
            outputs=msg_input
        )
        
        clear_btn.click(
            handle_clear,
            outputs=[chatbot, msg_input]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with sharing enabled for HuggingFace Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )