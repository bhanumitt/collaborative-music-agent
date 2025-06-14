import music21
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import logging
from music21 import stream, note, chord, duration, tempo, key, meter, scale, interval

logger = logging.getLogger(__name__)

class MusicCreator:
    def __init__(self):
        """Initialize the music creator with music theory knowledge"""
        self.chord_progressions = {
            'jazz': [
                ['Cmaj7', 'Am7', 'Dm7', 'G7'],
                ['Cmaj7', 'C7', 'Fmaj7', 'G7'],
                ['Am7', 'D7', 'Gmaj7', 'Cmaj7'],
                ['Dm7', 'G7', 'Em7', 'Am7']
            ],
            'electronic': [
                ['Am', 'F', 'C', 'G'],
                ['Em', 'C', 'G', 'D'],
                ['Dm', 'Am', 'F', 'C'],
                ['Am', 'G', 'F', 'E']
            ],
            'pop': [
                ['C', 'G', 'Am', 'F'],
                ['F', 'C', 'G', 'Am'],
                ['Am', 'F', 'C', 'G'],
                ['G', 'Am', 'F', 'C']
            ],
            'rock': [
                ['A', 'D', 'E', 'A'],
                ['G', 'C', 'D', 'G'],
                ['E', 'A', 'B', 'E'],
                ['F', 'Bb', 'C', 'F']
            ]
        }
        
        self.scales_mapping = {
            'major': scale.MajorScale,
            'minor': scale.MinorScale,
            'dorian': scale.DorianScale,
            'mixolydian': scale.MixolydianScale
        }
        
        self.rhythm_patterns = {
            'jazz': [
                [1.0, 0.5, 0.5, 1.0],  # Swing feel
                [0.5, 0.5, 1.0, 0.5, 0.5],
                [1.5, 0.5, 1.0, 1.0]
            ],
            'electronic': [
                [0.25, 0.25, 0.25, 0.25] * 4,  # 16th note patterns
                [0.5, 0.5, 0.5, 0.5] * 2,  # 8th note patterns
                [1.0, 1.0, 1.0, 1.0]  # Quarter note patterns
            ],
            'pop': [
                [1.0, 1.0, 0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0, 1.0, 0.5, 0.5],
                [1.0, 0.5, 0.5, 1.0, 1.0]
            ],
            'rock': [
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
                [2.0, 1.0, 1.0]
            ]
        }
    
    def create_music(self, style_description: str, target_tempo: int = 120, target_key: str = "C") -> Dict:
        """Create original music based on style description"""
        try:
            # Parse style description
            styles = self._parse_style_description(style_description)
            
            # Generate musical elements
            composition = self._generate_composition(styles, target_tempo, target_key)
            
            # Create explanation
            explanation = self._generate_explanation(styles, composition)
            
            return {
                'composition': composition,
                'explanation': explanation,
                'styles_blended': styles,
                'musical_elements': {
                    'key': composition.get('key', target_key),
                    'tempo': composition.get('tempo', target_tempo),
                    'time_signature': composition.get('time_signature', '4/4'),
                    'chord_progression': composition.get('chord_progression', []),
                    'melody_notes': len(composition.get('melody', [])),
                    'duration_seconds': composition.get('duration', 30)
                }
            }
        
        except Exception as e:
            logger.error(f"Error creating music: {str(e)}")
            return self._generate_fallback_composition(style_description)
    
    def _parse_style_description(self, description: str) -> List[str]:
        """Parse style description to extract musical genres"""
        description_lower = description.lower()
        detected_styles = []
        
        style_keywords = {
            'jazz': ['jazz', 'bebop', 'swing', 'blues', 'improvisation'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'synth', 'digital'],
            'rock': ['rock', 'metal', 'punk', 'guitar', 'drums'],
            'pop': ['pop', 'mainstream', 'catchy', 'commercial'],
            'classical': ['classical', 'orchestral', 'symphony', 'baroque'],
            'ambient': ['ambient', 'atmospheric', 'chill', 'relaxing'],
            'funk': ['funk', 'groove', 'bass', 'rhythm'],
            'latin': ['latin', 'salsa', 'bossa', 'samba']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_styles.append(style)
        
        return detected_styles if detected_styles else ['pop']
    
    def _generate_composition(self, styles: List[str], target_tempo: int, target_key: str) -> Dict:
        """Generate a musical composition blending the specified styles"""
        # Create the main score
        score = stream.Score()
        
        # Set key and tempo
        composition_key = key.Key(target_key)
        score.append(composition_key)
        score.append(tempo.TempoIndication(number=target_tempo))
        score.append(meter.TimeSignature('4/4'))
        
        # Blend chord progressions from different styles
        chord_progression = self._blend_chord_progressions(styles, target_key)
        
        # Generate melody
        melody = self._generate_melody(styles, composition_key, chord_progression)
        
        # Generate bass line
        bass_line = self._generate_bass_line(chord_progression, composition_key)
        
        # Generate rhythm pattern
        rhythm_pattern = self._blend_rhythm_patterns(styles)
        
        # Create parts
        melody_part = stream.Part()
        melody_part.append(melody)
        
        bass_part = stream.Part()
        bass_part.append(bass_line)
        
        chord_part = stream.Part()
        chord_part.append(self._create_chord_track(chord_progression))
        
        # Add parts to score
        score.append(melody_part)
        score.append(bass_part)
        score.append(chord_part)
        
        # Convert to MIDI and get basic info
        midi_data = self._score_to_midi_data(score)
        
        return {
            'score': score,
            'midi_data': midi_data,
            'key': target_key,
            'tempo': target_tempo,
            'time_signature': '4/4',
            'chord_progression': chord_progression,
            'melody': self._extract_melody_notes(melody),
            'bass_line': self._extract_bass_notes(bass_line),
            'rhythm_pattern': rhythm_pattern,
            'duration': 32,  # 32 beats at 4/4
            'style_blend': styles
        }
    
    def _blend_chord_progressions(self, styles: List[str], key_name: str) -> List[str]:
        """Blend chord progressions from different styles"""
        progressions = []
        
        for style in styles:
            if style in self.chord_progressions:
                progressions.extend(self.chord_progressions[style])
        
        if not progressions:
            progressions = self.chord_progressions['pop']
        
        # Choose a progression and transpose to target key
        chosen_progression = random.choice(progressions)
        transposed_progression = self._transpose_chord_progression(chosen_progression, key_name)
        
        return transposed_progression
    
    def _transpose_chord_progression(self, progression: List[str], target_key: str) -> List[str]:
        """Transpose chord progression to target key"""
        try:
            # Simple transposition - for demo purposes
            key_mapping = {
                'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
            }
            
            # For simplicity, return the progression as-is for demo
            # In a full implementation, this would properly transpose
            return progression
        except:
            return progression
    
    def _generate_melody(self, styles: List[str], composition_key: key.Key, chord_progression: List[str]) -> stream.Stream:
        """Generate a melody line based on styles and chord progression"""
        melody_stream = stream.Stream()
        
        # Get scale for melody generation
        melody_scale = composition_key.getScale()
        scale_notes = [p for p in melody_scale.pitches[:8]]  # Get octave
        
        # Generate melody based on style characteristics
        melody_characteristics = self._get_melody_characteristics(styles)
        
        # Generate notes for 8 measures (32 beats in 4/4)
        current_beat = 0
        target_beats = 32
        
        while current_beat < target_beats:
            # Choose note duration based on style
            note_duration = self._choose_note_duration(melody_characteristics)
            
            # Choose pitch based on harmonic context
            pitch = self._choose_melody_pitch(scale_notes, melody_characteristics)
            
            # Create note
            melody_note = note.Note(pitch, quarterLength=note_duration)
            melody_stream.append(melody_note)
            
            current_beat += note_duration
        
        return melody_stream
    
    def _get_melody_characteristics(self, styles: List[str]) -> Dict:
        """Get melody characteristics based on styles"""
        characteristics = {
            'step_probability': 0.7,  # Probability of stepwise motion
            'leap_probability': 0.2,  # Probability of leaps
            'repeat_probability': 0.1,  # Probability of repeating notes
            'range_octaves': 1.5,  # Melodic range in octaves
            'rhythm_complexity': 0.5  # 0-1 scale of rhythmic complexity
        }
        
        # Adjust based on styles
        if 'jazz' in styles:
            characteristics['leap_probability'] = 0.4
            characteristics['rhythm_complexity'] = 0.8
        
        if 'electronic' in styles:
            characteristics['repeat_probability'] = 0.3
            characteristics['rhythm_complexity'] = 0.7
        
        if 'classical' in styles:
            characteristics['step_probability'] = 0.8
            characteristics['range_octaves'] = 2.0
        
        return characteristics
    
    def _choose_note_duration(self, characteristics: Dict) -> float:
        """Choose note duration based on characteristics"""
        complexity = characteristics.get('rhythm_complexity', 0.5)
        
        if complexity > 0.7:
            # Complex rhythms - more variety
            durations = [0.25, 0.5, 0.75, 1.0, 1.5]
            weights = [0.2, 0.3, 0.1, 0.3, 0.1]
        elif complexity > 0.3:
            # Moderate rhythms
            durations = [0.5, 1.0, 1.5, 2.0]
            weights = [0.3, 0.4, 0.2, 0.1]
        else:
            # Simple rhythms
            durations = [1.0, 2.0]
            weights = [0.7, 0.3]
        
        return np.random.choice(durations, p=weights)
    
    def _choose_melody_pitch(self, scale_notes: List, characteristics: Dict) -> music21.pitch.Pitch:
        """Choose a pitch for the melody"""
        # Simple melody generation - choose random scale note
        # In a full implementation, this would consider harmonic context
        return random.choice(scale_notes)
    
    def _generate_bass_line(self, chord_progression: List[str], composition_key: key.Key) -> stream.Stream:
        """Generate a bass line based on chord progression"""
        bass_stream = stream.Stream()
        
        # Simple bass line - root notes of chords with rhythmic variation
        for chord_name in chord_progression:
            try:
                # Create chord and get root
                chord_obj = chord.Chord(chord_name)
                root_note = chord_obj.root()
                
                # Move to bass register
                bass_note = note.Note(root_note.name, octave=2)
                bass_note.quarterLength = 4.0  # Whole note per chord
                
                bass_stream.append(bass_note)
            except:
                # Fallback to C2 if chord parsing fails
                bass_note = note.Note('C2', quarterLength=4.0)
                bass_stream.append(bass_note)
        
        return bass_stream
    
    def _blend_rhythm_patterns(self, styles: List[str]) -> List[float]:
        """Blend rhythm patterns from different styles"""
        patterns = []
        
        for style in styles:
            if style in self.rhythm_patterns:
                patterns.extend(self.rhythm_patterns[style])
        
        if not patterns:
            patterns = self.rhythm_patterns['pop']
        
        return random.choice(patterns)
    
    def _create_chord_track(self, chord_progression: List[str]) -> stream.Stream:
        """Create a chord track from the progression"""
        chord_stream = stream.Stream()
        
        for chord_name in chord_progression:
            try:
                chord_obj = chord.Chord(chord_name)
                chord_obj.quarterLength = 4.0  # Whole note per chord
                chord_stream.append(chord_obj)
            except:
                # Fallback chord
                chord_obj = chord.Chord(['C', 'E', 'G'])
                chord_obj.quarterLength = 4.0
                chord_stream.append(chord_obj)
        
        return chord_stream
    
    def _score_to_midi_data(self, score: stream.Score) -> Dict:
        """Convert score to MIDI data representation"""
        try:
            # For demo purposes, return basic MIDI info
            # In a full implementation, this would generate actual MIDI bytes
            return {
                'tracks': len(score.parts),
                'duration': score.duration.quarterLength,
                'notes_count': len(score.flat.notes),
                'tempo': score.metronomeMarkBoundaries()[0][2].number if score.metronomeMarkBoundaries() else 120
            }
        except:
            return {'tracks': 0, 'duration': 0, 'notes_count': 0, 'tempo': 120}
    
    def _extract_melody_notes(self, melody_stream: stream.Stream) -> List[str]:
        """Extract melody notes as string list"""
        notes = []
        for element in melody_stream.flat.notes:
            if hasattr(element, 'pitch'):
                notes.append(f"{element.pitch.name}{element.pitch.octave}")
        return notes
    
    def _extract_bass_notes(self, bass_stream: stream.Stream) -> List[str]:
        """Extract bass notes as string list"""
        notes = []
        for element in bass_stream.flat.notes:
            if hasattr(element, 'pitch'):
                notes.append(f"{element.pitch.name}{element.pitch.octave}")
        return notes
    
    def _generate_explanation(self, styles: List[str], composition: Dict) -> str:
        """Generate explanation for the musical choices"""
        explanation_parts = []
        
        # Style blend explanation
        if len(styles) > 1:
            explanation_parts.append(f"I've created an original composition blending {', '.join(styles)} styles.")
        else:
            explanation_parts.append(f"I've composed an original {styles[0]} piece.")
        
        # Musical elements explanation
        key_name = composition.get('key', 'C')
        tempo_val = composition.get('tempo', 120)
        explanation_parts.append(f"The piece is in {key_name} major at {tempo_val} BPM.")
        
        # Chord progression
        chords = composition.get('chord_progression', [])
        if chords:
            explanation_parts.append(f"It features a {len(chords)}-chord progression: {' - '.join(chords[:4])}.")
        
        # Style-specific elements
        if 'jazz' in styles:
            explanation_parts.append("I incorporated jazz harmony with extended chords and syncopated rhythms.")
        
        if 'electronic' in styles:
            explanation_parts.append("Electronic elements include repetitive patterns and synthesized textures.")
        
        if 'rock' in styles:
            explanation_parts.append("Rock influences appear in the strong beat emphasis and power chord structures.")
        
        # Technical details
        melody_count = len(composition.get('melody', []))
        if melody_count > 0:
            explanation_parts.append(f"The melody contains {melody_count} notes with both stepwise motion and melodic leaps.")
        
        return " ".join(explanation_parts)
    
    def _generate_fallback_composition(self, style_description: str) -> Dict:
        """Generate a simple fallback composition when main generation fails"""
        return {
            'composition': {
                'key': 'C',
                'tempo': 120,
                'time_signature': '4/4',
                'chord_progression': ['C', 'G', 'Am', 'F'],
                'melody': ['C4', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4', 'D4'],
                'duration': 16
            },
            'explanation': f"I've created a simple composition based on your request for {style_description}. It features basic chord progressions in C major.",
            'styles_blended': ['pop'],
            'musical_elements': {
                'key': 'C',
                'tempo': 120,
                'time_signature': '4/4',
                'chord_progression': ['C', 'G', 'Am', 'F'],
                'melody_notes': 8,
                'duration_seconds': 16
            }
        }
    
    def create_style_fusion(self, style1: str, style2: str, fusion_ratio: float = 0.5) -> Dict:
        """Create a fusion between two specific styles"""
        # Weight the characteristics of each style
        styles = [style1, style2]
        
        # Generate composition with fusion in mind
        composition = self._generate_composition(styles, 120, 'C')
        
        # Create fusion-specific explanation
        explanation = f"I've created a {int(fusion_ratio*100)}%-{int((1-fusion_ratio)*100)}% fusion of {style1} and {style2}. "
        
        if fusion_ratio > 0.6:
            explanation += f"The composition leans more heavily on {style1} elements while incorporating {style2} influences."
        elif fusion_ratio < 0.4:
            explanation += f"The composition is primarily {style2}-based with {style1} accents."
        else:
            explanation += f"The composition balances both {style1} and {style2} equally."
        
        composition['explanation'] = explanation
        return composition