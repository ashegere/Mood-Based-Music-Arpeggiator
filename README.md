# AI Music Arpeggiator

An AI-powered music generation system that creates creative arpeggios based on mood, key, tempo, and pattern styles using compound AI systems.

## 🎵 Features

- **AI-Powered Generation**: Generate unique arpeggios using advanced AI algorithms
- **Mood-Based Composition**: Choose from various moods (Happy, Calm, Energetic, Dark, etc.)
- **Multiple Pattern Styles**: AI Generated, Ascending, Descending, Alternating, Random
- **Web Interface**: Modern, responsive frontend with real-time visualization
- **MIDI Export**: Download generated patterns for use in DAWs
- **Audio Playback**: Built-in Web Audio API playback
- **Piano Roll Visualization**: Real-time display of generated musical patterns

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modern web browser with Web Audio API support

### Installation & Running

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Music-Arpeggiator
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the complete application**
   ```bash
   # Simple launcher (recommended)
   ./run.sh

   # Or manually:
   # Terminal 1: Start backend
   cd backend
   python3 -m uvicorn app.main:app --reload

   # Terminal 2: Start frontend
   cd frontend
   python3 -m http.server 3000
   ```

4. **Open your browser**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8006/docs

## 🎨 Frontend Interface

The web interface provides:

- **Control Panel**: Adjust key, mood, BPM, pattern style, and duration
- **Piano Roll**: Visual representation of generated arpeggios
- **Playback Controls**: Play, pause, stop, and download functionality
- **Real-time Feedback**: Status updates and pattern descriptions

## 🔧 API Endpoints

### Core Endpoints

- `GET /` - Welcome message and API info
- `GET /health` - Health check
- `GET /api/moods` - Available moods
- `GET /api/pattern-styles` - Available pattern styles
- `POST /api/generate` - Generate arpeggio with visualization data
- `POST /api/generate/midi` - Generate and download MIDI file

### Request/Response Examples

**Generate Arpeggio:**
```json
POST /api/generate
{
  "key": "C",
  "mood": "happy",
  "bpm": 120,
  "num_bars": 2,
  "pattern_style": "ai-generated"
}
```

**Response:**
```json
{
  "notes": [
    {
      "pitch": 60,
      "start_time": 0.0,
      "end_time": 0.5,
      "velocity": 100
    }
  ],
  "midi_base64": "...",
  "tempo": 120,
  "key": "C",
  "mood": "happy",
  "duration": 2.0,
  "pattern_description": "An uplifting arpeggio pattern..."
}
```

## 🏗️ Architecture

### Backend (FastAPI)
- **AI Engine**: Hybrid music generation using pattern generators and music theory
- **MIDI Processing**: Conversion between MIDI and various formats
- **REST API**: FastAPI with automatic OpenAPI documentation
- **CORS Support**: Configured for frontend integration

### Frontend (Vanilla JS)
- **HTML5 Canvas**: Piano roll visualization
- **Web Audio API**: Real-time audio synthesis
- **Responsive Design**: Mobile-friendly interface
- **Modern CSS**: Dark theme with smooth animations

## 🎼 Music Theory Integration

The system incorporates:
- **Scale Theory**: Proper note selection based on musical keys
- **Rhythm Patterns**: Various arpeggio styles and timing
- **Mood Mapping**: Emotional characteristics translated to musical parameters
- **Tempo Control**: BPM-based timing calculations

## 🔍 Development

### Project Structure
```
Music-Arpeggiator/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Main API application
│   │   ├── models/         # Pydantic schemas
│   │   └── services/       # AI and MIDI services
│   └── test_generator.py   # Testing utilities
├── frontend/                # Web interface
│   ├── index.html          # Main HTML
│   ├── styles.css          # Modern CSS
│   ├── app.js             # Frontend logic
│   └── README.md          # Frontend docs
├── run.sh                  # Launcher script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Adding New Features

1. **Backend**: Add new endpoints in `backend/app/main.py`
2. **Frontend**: Update `frontend/app.js` and HTML templates
3. **Models**: Modify schemas in `backend/app/models/schemas.py`

## 📝 License

MIT License - see individual component licenses for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🎯 Future Enhancements

- [ ] Multiple instrument support
- [ ] Chord progression generation
- [ ] Real-time collaboration
- [ ] Plugin architecture for new AI models
- [ ] Advanced music theory features
 