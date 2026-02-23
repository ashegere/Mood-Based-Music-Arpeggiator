from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from .models.schemas import (
    ArpeggiatorRequest,
    ArpeggiatorResponse,
    NoteData,
    HealthResponse,
    MoodsResponse
)
from .services.hybrid_generator import HybridMusicGenerator
from .services.midi_processor import MidiProcessor
from .config import settings
from .database import engine, Base
from .routers import auth
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered arpeggiator using compound AI system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)

# Global instances
generator: HybridMusicGenerator = None
midi_processor: MidiProcessor = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global generator, midi_processor

    logger.info("Starting up AI Arpeggiator API...")

    try:
        # Create database tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Initialize services
        midi_processor = MidiProcessor()
        generator = HybridMusicGenerator()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "ready" if generator else "initializing"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if generator else "initializing",
        model_loaded=generator is not None,
        version=settings.VERSION
    )


@app.get("/api/moods", response_model=MoodsResponse, tags=["Configuration"])
async def get_moods():
    """Get available moods"""
    if not generator:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    moods = generator.get_available_moods()
    
    return MoodsResponse(
        moods=moods,
        default='happy'
    )


@app.post("/api/generate", response_model=ArpeggiatorResponse, tags=["Generation"])
async def generate_arpeggio(request: ArpeggiatorRequest):
    """
    Generate an arpeggio based on parameters
    
    - **key**: Musical key (C, D, E, F, G, A, B with # or b)
    - **mood**: Mood descriptor (happy, calm, energetic, dark, etc.)
    - **bpm**: Tempo in beats per minute (40-240)
    - **num_bars**: Number of bars to generate (1-8)
    - **pattern_style**: Pattern style (ascending, descending, alternating, random, ai-generated)
    - **seed**: Optional random seed for reproducibility
    """
    
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again in a moment."
        )
    
    try:
        logger.info(f"Received generation request: {request.dict()}")
        
        # Generate arpeggio
        midi, description = generator.generate_arpeggio(
            key=request.key,
            mood=request.mood,
            bpm=request.bpm,
            num_bars=request.num_bars,
            pattern_style=request.pattern_style,
            seed=request.seed
        )
        
        # Convert MIDI to base64
        midi_base64 = midi_processor.midi_to_base64(midi)
        
        # Extract note data for frontend playback
        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append(NoteData(
                    pitch=note.pitch,
                    start_time=note.start,
                    end_time=note.end,
                    velocity=note.velocity
                ))
        
        # Get duration
        duration = midi.get_end_time()
        
        response = ArpeggiatorResponse(
            notes=notes,
            midi_base64=midi_base64,
            tempo=request.bpm,
            key=request.key,
            mood=request.mood,
            duration=duration,
            pattern_description=description
        )
        
        logger.info(f"Successfully generated arpeggio with {len(notes)} notes, duration: {duration:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating arpeggio: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate arpeggio: {str(e)}"
        )


@app.post("/api/generate/midi", tags=["Generation"])
async def generate_arpeggio_midi(request: ArpeggiatorRequest):
    """
    Generate and download MIDI file directly
    """
    
    if not generator:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        # Generate arpeggio
        midi, _ = generator.generate_arpeggio(
            key=request.key,
            mood=request.mood,
            bpm=request.bpm,
            num_bars=request.num_bars,
            pattern_style=request.pattern_style,
            seed=request.seed
        )
        
        # Convert to bytes
        midi_bytes = midi_processor.midi_to_bytes(midi)
        
        # Create filename
        filename = f"arpeggio_{request.key}_{request.mood}_{request.bpm}bpm.mid"
        
        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating MIDI file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pattern-styles", tags=["Configuration"])
async def get_pattern_styles():
    """Get available pattern styles"""
    return {
        "styles": [
            {
                "value": "ai-generated",
                "label": "AI Generated",
                "description": "Creative pattern generated by AI"
            },
            {
                "value": "ascending",
                "label": "Ascending",
                "description": "Notes move upward in pitch"
            },
            {
                "value": "descending",
                "label": "Descending",
                "description": "Notes move downward in pitch"
            },
            {
                "value": "alternating",
                "label": "Alternating",
                "description": "Notes alternate up and down"
            },
            {
                "value": "random",
                "label": "Random",
                "description": "Random pattern within scale"
            }
        ],
        "default": "ai-generated"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8006,
        reload=settings.DEBUG
    )