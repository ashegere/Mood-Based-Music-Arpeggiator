"""Favorites router — save, list, download, and delete user's MIDI files."""

import base64
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.saved_midi import SavedMidi
from ..models.user import User

router = APIRouter(prefix="/api/favorites", tags=["favorites"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SaveFavoriteRequest(BaseModel):
    midi_base64: str
    mood: str
    key: str
    scale: str
    tempo: int
    note_count: int
    duration_seconds: float


class SavedMidiResponse(BaseModel):
    id: int
    filename: str
    mood: str
    key: str
    scale: str
    tempo: int
    note_count: int
    duration_seconds: float
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=SavedMidiResponse, status_code=status.HTTP_201_CREATED)
async def save_favorite(
    payload: SaveFavoriteRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Save a generated MIDI to the current user's favorites."""
    try:
        midi_bytes = base64.b64decode(payload.midi_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 MIDI data")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_mood = "".join(c if c.isalnum() else "_" for c in payload.mood.strip())[:40]
    safe_key = payload.key.replace("#", "sharp").replace("b", "flat")
    filename = f"{safe_mood}-{ts}-{safe_key}.mid"

    entry = SavedMidi(
        user_id=current_user.id,
        filename=filename,
        midi_data=midi_bytes,
        mood=payload.mood,
        key=payload.key,
        scale=payload.scale,
        tempo=payload.tempo,
        note_count=payload.note_count,
        duration_seconds=payload.duration_seconds,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("", response_model=List[SavedMidiResponse])
async def list_favorites(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Return all saved MIDIs for the current user, newest first."""
    return (
        db.query(SavedMidi)
        .filter(SavedMidi.user_id == current_user.id)
        .order_by(SavedMidi.created_at.desc())
        .all()
    )


@router.get("/{midi_id}/download")
async def download_favorite(
    midi_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Stream the raw MIDI file for a saved entry."""
    entry = db.query(SavedMidi).filter(
        SavedMidi.id == midi_id,
        SavedMidi.user_id == current_user.id,
    ).first()
    if not entry:
        raise HTTPException(status_code=404, detail="MIDI not found")

    return Response(
        content=entry.midi_data,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{entry.filename}"'},
    )


@router.delete("/{midi_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_favorite(
    midi_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Permanently delete a saved MIDI."""
    entry = db.query(SavedMidi).filter(
        SavedMidi.id == midi_id,
        SavedMidi.user_id == current_user.id,
    ).first()
    if not entry:
        raise HTTPException(status_code=404, detail="MIDI not found")

    db.delete(entry)
    db.commit()
