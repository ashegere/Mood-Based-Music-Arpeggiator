"""SQLAlchemy ORM model for the saved_midis table."""

from sqlalchemy import Column, Integer, String, LargeBinary, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class SavedMidi(Base):
    __tablename__ = "saved_midis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)          # <mood>-<timestamp>-<key>.mid
    midi_data = Column(LargeBinary, nullable=False)    # raw MIDI bytes
    mood = Column(String, nullable=False)
    key = Column(String, nullable=False)
    scale = Column(String, nullable=False)
    tempo = Column(Integer, nullable=False)
    note_count = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="saved_midis")
