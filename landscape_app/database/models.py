from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from .db import Base


class LandscapeItem(Base):
    __tablename__ = "landscape_items"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(Text, nullable=False)
    original_filename = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    comments = Column(Text, nullable=True)
    transcription = Column(Text, nullable=True)
    content_type = Column(String(100), nullable=True)
    coordinates = Column(String(100), nullable=True)
    latitude = Column(String(50), nullable=True)
    longitude = Column(String(50), nullable=True)
    source_type = Column(String(50), default="upload", nullable=False)
    status = Column(String(50), default="approved", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
