from .db import SessionLocal
from .models import LandscapeItem


def add_landscape_item(
    image_path,
    original_filename,
    location,
    comments=None,
    transcription=None,
    content_type=None,
    coordinates=None,
    latitude=None,
    longitude=None,
    source_type="upload",
    status="approved",
):
    session = SessionLocal()
    try:
        item = LandscapeItem(
            image_path=image_path,
            original_filename=original_filename,
            location=location,
            comments=comments,
            transcription=transcription,
            content_type=content_type,
            coordinates=coordinates,
            latitude=latitude,
            longitude=longitude,
            source_type=source_type,
            status=status,
        )
        session.add(item)
        session.commit()
        session.refresh(item)
        return item
    finally:
        session.close()


def get_all_landscape_items(limit=None):
    session = SessionLocal()
    try:
        query = session.query(LandscapeItem).order_by(LandscapeItem.created_at.desc())
        if limit:
            query = query.limit(limit)
        rows = query.all()

        return [
            {
                "id": row.id,
                "image_path": row.image_path,
                "original_filename": row.original_filename,
                "location": row.location,
                "comments": row.comments,
                "transcription": row.transcription,
                "content_type": row.content_type,
                "coordinates": row.coordinates,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "source_type": row.source_type,
                "status": row.status,
            }
            for row in rows
        ]
    finally:
        session.close()
