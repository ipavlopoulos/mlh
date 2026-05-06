import os
import uuid

from flask import Blueprint, current_app, flash, redirect, render_template, request, send_from_directory, url_for
from geopy.geocoders import Nominatim
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS
from werkzeug.utils import secure_filename

from .database.db import Base, engine
from .database.landscape_repository import add_landscape_item, get_all_landscape_items


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LANDSCAPE_FOLDER = os.path.join(UPLOAD_DIR, "landscape")
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MAX_IMAGE_SIDE = 1600
DEFAULT_JPEG_QUALITY = 82

landscape_bp = Blueprint(
    "landscape",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/landscape",
)


def init_landscape_app():
    os.makedirs(LANDSCAPE_FOLDER, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()
        if not exif_data:
            return {}

        data = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                data["GPSInfo"] = {
                    GPSTAGS.get(t, t): value[t]
                    for t in value
                }
            else:
                data[tag] = value
        return data
    except Exception:
        return {}


def get_coordinates(exif):
    gps = exif.get("GPSInfo")
    if not gps:
        return None

    def to_float(value):
        try:
            if isinstance(value, tuple):
                return float(value[0]) / float(value[1])
            return float(value)
        except Exception:
            return 0.0

    def convert(values):
        try:
            degrees = to_float(values[0])
            minutes = to_float(values[1])
            seconds = to_float(values[2])
            return degrees + (minutes / 60) + (seconds / 3600)
        except Exception:
            return None

    try:
        lat = convert(gps.get("GPSLatitude"))
        lon = convert(gps.get("GPSLongitude"))
        if lat is None or lon is None:
            return None
        if gps.get("GPSLatitudeRef") != "N":
            lat = -lat
        if gps.get("GPSLongitudeRef") != "E":
            lon = -lon
        return lat, lon
    except Exception:
        return None


def reverse_geocode(lat, lon):
    try:
        geolocator = Nominatim(user_agent="landscape_app")
        location = geolocator.reverse((lat, lon), language="en")
        if not location:
            return None, None, None

        address = location.raw.get("address", {})
        country = address.get("country")
        city = address.get("city") or address.get("town") or address.get("village")
        street = address.get("road")
        return country, city, street
    except Exception:
        return None, None, None


def save_optimized_image(file, filepath):
    try:
        image = Image.open(file.stream)
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    image = ImageOps.exif_transpose(image).convert("RGB")
    max_side = current_app.config.get("LANDSCAPE_MAX_IMAGE_SIDE", DEFAULT_MAX_IMAGE_SIDE)
    jpeg_quality = current_app.config.get("LANDSCAPE_JPEG_QUALITY", DEFAULT_JPEG_QUALITY)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.save(filepath, "JPEG", quality=jpeg_quality, optimize=True)


@landscape_bp.route("/", methods=["GET", "POST"])
def index():
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        country = request.form.get("country")
        city = request.form.get("city")
        street = request.form.get("street")
        content_type = request.form.get("content_type", "")
        transcription = request.form.get("transcription", "")
        comments = request.form.get("comments", "")

        if not file:
            error = "Image is required."
        elif file.mimetype not in ALLOWED_IMAGE_MIMES:
            error = "Upload a JPEG, PNG, or WebP image."
        else:
            try:
                original_filename = secure_filename(file.filename) or "upload"
                stem = os.path.splitext(original_filename)[0] or "upload"
                filename = f"{uuid.uuid4()}_{stem}.jpg"
                filepath = os.path.join(LANDSCAPE_FOLDER, filename)
                save_optimized_image(file, filepath)

                exif = get_exif_data(filepath)
                coords = get_coordinates(exif)
                latitude = None
                longitude = None
                coordinates = None

                if coords:
                    latitude = str(coords[0])
                    longitude = str(coords[1])
                    coordinates = f"{coords[0]}, {coords[1]}"
                else:
                    lat_form = request.form.get("latitude")
                    lon_form = request.form.get("longitude")
                    if lat_form and lon_form:
                        latitude = lat_form
                        longitude = lon_form
                        coordinates = f"{lat_form}, {lon_form}"

                if latitude and longitude and not (country or city or street):
                    auto_country, auto_city, auto_street = reverse_geocode(latitude, longitude)
                    country = auto_country
                    city = auto_city
                    street = auto_street

                location = ", ".join(filter(None, [street, city, country])) or "Unknown location"
                add_landscape_item(
                    image_path=f"landscape/{filename}",
                    original_filename=file.filename,
                    location=location,
                    transcription=transcription,
                    comments=comments,
                    content_type=content_type,
                    coordinates=coordinates,
                    latitude=latitude,
                    longitude=longitude,
                )

                flash("Thank you! Your wall writing has been added.", "success")
                return redirect(url_for("landscape.index"))
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                current_app.logger.exception("Landscape upload failed")
                error = str(exc)

    items = get_all_landscape_items(limit=5)
    return render_template("landscape.html", items=items, error=error)


@landscape_bp.errorhandler(413)
def upload_too_large(error):
    items = get_all_landscape_items(limit=5)
    return render_template(
        "landscape.html",
        items=items,
        error="The image is too large. Please upload a smaller photo.",
    ), 413


@landscape_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)
