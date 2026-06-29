import os
import base64
import requests
from io import BytesIO
from PIL import Image


def load_image_from_url(url):
    """Load a PIL Image from a URL. Returns None on failure."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to fetch image from {url}: {e}")
        return None


def load_image_from_source(image_source):
    """Load a PIL Image from a URL or local path. Raises on failure."""
    if not image_source:
        raise ValueError("Empty image source provided")
    image_source = str(image_source).strip()
    if image_source.startswith("http"):
        response = requests.get(image_source, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        image = Image.open(image_source).convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Invalid image dimensions")
    return image


def image_to_base64(image):
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def encode_image_to_base64(image_path_or_url):
    """Load an image from a URL or local path and return as a base64 string."""
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url, timeout=10)
        response.raise_for_status()
        image_content = response.content
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
        with open(image_path_or_url, "rb") as f:
            image_content = f.read()
    return base64.b64encode(image_content).decode("utf-8")


def normalize_to_list(data):
    """Normalize JSON data (dict or list) to a flat list of entries."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        return [data]
    return [data]
