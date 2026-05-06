import io
import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from timm import create_model
from torch.nn.functional import softmax
from torchvision import transforms


ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}


class SwinPredictor:
    def __init__(self, model_path, labels, image_size):
        self.model_path = model_path
        self.labels = labels
        self.device = torch.device("cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.load()

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        print(f"Attempting to load model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at: {self.model_path}")
            self.model = None
            return

        model = create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=len(self.labels),
        )
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        print("Model loaded successfully.")

    def predict_file(self, image_file):
        if not self.loaded:
            raise RuntimeError("Model failed to load. Check model path and file.")
        if not image_file or not image_file.filename:
            raise ValueError("No image file provided.")
        if image_file.mimetype not in ALLOWED_IMAGE_MIMES:
            raise ValueError("Unsupported image type. Upload a JPEG, PNG, or WebP image.")

        try:
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "filename": image_file.filename,
            "prediction": self.labels[pred_idx],
            "healthy_prob": round(float(probs[0]), 4),
            "patient_prob": round(float(probs[1]), 4),
        }
