import os
import cv2
import numpy as np
from .labels import LABELS


class NeedDetector:
    def __init__(self, known_faces_folder: str):
        self.known_faces_folder = known_faces_folder
        self.known_data = []
        self._load_known_faces()

    def _preprocess_image(self, image_path: str):
        """
        Load image, resize, convert to grayscale, normalize.
        Returns processed image or None if failed.
        """
        image = cv2.imread(image_path)

        if image is None:
            return None

        image = cv2.resize(image, (200, 200))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("float32") / 255.0
        return gray

    def _load_known_faces(self):
        """
        Load all reference images from known_faces folder.
        """
        if not os.path.exists(self.known_faces_folder):
            raise FileNotFoundError(
                f"Known faces folder not found: {self.known_faces_folder}"
            )

        for filename in sorted(os.listdir(self.known_faces_folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(self.known_faces_folder, filename)
            processed = self._preprocess_image(file_path)

            if processed is None:
                print(f"Warning: Could not load image: {filename}")
                continue

            label = LABELS.get(filename, "Unknown")

            self.known_data.append({
                "filename": filename,
                "label": label,
                "image": processed
            })

        if len(self.known_data) == 0:
            raise ValueError("No valid reference images found in known_faces folder.")

    def _calculate_score(self, img1, img2):
        """
        Lower score = better match
        Uses Mean Squared Error.
        """
        diff = (img1 - img2) ** 2
        mse = np.mean(diff)
        return mse

    def predict_need(self, test_image_path: str):
        """
        Compare test image with all known images and return best match.
        Returns:
            predicted_label, matched_filename, score
        """
        if not os.path.exists(test_image_path):
            return "Test image not found", None, None

        test_img = self._preprocess_image(test_image_path)

        if test_img is None:
            return "Could not read test image", None, None

        best_score = float("inf")
        best_label = None
        best_filename = None

        for item in self.known_data:
            score = self._calculate_score(test_img, item["image"])

            if score < best_score:
                best_score = score
                best_label = item["label"]
                best_filename = item["filename"]

        return best_label, best_filename, best_score
