"""
face_engine.py
Core Face Detection, Dataset Management, LBPH Model Training & Recognition
"""

import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def safe_imread(file_path: str) -> Optional[np.ndarray]:
    """
    Safely reads an image from disk using numpy to support unicode paths and spaces on Windows.
    """
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, 'rb') as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def safe_imwrite(file_path: str, img: np.ndarray) -> bool:
    """
    Safely writes an image to disk using cv2.imencode to support unicode paths.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".jpg"
        success, encoded = cv2.imencode(ext, img)
        if success:
            with open(file_path, "wb") as f:
                f.write(encoded)
            return True
        return False
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False


class FaceEngine:
    """
    High-level Face Recognition engine using OpenCV Haar Cascades and LBPH Recognizer.
    Supports enrollment from saved image files/folders, live webcam capture, training, and recognition.
    """

    TARGET_FACE_SIZE = (200, 200)

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.models_dir = self.base_dir / "models"
        self.dataset_dir = self.base_dir / "dataset"
        self.logs_dir = self.base_dir / "logs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.cascade_path = self._find_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))

        self.model_file = self.models_dir / "trainer.yml"
        self.labels_file = self.models_dir / "labels.json"

        self.recognizer: Optional[Any] = None
        self.label_map: Dict[int, str] = {}  # {1: "Alice", 2: "Bob"}
        self.name_to_id: Dict[str, int] = {}

        # Default confidence threshold for LBPH (distance: lower is closer)
        # Anything with distance <= threshold is accepted as known person.
        self.confidence_threshold = 78.0

        self.load_model()

    def _find_cascade_path(self) -> Path:
        """Locates the Haar cascade XML file, prioritizing local models/ directory."""
        local_cascade = self.models_dir / "haarcascade_frontalface_default.xml"
        if local_cascade.exists():
            return local_cascade
        
        # Check cv2.data
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            builtin = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if builtin.exists():
                return builtin

        return local_cascade

    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.15, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60)) -> List[Tuple[int, int, int, int]]:
        """
        Detects frontal faces in the input BGR image.
        Returns list of bounding boxes: [(x, y, w, h), ...]
        """
        if image is None or image.size == 0:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for normalized contrast
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_face_roi(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops, converts to grayscale, equalizes histogram and resizes face ROI to standard size.
        """
        x, y, w, h = box
        # Add slight padding if within bounds
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)

        face_roi = image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.equalizeHist(gray_roi)
        resized_roi = cv2.resize(gray_roi, self.TARGET_FACE_SIZE, interpolation=cv2.INTER_AREA)
        return resized_roi

    # -------------------------------------------------------------
    # Dataset & Enrollment from Saved Images
    # -------------------------------------------------------------

    def sanitize_person_name(self, name: str) -> str:
        """Sanitizes person name to create a safe directory name."""
        cleaned = "".join(c for c in name.strip() if c.isalnum() or c in (" ", "_", "-"))
        cleaned = cleaned.replace(" ", "_")
        return cleaned or "Unnamed"

    def get_registered_persons(self) -> List[Dict[str, Any]]:
        """
        Scans dataset/ directory and returns list of registered persons with sample counts.
        """
        persons = []
        if not self.dataset_dir.exists():
            return persons

        for entry in sorted(self.dataset_dir.iterdir()):
            if entry.is_dir():
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                images = [f for f in entry.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
                persons.append({
                    "name": entry.name,
                    "count": len(images),
                    "dir_path": str(entry),
                    "is_in_model": entry.name in self.name_to_id
                })
        return persons

    def enroll_from_images(self, person_name: str, file_paths: List[str]) -> Tuple[int, int]:
        """
        Extracts face regions from provided image files and saves them under dataset/<person_name>/.
        Returns: (faces_extracted_count, total_files_processed)
        """
        clean_name = self.sanitize_person_name(person_name)
        person_dir = self.dataset_dir / clean_name
        person_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        total_files = len(file_paths)

        # Existing files count to continue numbering
        existing_count = len(list(person_dir.glob("*.jpg")))

        for idx, fpath in enumerate(file_paths):
            img = safe_imread(fpath)
            if img is None:
                continue

            boxes = self.detect_faces(img)
            if not boxes:
                # Fallback: if no face detected with default params, try more sensitive detection
                boxes = self.detect_faces(img, scale_factor=1.08, min_neighbors=3, min_size=(40, 40))

            if not boxes:
                # If still no face detected, if the user explicitly provided a close crop photo,
                # we can use the whole image resized to 200x200
                h, w = img.shape[:2]
                if w >= 80 and h >= 80:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    face_roi = cv2.resize(gray, self.TARGET_FACE_SIZE)
                    save_path = person_dir / f"face_{existing_count + extracted_count + 1:04d}.jpg"
                    cv2.imwrite(str(save_path), face_roi)
                    extracted_count += 1
                continue

            # If faces found, extract the largest face (in case of background people)
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
            primary_box = boxes[0]

            face_roi = self.extract_face_roi(img, primary_box)
            save_path = person_dir / f"face_{existing_count + extracted_count + 1:04d}.jpg"
            cv2.imwrite(str(save_path), face_roi)
            extracted_count += 1

        return extracted_count, total_files

    def enroll_from_folder(self, person_name: str, folder_path: str) -> Tuple[int, int]:
        """
        Reads all images in folder_path and enrolls them for person_name.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return 0, 0

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
        return self.enroll_from_images(person_name, image_files)

    def save_single_face_sample(self, person_name: str, face_img: np.ndarray) -> bool:
        """
        Saves a single face ROI (e.g. captured from webcam) to person's dataset folder.
        """
        clean_name = self.sanitize_person_name(person_name)
        person_dir = self.dataset_dir / clean_name
        person_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(person_dir.glob("*.jpg")))
        save_path = person_dir / f"face_{existing_count + 1:04d}.jpg"
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img

        face_gray = cv2.equalizeHist(face_gray)
        face_roi = cv2.resize(face_gray, self.TARGET_FACE_SIZE)
        return cv2.imwrite(str(save_path), face_roi)

    def delete_person(self, person_name: str) -> bool:
        """Deletes all saved images for person_name."""
        clean_name = self.sanitize_person_name(person_name)
        person_dir = self.dataset_dir / clean_name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir, ignore_errors=True)
            return True
        return False

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    def train_model(self) -> Tuple[bool, str]:
        """
        Trains LBPHFaceRecognizer on all faces in dataset/ and saves trainer.yml + labels.json.
        """
        persons = self.get_registered_persons()
        if not persons:
            return False, "No dataset found. Please enroll at least one person with photos."

        faces = []
        labels = []
        label_map: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}
        current_id = 1

        total_samples = 0
        for p in persons:
            p_name = p["name"]
            p_dir = Path(p["dir_path"])
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            p_images = [f for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

            if not p_images:
                continue

            label_map[current_id] = p_name
            name_to_id[p_name] = current_id

            for img_path in p_images:
                img = safe_imread(str(img_path))
                if img is None:
                    continue

                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

                # Normalize size
                if gray.shape != self.TARGET_FACE_SIZE:
                    gray = cv2.resize(gray, self.TARGET_FACE_SIZE)

                faces.append(gray)
                labels.append(current_id)
                total_samples += 1

            current_id += 1

        if not faces or len(label_map) == 0:
            return False, "No valid face samples found in dataset directory."

        try:
            # Create LBPH Face Recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8
            )
            recognizer.train(faces, np.array(labels, dtype=np.int32))

            # Save trainer model
            recognizer.write(str(self.model_file))

            # Save labels mapping with metadata
            metadata = {
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_persons": len(label_map),
                "total_samples": total_samples,
                "label_map": {str(k): v for k, v in label_map.items()},
                "name_to_id": name_to_id
            }
            with open(self.labels_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            # Update in-memory recognizer
            self.recognizer = recognizer
            self.label_map = label_map
            self.name_to_id = name_to_id

            return True, f"Successfully trained with {total_samples} faces across {len(label_map)} persons."
        except Exception as e:
            return False, f"Training failed: {str(e)}"

    def is_trained(self) -> bool:
        """Returns True if trained model and labels exist."""
        return self.recognizer is not None and len(self.label_map) > 0

    def load_model(self) -> bool:
        """Loads trained LBPH model and labels if files exist."""
        if self.model_file.exists() and self.labels_file.exists():
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(str(self.model_file))
                with open(self.labels_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                raw_map = data.get("label_map", {})
                self.label_map = {int(k): v for k, v in raw_map.items()}
                self.name_to_id = data.get("name_to_id", {})
                self.recognizer = recognizer
                return True
            except Exception as e:
                print(f"Could not load trained model: {e}")
                self.recognizer = None
                self.label_map = {}
                self.name_to_id = {}
        return False

    # -------------------------------------------------------------
    # Recognition
    # -------------------------------------------------------------

    def calculate_confidence_percentage(self, distance: float) -> float:
        """
        Converts LBPH distance (where 0 is a perfect match) to an intuitive 0-100% confidence.
        Distance ~ 0-45 -> 80-100%
        Distance ~ 45-75 -> 50-80%
        Distance > 90 -> <35%
        """
        if distance < 0:
            return 100.0
        # Linear decay mapping from 0 (100%) to 110 (0%)
        conf = 100.0 - (distance / 1.1)
        return float(np.clip(round(conf, 1), 0.0, 100.0))

    def recognize_frame(self, frame: np.ndarray, custom_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detects faces in frame and performs LBPH prediction on each face.
        Returns list of detections with bounding box, recognized name, confidence %, raw distance, and is_known flag.
        """
        if frame is None or frame.size == 0:
            return []

        thresh = custom_threshold if custom_threshold is not None else self.confidence_threshold
        boxes = self.detect_faces(frame)
        results = []

        for box in boxes:
            face_roi = self.extract_face_roi(frame, box)

            if self.is_trained() and self.recognizer is not None:
                try:
                    label_id, distance = self.recognizer.predict(face_roi)
                    confidence_pct = self.calculate_confidence_percentage(distance)

                    # LBPH distance <= threshold means recognized known person
                    if distance <= thresh and label_id in self.label_map:
                        person_name = self.label_map[label_id]
                        is_known = True
                    else:
                        person_name = "Unknown"
                        is_known = False

                    results.append({
                        "box": box,
                        "label_id": label_id,
                        "name": person_name,
                        "distance": round(float(distance), 2),
                        "confidence": confidence_pct,
                        "is_known": is_known
                    })
                except Exception as e:
                    results.append({
                        "box": box,
                        "label_id": -1,
                        "name": "Error",
                        "distance": 999.0,
                        "confidence": 0.0,
                        "is_known": False
                    })
            else:
                # Model not trained yet
                results.append({
                    "box": box,
                    "label_id": -1,
                    "name": "Untrained",
                    "distance": 999.0,
                    "confidence": 0.0,
                    "is_known": False
                })

        return results

    def draw_detections(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws modern bounding boxes and clean label badges on the image.
        """
        annotated = image.copy()

        for res in results:
            x, y, w, h = res["box"]
            name = res["name"]
            conf = res["confidence"]
            is_known = res["is_known"]

            # Colors in BGR
            # Green for known, Crimson red for unknown/untrained
            if is_known:
                box_color = (46, 204, 113)    # Emerald Green
                text_bg = (39, 174, 96)
                badge_text = f"{name} ({conf:.0f}%)"
            elif name == "Untrained":
                box_color = (52, 152, 219)   # Blue
                text_bg = (41, 128, 185)
                badge_text = "Model Untrained"
            else:
                box_color = (60, 60, 235)    # Red / Coral
                text_bg = (40, 40, 190)
                badge_text = f"Unknown ({conf:.0f}%)"

            # Draw rounded or corner accent box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Draw corner accents for high-tech aesthetic
            corner_len = min(20, w // 4, h // 4)
            t = 3
            # Top-left
            cv2.line(annotated, (x, y), (x + corner_len, y), box_color, t)
            cv2.line(annotated, (x, y), (x, y + corner_len), box_color, t)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), box_color, t)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), box_color, t)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), box_color, t)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), box_color, t)
            # Bottom-right
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), box_color, t)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), box_color, t)

            # Badge background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)

            badge_y1 = max(0, y - text_h - 12)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = x + text_w + 16

            # If badge overflows top, put it inside box
            if y - text_h - 12 < 0:
                badge_y1 = y
                badge_y2 = y + text_h + 12

            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), text_bg, -1)
            cv2.putText(
                annotated,
                badge_text,
                (badge_x1 + 8, badge_y2 - 6),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        return annotated
