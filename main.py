"""
main.py
Main entry point for Face Recognition Studio
"""

import sys
import tkinter as tk
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import cv2
    import PIL
    import numpy
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements using: pip install -r requirements.txt")
    sys.exit(1)

from src.gui import FaceRecognitionApp


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
