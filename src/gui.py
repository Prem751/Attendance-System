"""
gui.py
Modern Tkinter Graphical User Interface for Face Recognition with OpenCV
"""

import os
import sys
import time
import threading
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from src.face_engine import FaceEngine, safe_imread, safe_imwrite


class FaceRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition Studio - OpenCV & Tkinter")
        self.root.geometry("1100x750")
        self.root.minsize(950, 650)

        # High-DPI awareness on Windows
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.engine = FaceEngine()

        # Video stream states
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.camera_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[Any] = None
        self.last_results: List[Dict[str, Any]] = []

        # History log
        self.history_records: List[Dict[str, Any]] = []

        # Styling
        self.setup_styles()

        # Build UI layout
        self.build_ui()

        # Bind window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initial refresh
        self.refresh_dataset_tree()
        self.update_model_status_card()

    def setup_styles(self):
        self.style = ttk.Style(self.root)
        # Use clam or default
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Palette colors
        self.bg_main = "#f4f6f9"
        self.header_bg = "#1f2937"
        self.header_fg = "#ffffff"
        self.accent_color = "#3b82f6"
        self.accent_hover = "#2563eb"
        self.success_color = "#10b981"
        self.danger_color = "#ef4444"

        self.root.configure(bg=self.bg_main)

        self.style.configure(".", background=self.bg_main, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=self.bg_main)
        self.style.configure("Header.TFrame", background=self.header_bg)

        self.style.configure(
            "TNotebook",
            background=self.bg_main,
            borderwidth=0
        )
        self.style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=[16, 8],
            background="#e5e7eb",
            foreground="#374151"
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff")],
            foreground=[("selected", "#1d4ed8")]
        )

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.accent_color,
            foreground="#ffffff",
            padding=[12, 6]
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", self.accent_hover)]
        )

        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.success_color,
            foreground="#ffffff",
            padding=[12, 6]
        )

        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 9),
            background=self.danger_color,
            foreground="#ffffff",
            padding=[10, 4]
        )

        self.style.configure(
            "Card.TFrame",
            background="#ffffff",
            relief="solid",
            borderwidth=1
        )

        self.style.configure(
            "Treeview",
            font=("Segoe UI", 10),
            rowheight=26,
            background="#ffffff",
            fieldbackground="#ffffff"
        )
        self.style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background="#f3f4f6",
            foreground="#111827"
        )

    def build_ui(self):
        # 1. Top Header Banner
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 14))
        header_frame.pack(fill="x", side="top")

        title_lbl = tk.Label(
            header_frame,
            text="👤 Face Recognition Studio",
            font=("Segoe UI", 16, "bold"),
            fg=self.header_fg,
            bg=self.header_bg
        )
        title_lbl.pack(side="left")

        subtitle_lbl = tk.Label(
            header_frame,
            text="OpenCV & LBPH Engine | Recognize from Saved Images & Live Stream",
            font=("Segoe UI", 9),
            fg="#9ca3af",
            bg=self.header_bg
        )
        subtitle_lbl.pack(side="left", padx=16, pady=4)

        self.model_status_badge = tk.Label(
            header_frame,
            text="● Checking Model...",
            font=("Segoe UI", 10, "bold"),
            fg="#fbbf24",
            bg=self.header_bg
        )
        self.model_status_badge.pack(side="right")

        # 2. Main Tab Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=10)

        # Tab 1: Recognition (Live Webcam & Image File)
        self.tab_recognition = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_recognition, text="  🔍 Recognize Faces  ")
        self.build_recognition_tab()

        # Tab 2: Dataset & Enrollment (From Saved Images)
        self.tab_dataset = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_dataset, text="  📁 Saved Images & Enrollment  ")
        self.build_dataset_tab()

        # Tab 3: Model & Training
        self.tab_model = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_model, text="  ⚙️ Model & Training  ")
        self.build_model_tab()

        # Tab 4: History Logs
        self.tab_logs = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_logs, text="  📜 Recognition History  ")
        self.build_logs_tab()

        # Bottom status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready. Select an option to start.",
            bd=1,
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 9),
            bg="#e5e7eb",
            fg="#374151",
            padx=10,
            pady=4
        )
        self.status_bar.pack(side="bottom", fill="x")

    # =========================================================================
    # TAB 1: FACE RECOGNITION (Live Webcam & Static Images)
    # =========================================================================
    def build_recognition_tab(self):
        # Sub-frames: Left (Controls + details), Right (Display Canvas)
        container = ttk.Frame(self.tab_recognition)
        container.pack(fill="both", expand=True)

        left_panel = ttk.Frame(container, width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        right_panel = ttk.Frame(container)
        right_panel.pack(side="right", fill="both", expand=True)

        # Mode Selection Card
        mode_card = ttk.LabelFrame(left_panel, text=" Recognition Source ", padding=10)
        mode_card.pack(fill="x", pady=(0, 10))

        self.rec_mode_var = tk.StringVar(value="webcam")

        ttk.Radiobutton(
            mode_card,
            text="Live Webcam Stream",
            variable=self.rec_mode_var,
            value="webcam",
            command=self.on_rec_mode_change
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            mode_card,
            text="Image File from Disk",
            variable=self.rec_mode_var,
            value="image",
            command=self.on_rec_mode_change
        ).pack(anchor="w", pady=2)

        # Webcam Controls Frame
        self.webcam_ctrl_frame = ttk.LabelFrame(left_panel, text=" Webcam Controls ", padding=10)
        self.webcam_ctrl_frame.pack(fill="x", pady=(0, 10))

        cam_idx_frame = ttk.Frame(self.webcam_ctrl_frame)
        cam_idx_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(cam_idx_frame, text="Camera Index:").pack(side="left")
        self.cam_index_spin = ttk.Spinbox(cam_idx_frame, from_=0, to=4, width=5)
        self.cam_index_spin.set(0)
        self.cam_index_spin.pack(side="right")

        self.btn_toggle_camera = ttk.Button(
            self.webcam_ctrl_frame,
            text="▶ Start Camera",
            style="Success.TButton",
            command=self.toggle_camera
        )
        self.btn_toggle_camera.pack(fill="x", pady=4)

        self.btn_snapshot = ttk.Button(
            self.webcam_ctrl_frame,
            text="📷 Take Snapshot",
            command=self.take_snapshot
        )
        self.btn_snapshot.pack(fill="x", pady=4)

        # Image File Controls Frame
        self.image_ctrl_frame = ttk.LabelFrame(left_panel, text=" Image File Controls ", padding=10)
        # Hidden by default

        self.btn_select_test_img = ttk.Button(
            self.image_ctrl_frame,
            text="📂 Select Test Image...",
            style="Primary.TButton",
            command=self.select_and_recognize_image
        )
        self.btn_select_test_img.pack(fill="x", pady=4)

        self.btn_load_sample = ttk.Button(
            self.image_ctrl_frame,
            text="🎯 Load Sample Face Image",
            command=self.load_sample_face
        )
        self.btn_load_sample.pack(fill="x", pady=4)

        # Detection Details Card
        details_card = ttk.LabelFrame(left_panel, text=" Detected Faces ", padding=8)
        details_card.pack(fill="both", expand=True)

        self.face_listbox = tk.Listbox(
            details_card,
            font=("Segoe UI", 9),
            height=10,
            bg="#ffffff",
            selectbackground=self.accent_color
        )
        self.face_listbox.pack(fill="both", expand=True, pady=(0, 6))

        # Right Panel: Video / Image Display Canvas
        display_frame = ttk.Frame(right_panel, relief="solid", borderwidth=1)
        display_frame.pack(fill="both", expand=True)

        self.display_canvas = tk.Canvas(display_frame, bg="#111827", highlightthickness=0)
        self.display_canvas.pack(fill="both", expand=True)

        # Initial canvas placeholder
        self.display_canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas_text = self.display_canvas.create_text(
            350, 250,
            text="Camera stream or image will be displayed here\nClick 'Start Camera' or switch to 'Image File'",
            fill="#9ca3af",
            font=("Segoe UI", 12),
            justify="center"
        )

    def on_rec_mode_change(self):
        mode = self.rec_mode_var.get()
        if mode == "webcam":
            self.image_ctrl_frame.pack_forget()
            self.webcam_ctrl_frame.pack(fill="x", pady=(0, 10))
        else:
            if self.is_camera_running:
                self.stop_camera()
            self.webcam_ctrl_frame.pack_forget()
            self.image_ctrl_frame.pack(fill="x", pady=(0, 10))

    def on_canvas_resize(self, event):
        if not self.is_camera_running and self.current_frame is None:
            self.display_canvas.coords(self.canvas_text, event.width // 2, event.height // 2)

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not self.engine.is_trained():
            if not messagebox.askyesno(
                "Model Not Trained",
                "The face recognition model has not been trained yet with saved images.\n"
                "Detected faces will show as 'Untrained'.\n\nDo you want to start the camera anyway?"
            ):
                return

        try:
            cam_idx = int(self.cam_index_spin.get())
        except ValueError:
            cam_idx = 0

        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {cam_idx}.\nPlease check your webcam connection.")
            return

        self.is_camera_running = True
        self.btn_toggle_camera.config(text="⏹ Stop Camera", style="Danger.TButton")
        self.set_status("Webcam active. Recognizing faces in real time...")

        self.camera_loop()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.btn_toggle_camera.config(text="▶ Start Camera", style="Success.TButton")
        self.set_status("Camera stopped.")

    def camera_loop(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Mirror horizontally for natural webcam experience
            frame = cv2.flip(frame, 1)

            # Recognize faces
            results = self.engine.recognize_frame(frame)
            self.last_results = results
            self.current_frame = frame.copy()

            # Draw detections
            annotated = self.engine.draw_detections(frame, results)

            # Update details list
            self.update_face_listbox(results, source="Webcam")

            # Display on canvas
            self.render_frame_to_canvas(annotated)

        # Schedule next frame (~30 FPS)
        if self.is_camera_running:
            self.root.after(30, self.camera_loop)

    def render_frame_to_canvas(self, bgr_image: Any):
        canvas_w = self.display_canvas.winfo_width()
        canvas_h = self.display_canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            return

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb_image.shape[:2]

        # Aspect-fit resizing
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))

        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)
        self.photo_img = ImageTk.PhotoImage(pil_img)

        self.display_canvas.delete("all")
        # Center in canvas
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.display_canvas.create_image(x, y, anchor="nw", image=self.photo_img)

    def update_face_listbox(self, results: List[Dict[str, Any]], source: str = "Webcam"):
        self.face_listbox.delete(0, tk.END)
        if not results:
            self.face_listbox.insert(tk.END, "No faces detected")
            return

        for i, res in enumerate(results, 1):
            name = res["name"]
            conf = res["confidence"]
            status = "MATCH" if res["is_known"] else "UNKNOWN"
            item_text = f"#{i}: {name} ({conf:.0f}%) [{status}]"
            self.face_listbox.insert(tk.END, item_text)

            # Log periodically to history if confident known person
            if res["is_known"]:
                self.record_history(name, conf, source)

    def record_history(self, name: str, confidence: float, source: str):
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        # Throttle duplicate entries within 5 seconds for the same person
        if self.history_records:
            last = self.history_records[-1]
            if last["name"] == name and (time.time() - last["timestamp_raw"]) < 5.0:
                return

        rec = {
            "time": now_str,
            "name": name,
            "confidence": f"{confidence:.1f}%",
            "source": source,
            "timestamp_raw": time.time()
        }
        self.history_records.append(rec)
        self.tree_logs.insert("", 0, values=(rec["time"], rec["name"], rec["confidence"], rec["source"]))

    def take_snapshot(self):
        if self.current_frame is None:
            messagebox.showinfo("Snapshot", "No active camera frame to capture.")
            return

        snap_dir = self.engine.base_dir / "logs" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = snap_dir / filename

        annotated = self.engine.draw_detections(self.current_frame, self.last_results)
        safe_imwrite(str(save_path), annotated)
        messagebox.showinfo("Snapshot Saved", f"Snapshot saved to:\n{save_path}")

    def select_and_recognize_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Face Image to Recognize",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self.process_static_image(file_path)

    def load_sample_face(self):
        sample_path = self.engine.base_dir / "test_samples" / "sample_lena.jpg"
        if not sample_path.exists():
            messagebox.showinfo("Sample", "Sample image not found in test_samples/.")
            return
        self.process_static_image(str(sample_path))

    def process_static_image(self, file_path: str):
        img = safe_imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image:\n{file_path}")
            return

        self.current_frame = img.copy()
        results = self.engine.recognize_frame(img)
        self.last_results = results

        annotated = self.engine.draw_detections(img, results)
        self.render_frame_to_canvas(annotated)
        self.update_face_listbox(results, source=f"File: {Path(file_path).name}")

        msg = f"Processed {Path(file_path).name}: Found {len(results)} face(s)."
        self.set_status(msg)

    # =========================================================================
    # TAB 2: DATASET & ENROLLMENT (From Saved Images)
    # =========================================================================
    def build_dataset_tab(self):
        container = ttk.Frame(self.tab_dataset)
        container.pack(fill="both", expand=True)

        # Left Column: Enroll New Person Form
        enroll_card = ttk.LabelFrame(container, text=" Enroll Person from Saved Images ", padding=14)
        enroll_card.pack(side="left", fill="both", expand=False, padx=(0, 10))
        enroll_card.configure(width=420)

        desc_lbl = tk.Label(
            enroll_card,
            text="Add a person by selecting their previously saved photos\n(or folder of images). The engine extracts face regions\nand adds them to the training dataset.",
            font=("Segoe UI", 9),
            justify="left",
            fg="#4b5563",
            bg=self.bg_main
        )
        desc_lbl.pack(anchor="w", pady=(0, 12))

        # Name Entry
        ttk.Label(enroll_card, text="Person Name / Label:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.entry_person_name = ttk.Entry(enroll_card, font=("Segoe UI", 11), width=32)
        self.entry_person_name.pack(fill="x", pady=(4, 12))
        self.entry_person_name.insert(0, "Person_1")

        # Source Selection Buttons
        sep1 = ttk.Separator(enroll_card, orient="horizontal")
        sep1.pack(fill="x", pady=6)

        ttk.Label(enroll_card, text="Select Image Source:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 6))

        btn_folder = ttk.Button(
            enroll_card,
            text="📁 Select Folder of Saved Photos...",
            style="Primary.TButton",
            command=self.enroll_from_folder_dialog
        )
        btn_folder.pack(fill="x", pady=4)

        btn_files = ttk.Button(
            enroll_card,
            text="🖼️ Select Specific Image Files...",
            command=self.enroll_from_files_dialog
        )
        btn_files.pack(fill="x", pady=4)

        btn_webcam_enroll = ttk.Button(
            enroll_card,
            text="🎥 Capture Photos via Webcam (Burst)",
            command=self.enroll_via_webcam_burst
        )
        btn_webcam_enroll.pack(fill="x", pady=4)

        # Progress indicator
        self.enroll_progress = ttk.Progressbar(enroll_card, mode="determinate")
        self.enroll_progress.pack(fill="x", pady=(16, 6))

        self.enroll_status_lbl = tk.Label(
            enroll_card,
            text="Ready to enroll.",
            font=("Segoe UI", 9),
            fg="#374151",
            bg=self.bg_main
        )
        self.enroll_status_lbl.pack(anchor="w")

        # Right Column: Existing Enrolled Persons Management Table
        manage_card = ttk.LabelFrame(container, text=" Enrolled Persons in Dataset ", padding=10)
        manage_card.pack(side="right", fill="both", expand=True)

        # Treeview for enrolled persons
        columns = ("name", "count", "status")
        self.dataset_tree = ttk.Treeview(manage_card, columns=columns, show="headings", selectmode="browse")
        self.dataset_tree.heading("name", text="Person Name")
        self.dataset_tree.heading("count", text="Saved Face Samples")
        self.dataset_tree.heading("status", text="Model Status")

        self.dataset_tree.column("name", width=160, anchor="w")
        self.dataset_tree.column("count", width=130, anchor="center")
        self.dataset_tree.column("status", width=120, anchor="center")

        tree_scroll = ttk.Scrollbar(manage_card, orient="vertical", command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=tree_scroll.set)

        self.dataset_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # Bottom toolbar for Dataset Table
        tree_toolbar = ttk.Frame(manage_card)
        tree_toolbar.pack(side="bottom", fill="x", pady=(8, 0))

        btn_del_person = ttk.Button(
            tree_toolbar,
            text="🗑️ Delete Selected Person",
            style="Danger.TButton",
            command=self.delete_selected_person
        )
        btn_del_person.pack(side="left", padx=4)

        btn_open_folder = ttk.Button(
            tree_toolbar,
            text="📂 Open Dataset Folder",
            command=self.open_dataset_folder
        )
        btn_open_folder.pack(side="left", padx=4)

        btn_refresh = ttk.Button(
            tree_toolbar,
            text="🔄 Refresh List",
            command=self.refresh_dataset_tree
        )
        btn_refresh.pack(side="right", padx=4)

    def refresh_dataset_tree(self):
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)

        persons = self.engine.get_registered_persons()
        for p in persons:
            status = "Trained" if p["is_in_model"] else "Pending Training"
            self.dataset_tree.insert("", "end", values=(p["name"], f"{p['count']} photos", status))

        self.update_model_status_card()

    def enroll_from_folder_dialog(self):
        person_name = self.entry_person_name.get().strip()
        if not person_name:
            messagebox.showwarning("Name Required", "Please enter a Person Name / Label before selecting photos.")
            return

        folder = filedialog.askdirectory(title=f"Select Folder with Saved Photos for '{person_name}'")
        if not folder:
            return

        def task():
            self.enroll_status_lbl.config(text="Processing images and extracting faces...", fg="#2563eb")
            self.enroll_progress.config(mode="indeterminate")
            self.enroll_progress.start(10)

            extracted, total = self.engine.enroll_from_folder(person_name, folder)

            self.enroll_progress.stop()
            self.enroll_progress.config(mode="determinate", value=100)

            msg = f"Done! Extracted {extracted} face sample(s) from {total} images in folder."
            self.enroll_status_lbl.config(text=msg, fg="#10b981")
            self.refresh_dataset_tree()

            if extracted > 0:
                if messagebox.askyesno("Train Model Now?", f"{extracted} faces added for '{person_name}'.\nWould you like to train the model now?"):
                    self.run_training()

        threading.Thread(target=task, daemon=True).start()

    def enroll_from_files_dialog(self):
        person_name = self.entry_person_name.get().strip()
        if not person_name:
            messagebox.showwarning("Name Required", "Please enter a Person Name / Label before selecting photos.")
            return

        file_paths = filedialog.askopenfilenames(
            title=f"Select Saved Photo Files for '{person_name}'",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")]
        )
        if not file_paths:
            return

        def task():
            self.enroll_status_lbl.config(text=f"Extracting faces from {len(file_paths)} files...", fg="#2563eb")
            self.enroll_progress.config(mode="determinate", maximum=len(file_paths), value=0)

            extracted, total = self.engine.enroll_from_images(person_name, list(file_paths))

            self.enroll_progress.config(value=total)
            msg = f"Done! Extracted {extracted} face(s) from {total} files."
            self.enroll_status_lbl.config(text=msg, fg="#10b981")
            self.refresh_dataset_tree()

            if extracted > 0:
                if messagebox.askyesno("Train Model Now?", f"{extracted} faces added for '{person_name}'.\nWould you like to train the model now?"):
                    self.run_training()

        threading.Thread(target=task, daemon=True).start()

    def enroll_via_webcam_burst(self):
        person_name = self.entry_person_name.get().strip()
        if not person_name:
            messagebox.showwarning("Name Required", "Please enter a Person Name / Label first.")
            return

        try:
            cam_idx = int(self.cam_index_spin.get())
        except Exception:
            cam_idx = 0

        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {cam_idx}.")
            return

        burst_window = tk.Toplevel(self.root)
        burst_window.title(f"Webcam Burst Capture for '{person_name}'")
        burst_window.geometry("640x540")
        burst_window.transient(self.root)
        burst_window.grab_set()

        lbl_instruct = tk.Label(
            burst_window,
            text=f"Capturing face samples for: {person_name}\nLook directly at the camera, then turn slightly left/right.",
            font=("Segoe UI", 10, "bold"),
            pady=8
        )
        lbl_instruct.pack()

        burst_canvas = tk.Canvas(burst_window, width=580, height=380, bg="#111827")
        burst_canvas.pack(pady=4)

        lbl_count = tk.Label(burst_window, text="Samples Captured: 0 / 25", font=("Segoe UI", 11, "bold"), fg="#1d4ed8")
        lbl_count.pack(pady=6)

        captured_count = [0]
        max_samples = 25
        is_active = [True]

        def stop_burst():
            is_active[0] = False
            cap.release()
            burst_window.destroy()
            self.refresh_dataset_tree()
            if captured_count[0] >= 5:
                if messagebox.askyesno("Train Model Now?", f"Captured {captured_count[0]} face samples.\nRetrain model now?"):
                    self.run_training()

        btn_finish = ttk.Button(burst_window, text="Finish & Close", style="Primary.TButton", command=stop_burst)
        btn_finish.pack(pady=6)

        def loop():
            if not is_active[0]:
                return

            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                boxes = self.engine.detect_faces(frame)

                if boxes and captured_count[0] < max_samples:
                    # Capture largest face every few ticks
                    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
                    x, y, w, h = boxes[0]
                    face_roi = self.engine.extract_face_roi(frame, boxes[0])
                    self.engine.save_single_face_sample(person_name, face_roi)
                    captured_count[0] += 1
                    lbl_count.config(text=f"Samples Captured: {captured_count[0]} / {max_samples}")

                    # Highlight green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 204, 113), 2)
                elif boxes:
                    x, y, w, h = boxes[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

                # Render to canvas
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (580, 380))
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                burst_canvas.create_image(0, 0, anchor="nw", image=photo)
                burst_canvas.image = photo

                if captured_count[0] >= max_samples:
                    lbl_instruct.config(text=f"All {max_samples} samples collected successfully!", fg="#10b981")

            burst_window.after(80, loop)

        loop()

    def delete_selected_person(self):
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showinfo("Select Person", "Please select a person from the table first.")
            return

        values = self.dataset_tree.item(selected[0], "values")
        person_name = values[0]

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete all saved photos for '{person_name}'?"):
            self.engine.delete_person(person_name)
            self.refresh_dataset_tree()
            self.set_status(f"Deleted '{person_name}' from dataset.")
            if messagebox.askyesno("Retrain Model?", "Dataset changed. Retrain model now?"):
                self.run_training()

    def open_dataset_folder(self):
        folder = self.engine.dataset_dir
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(folder)])

    # =========================================================================
    # TAB 3: MODEL & TRAINING
    # =========================================================================
    def build_model_tab(self):
        container = ttk.Frame(self.tab_model)
        container.pack(fill="both", expand=True)

        left_card = ttk.LabelFrame(container, text=" Model Overview & Actions ", padding=14)
        left_card.pack(side="left", fill="both", expand=False, padx=(0, 10))
        left_card.configure(width=380)

        # Status lines
        self.lbl_trained_state = tk.Label(
            left_card,
            text="Model Status: Checking...",
            font=("Segoe UI", 11, "bold"),
            fg="#dc2626",
            bg=self.bg_main
        )
        self.lbl_trained_state.pack(anchor="w", pady=(0, 6))

        self.lbl_trained_time = tk.Label(left_card, text="Last Trained: Never", font=("Segoe UI", 9), fg="#4b5563", bg=self.bg_main)
        self.lbl_trained_time.pack(anchor="w", pady=2)

        self.lbl_trained_persons = tk.Label(left_card, text="Trained Persons: 0", font=("Segoe UI", 9), fg="#4b5563", bg=self.bg_main)
        self.lbl_trained_persons.pack(anchor="w", pady=2)

        self.lbl_trained_samples = tk.Label(left_card, text="Total Face Samples: 0", font=("Segoe UI", 9), fg="#4b5563", bg=self.bg_main)
        self.lbl_trained_samples.pack(anchor="w", pady=2)

        sep = ttk.Separator(left_card, orient="horizontal")
        sep.pack(fill="x", pady=14)

        # Confidence Threshold Slider
        ttk.Label(left_card, text="Recognition Sensitivity (LBPH Threshold):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_thresh_val = tk.Label(left_card, text=f"Threshold: {self.engine.confidence_threshold:.0f} (Balanced)", font=("Segoe UI", 9), fg="#2563eb", bg=self.bg_main)
        self.lbl_thresh_val.pack(anchor="w", pady=(2, 4))

        self.thresh_slider = ttk.Scale(
            left_card,
            from_=45.0,
            to=95.0,
            value=self.engine.confidence_threshold,
            command=self.on_thresh_change
        )
        self.thresh_slider.pack(fill="x", pady=(0, 12))

        tk.Label(
            left_card,
            text="Lower = Stricter (avoids false matches)\nHigher = Lenient (recognizes in difficult angles)",
            font=("Segoe UI", 8),
            fg="#6b7280",
            bg=self.bg_main,
            justify="left"
        ).pack(anchor="w", pady=(0, 14))

        # Train Button
        self.btn_train = ttk.Button(
            left_card,
            text="⚡ Train / Retrain Model Now",
            style="Primary.TButton",
            command=self.run_training
        )
        self.btn_train.pack(fill="x", pady=6)

        # Right Card: Real-time Training Console Log
        console_card = ttk.LabelFrame(container, text=" Training & Engine Console Log ", padding=10)
        console_card.pack(side="right", fill="both", expand=True)

        self.txt_console = tk.Text(
            console_card,
            bg="#111827",
            fg="#e5e7eb",
            font=("Consolas", 10),
            relief="flat",
            wrap="word"
        )
        scroll = ttk.Scrollbar(console_card, orient="vertical", command=self.txt_console.yview)
        self.txt_console.configure(yscrollcommand=scroll.set)

        self.txt_console.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.log_console("Face Recognition Studio initialized.")
        self.log_console(f"Base Directory: {self.engine.base_dir}")
        self.log_console(f"Haar Cascade: {self.engine.cascade_path.name}")

    def on_thresh_change(self, val):
        val_float = float(val)
        self.engine.confidence_threshold = val_float
        desc = "Strict" if val_float < 65 else ("Lenient" if val_float > 80 else "Balanced")
        self.lbl_thresh_val.config(text=f"Threshold: {val_float:.0f} ({desc})")

    def log_console(self, msg: str):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.txt_console.insert(tk.END, timestamp + msg + "\n")
        self.txt_console.see(tk.END)

    def update_model_status_card(self):
        if self.engine.is_trained():
            self.model_status_badge.config(text="● Model: Trained & Ready", fg="#34d399")
            self.lbl_trained_state.config(text="Model Status: Trained & Ready", fg="#10b981")
            self.lbl_trained_persons.config(text=f"Trained Persons: {len(self.engine.label_map)}")

            # Read metadata from labels.json if available
            if self.engine.labels_file.exists():
                try:
                    import json
                    with open(self.engine.labels_file, "r") as f:
                        meta = json.load(f)
                    self.lbl_trained_time.config(text=f"Last Trained: {meta.get('trained_at', 'Unknown')}")
                    self.lbl_trained_samples.config(text=f"Total Face Samples: {meta.get('total_samples', 'Unknown')}")
                except Exception:
                    pass
        else:
            self.model_status_badge.config(text="● Model: Untrained", fg="#f87171")
            self.lbl_trained_state.config(text="Model Status: Untrained", fg="#ef4444")
            self.lbl_trained_time.config(text="Last Trained: Never")
            self.lbl_trained_persons.config(text="Trained Persons: 0")
            self.lbl_trained_samples.config(text="Total Face Samples: 0")

    def run_training(self):
        def task():
            self.log_console("Starting LBPH face recognition model training...")
            self.set_status("Training model on saved images...")
            self.btn_train.config(state="disabled")

            success, message = self.engine.train_model()

            self.btn_train.config(state="normal")
            self.log_console(f"Training Result: {message}")

            if success:
                self.update_model_status_card()
                self.refresh_dataset_tree()
                self.set_status("Model trained successfully!")
                messagebox.showinfo("Training Success", message)
            else:
                self.set_status(f"Training failed: {message}")
                messagebox.showwarning("Training Notice", message)

        threading.Thread(target=task, daemon=True).start()

    # =========================================================================
    # TAB 4: RECOGNITION LOGS & HISTORY
    # =========================================================================
    def build_logs_tab(self):
        container = ttk.Frame(self.tab_logs)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x", pady=(0, 8))

        btn_export = ttk.Button(
            toolbar,
            text="📥 Export History to CSV",
            style="Primary.TButton",
            command=self.export_history_csv
        )
        btn_export.pack(side="left", padx=(0, 6))

        btn_clear = ttk.Button(
            toolbar,
            text="🧹 Clear History",
            command=self.clear_history
        )
        btn_clear.pack(side="left", padx=6)

        # Logs Table
        columns = ("time", "name", "confidence", "source")
        self.tree_logs = ttk.Treeview(container, columns=columns, show="headings")
        self.tree_logs.heading("time", text="Timestamp")
        self.tree_logs.heading("name", text="Recognized Person")
        self.tree_logs.heading("confidence", text="Confidence Score")
        self.tree_logs.heading("source", text="Detection Source")

        self.tree_logs.column("time", width=160, anchor="center")
        self.tree_logs.column("name", width=180, anchor="w")
        self.tree_logs.column("confidence", width=120, anchor="center")
        self.tree_logs.column("source", width=240, anchor="w")

        scroll = ttk.Scrollbar(container, orient="vertical", command=self.tree_logs.yview)
        self.tree_logs.configure(yscrollcommand=scroll.set)

        self.tree_logs.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def export_history_csv(self):
        if not self.history_records:
            messagebox.showinfo("Export", "No history records to export.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"recognition_history_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Recognized Person", "Confidence Score", "Detection Source"])
                for rec in self.history_records:
                    writer.writerow([rec["time"], rec["name"], rec["confidence"], rec["source"]])
            messagebox.showinfo("Exported", f"History successfully exported to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}")

    def clear_history(self):
        self.history_records.clear()
        for item in self.tree_logs.get_children():
            self.tree_logs.delete(item)
        self.set_status("Recognition history cleared.")

    def set_status(self, text: str):
        self.status_bar.config(text=f" {text}")

    def on_close(self):
        if self.is_camera_running:
            self.stop_camera()
        self.root.destroy()
