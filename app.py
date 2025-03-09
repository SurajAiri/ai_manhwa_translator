import sys
import os
import cv2
import pyperclip
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.text_area import detect_bubbles_with_yolo
from src.crop_image import crop_text_regions
from src.ocr import extract_text_from_images
from src.translate import Translator
from src.text_overlay import overlay_text
from src.utility import parse_translation

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QTextEdit, QComboBox, QCheckBox, QScrollArea,
                            QMessageBox, QSplitter, QFrame, QButtonGroup,
                            QRadioButton, QGroupBox, QStackedWidget)



class TranslationWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, translation_model, is_debug):
        super().__init__()
        self.image_path = image_path
        self.translation_model = translation_model
        self.is_debug = is_debug
        self.translator = Translator()
        
    def run(self):
        try:
            self.progress.emit("Loading image...")
            output = cv2.imread(self.image_path)
            if output is None:
                self.error.emit("Failed to load image. Please check the file path.")
                return
                
            self.progress.emit("Detecting text bubbles...")
            output_debug, result = detect_bubbles_with_yolo(self.image_path)
            
            # Filter bubbles with confidence > 0.5
            bubbles = [bubble for bubble in result if bubble['confidence'] > 0.5]
            
            self.progress.emit(f"Found {len(bubbles)} text bubbles.")
            
            self.progress.emit("Cropping text regions...")
            cropped_texts = crop_text_regions(output, bubbles, is_debug=self.is_debug)
            
            self.progress.emit("Extracting text from images...")
            extracted_texts = extract_text_from_images(cropped_texts)
            
            if self.translation_model == "manual":
                self.progress.emit("Manual translation mode selected. Preparing translation prompt...")
                texts = [text_info['text'] for text_info in extracted_texts]
                prompt = self.translator.manual_translate_prompt(texts)
                
                # Return early with the necessary data for manual translation
                self.finished.emit({
                    "status": "manual_translation_needed",
                    "prompt": prompt,
                    "extracted_texts": extracted_texts,
                    "output": output,
                    "output_debug": output_debug if self.is_debug else None
                })
                return
            
            else:
                self.progress.emit(f"Translating text using {self.translation_model} model...")
                texts = [text_info['text'] for text_info in extracted_texts]
                translated_text = self.translator.translate_text(texts, model=self.translation_model)
                translated_texts = parse_translation(translated_text)
                
                if len(translated_texts) != len(extracted_texts):
                    self.error.emit("Error: Number of translated texts does not match the number of extracted texts.")
                    return
                
                # Update translated text
                for i, text_info in enumerate(extracted_texts):
                    text_info['translated_text'] = translated_texts[i]
                
                self.progress.emit("Overlaying translated text on image...")
                output_result = overlay_text(output.copy(), extracted_texts)
                
                self.finished.emit({
                    "status": "completed",
                    "output": output,
                    "output_result": output_result,
                    "output_debug": output_debug if self.is_debug else None,
                    "extracted_texts": extracted_texts
                })
        
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class BatchTranslationWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    image_processed = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths, translation_model, is_debug):
        super().__init__()
        self.image_paths = image_paths
        self.translation_model = translation_model
        self.is_debug = is_debug
        self.translator = Translator()
        self.stop_requested = False
        
    def run(self):
        results = []
        all_texts = []
        all_extracted_texts = []
        
        try:
            # If manual translation, collect all texts first
            if self.translation_model == "manual":
                self.progress.emit("Collecting text from all images for manual translation...")
                
                for i, image_path in enumerate(self.image_paths):
                    if self.stop_requested:
                        break
                        
                    self.progress.emit(f"Processing image {i+1}/{len(self.image_paths)}: {os.path.basename(image_path)}")
                    
                    # Process image
                    output = cv2.imread(image_path)
                    if output is None:
                        self.error.emit(f"Failed to load image: {image_path}")
                        continue
                    
                    # Detect bubbles
                    output_debug, result = detect_bubbles_with_yolo(image_path)
                    bubbles = [bubble for bubble in result if bubble['confidence'] > 0.5]
                    
                    # Crop and extract text
                    cropped_texts = crop_text_regions(output, bubbles, is_debug=self.is_debug)
                    extracted_texts = extract_text_from_images(cropped_texts)
                    
                    # Store results for this image
                    all_extracted_texts.append({
                        "path": image_path,
                        "extracted_texts": extracted_texts,
                        "output": output,
                        "output_debug": output_debug if self.is_debug else None
                    })
                    
                    # Collect texts for translation
                    image_texts = [text_info['text'] for text_info in extracted_texts]
                    all_texts.extend([f"[Image {i+1}, Text {j+1}]: {text}" for j, text in enumerate(image_texts)])
                
                # Prepare single translation prompt for all images
                prompt = self.translator.manual_translate_prompt(all_texts)
                
                # Return with data for manual translation
                self.finished.emit({
                    "status": "manual_translation_needed",
                    "prompt": prompt,
                    "all_extracted_texts": all_extracted_texts
                })
                
            else:  # Automatic translation for each image
                for i, image_path in enumerate(self.image_paths):
                    if self.stop_requested:
                        break
                        
                    self.progress.emit(f"Processing image {i+1}/{len(self.image_paths)}: {os.path.basename(image_path)}")
                    
                    # Process image
                    output = cv2.imread(image_path)
                    if output is None:
                        self.error.emit(f"Failed to load image: {image_path}")
                        continue
                    
                    # Detect bubbles
                    output_debug, result = detect_bubbles_with_yolo(image_path)
                    bubbles = [bubble for bubble in result if bubble['confidence'] > 0.5]
                    
                    # Crop and extract text
                    cropped_texts = crop_text_regions(output, bubbles, is_debug=self.is_debug)
                    extracted_texts = extract_text_from_images(cropped_texts)
                    
                    # Translate text
                    texts = [text_info['text'] for text_info in extracted_texts]
                    translated_text = self.translator.translate_text(texts, model=self.translation_model)
                    translated_texts = parse_translation(translated_text)
                    
                    if len(translated_texts) != len(extracted_texts):
                        self.error.emit(f"Error in image {i+1}: Number of translated texts doesn't match extracted texts.")
                        continue
                    
                    # Update translated text
                    for j, text_info in enumerate(extracted_texts):
                        text_info['translated_text'] = translated_texts[j]
                    
                    # Overlay translated text
                    output_result = overlay_text(output.copy(), extracted_texts)
                    
                    # Store result
                    result = {
                        "path": image_path,
                        "output": output,
                        "output_result": output_result,
                        "output_debug": output_debug if self.is_debug else None,
                        "extracted_texts": extracted_texts
                    }
                    results.append(result)
                    
                    # Emit signal for this processed image
                    self.image_processed.emit(result)
                
                self.finished.emit({
                    "status": "completed",
                    "results": results
                })
                
        except Exception as e:
            self.error.emit(f"Error in batch processing: {str(e)}")
    
    def stop(self):
        self.stop_requested = True


class ManhwaTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Manhwa Translator")
        self.setGeometry(100, 100, 1200, 800)
        
        self.image_path = None
        self.output_image = None
        self.translated_image = None
        self.debug_image = None
        self.extracted_texts = None
        
        # Batch processing variables
        self.batch_mode = False
        self.batch_directory = None
        self.batch_image_paths = []
        self.batch_results = []
        self.current_batch_index = -1
        
        # Keep track of current image view mode
        self.current_view = "original"
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Processing Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Image", "Batch Processing"])
        self.mode_combo.currentIndexChanged.connect(self.toggle_processing_mode)
        mode_layout.addWidget(self.mode_combo)
        left_panel.addLayout(mode_layout)
        
        # Create stacked widget for different mode controls
        self.mode_stack = QStackedWidget()
        
        # -------------------------------------------
        # Single Image Mode Controls
        # -------------------------------------------
        single_mode_widget = QWidget()
        single_layout = QVBoxLayout(single_mode_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(browse_button)
        
        # Add single mode controls
        single_layout.addLayout(file_layout)
        
        # -------------------------------------------
        # Batch Mode Controls
        # -------------------------------------------
        batch_mode_widget = QWidget()
        batch_layout = QVBoxLayout(batch_mode_widget)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_path_label = QLabel("No directory selected")
        self.dir_path_label.setWordWrap(True)
        browse_dir_button = QPushButton("Select Directory")
        browse_dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_path_label)
        dir_layout.addWidget(browse_dir_button)
        
        # Batch navigation controls
        batch_nav_layout = QHBoxLayout()
        self.prev_image_button = QPushButton("< Previous")
        self.prev_image_button.clicked.connect(self.show_previous_image)
        self.prev_image_button.setEnabled(False)
        
        self.next_image_button = QPushButton("Next >")
        self.next_image_button.clicked.connect(self.show_next_image)
        self.next_image_button.setEnabled(False)
        
        self.image_counter_label = QLabel("0/0")
        
        batch_nav_layout.addWidget(self.prev_image_button)
        batch_nav_layout.addWidget(self.image_counter_label)
        batch_nav_layout.addWidget(self.next_image_button)
        
        batch_layout.addLayout(dir_layout)
        batch_layout.addLayout(batch_nav_layout)
        
        # Add both widgets to the stack
        self.mode_stack.addWidget(single_mode_widget)
        self.mode_stack.addWidget(batch_mode_widget)
        
        # Add the stack to the layout
        left_panel.addWidget(self.mode_stack)
        
        # Common controls for both modes
        # Translation model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Translation Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["manual", "openai", "gemini"])
        model_layout.addWidget(self.model_combo)
        
        # Debug mode checkbox
        debug_layout = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Debug Mode")
        debug_layout.addWidget(self.debug_checkbox)
        
        # Translate button
        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self.start_translation)
        self.translate_button.setEnabled(False)
        
        # Save button
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        
        # Batch save button
        self.batch_save_all_button = QPushButton("Save All Results")
        self.batch_save_all_button.clicked.connect(self.save_all_batch_results)
        self.batch_save_all_button.setEnabled(False)
        
        # Progress log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Manual translation area
        self.manual_translation_group = QFrame()
        manual_layout = QVBoxLayout(self.manual_translation_group)
        
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        
        copy_prompt_button = QPushButton("Copy Prompt to Clipboard")
        copy_prompt_button.clicked.connect(self.copy_prompt)
        
        self.response_text = QTextEdit()
        self.response_text.setPlaceholderText("Paste translated text here...")
        
        self.submit_translation_button = QPushButton("Submit Translation")
        self.submit_translation_button.clicked.connect(self.submit_manual_translation)
        
        manual_layout.addWidget(QLabel("Translation Prompt:"))
        manual_layout.addWidget(self.prompt_text)
        manual_layout.addWidget(copy_prompt_button)
        manual_layout.addWidget(QLabel("Translation Response:"))
        manual_layout.addWidget(self.response_text)
        manual_layout.addWidget(self.submit_translation_button)
        
        self.manual_translation_group.setVisible(False)
        
        # Add common controls to left panel
        left_panel.addLayout(model_layout)
        left_panel.addLayout(debug_layout)
        left_panel.addWidget(self.translate_button)
        left_panel.addWidget(self.save_button)
        left_panel.addWidget(self.batch_save_all_button)
        left_panel.addWidget(QLabel("Log:"))
        left_panel.addWidget(self.log_text)
        left_panel.addWidget(self.manual_translation_group)
        
        # Hide batch save button initially
        self.batch_save_all_button.setVisible(False)
        
        # Right panel for image display
        right_panel = QVBoxLayout()
        
        # Add image view toggle controls
        self.view_toggle_group = QGroupBox("Image View Toggle")
        toggle_layout = QHBoxLayout()
        
        self.original_radio = QRadioButton("Original")
        self.original_radio.setChecked(True)
        self.original_radio.toggled.connect(self.toggle_image_view)
        
        self.debug_radio = QRadioButton("Debug View")
        self.debug_radio.toggled.connect(self.toggle_image_view)
        
        self.translated_radio = QRadioButton("Translated")
        self.translated_radio.toggled.connect(self.toggle_image_view)
        
        toggle_layout.addWidget(self.original_radio)
        toggle_layout.addWidget(self.debug_radio)
        toggle_layout.addWidget(self.translated_radio)
        self.view_toggle_group.setLayout(toggle_layout)
        
        # Add zoom controls
        zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(lambda: self.zoom_image(1.2))
        
        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(lambda: self.zoom_image(0.8))
        
        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(lambda: self.fit_image_to_width())
        
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_reset_button)
        
        # Add the zoom and toggle controls to the top of the right panel
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.view_toggle_group)
        controls_layout.addLayout(zoom_layout)
        right_panel.addLayout(controls_layout)
        
        # Create a single image display area with scroll capabilities
        self.image_display_label = QLabel()
        self.image_display_label.setAlignment(Qt.AlignCenter)
        
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidget(self.image_display_label)
        self.image_scroll_area.setWidgetResizable(True)
        
        # Store the original scale factor for zoom reset
        self.current_scale_factor = 1.0
        self.current_pixmap = None
        
        # Add the image display to the right panel
        right_panel.addWidget(self.image_scroll_area)
        
        # Create horizontal splitter for left and right panels
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([400, 800])
        
        # Set central widget
        container = QWidget()
        main_layout_container = QVBoxLayout(container)
        main_layout_container.addWidget(main_splitter)
        self.setCentralWidget(container)
        
        # Add log message
        self.log("Welcome to AI Manhwa Translator. Please select an image file or directory to begin.")
        
        # Disable view toggle and zoom controls initially
        self.view_toggle_group.setEnabled(False)
        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)
        self.zoom_reset_button.setEnabled(False)
        
    def toggle_processing_mode(self, index):
        """Switch between single image and batch processing mode"""
        self.batch_mode = (index == 1)
        self.mode_stack.setCurrentIndex(index)
        
        # Reset UI elements based on mode
        self.translate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Show/hide batch-specific controls
        self.batch_save_all_button.setVisible(self.batch_mode)
        self.batch_save_all_button.setEnabled(False)
        
        # Reset data
        self.image_path = None
        self.output_image = None
        self.translated_image = None
        self.debug_image = None
        self.extracted_texts = None
        
        if self.batch_mode:
            self.batch_directory = None
            self.batch_image_paths = []
            self.batch_results = []
            self.current_batch_index = -1
            self.image_counter_label.setText("0/0")
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
        
        # Reset image display
        self.image_display_label.clear()
        self.image_display_label.setText("No image loaded")
        
        # Reset view toggles
        self.view_toggle_group.setEnabled(False)
        self.original_radio.setChecked(True)
        self.debug_radio.setEnabled(False)
        self.translated_radio.setEnabled(False)
        
        # Log mode change
        self.log(f"Switched to {'batch processing' if self.batch_mode else 'single image'} mode")
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.file_path_label.setText(file_path)
            self.translate_button.setEnabled(True)
            
            # Load and display the original image
            self.output_image = cv2.imread(file_path)
            self.display_image(cv_img=self.output_image)
            self.current_view = "original"
            self.original_radio.setChecked(True)
            
            # Enable zoom controls
            self.zoom_in_button.setEnabled(True)
            self.zoom_out_button.setEnabled(True)
            self.zoom_reset_button.setEnabled(True)
            
            # Clear other images
            self.translated_image = None
            self.debug_image = None
            
            # Disable the translated view toggle
            self.translated_radio.setEnabled(False)
            self.debug_radio.setEnabled(False)
            self.view_toggle_group.setEnabled(True)
            
            self.save_button.setEnabled(False)
            
            self.log(f"Selected image: {file_path}")
    
    def browse_directory(self):
        """Select a directory containing images for batch processing"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Images")
        
        if directory:
            self.batch_directory = directory
            self.dir_path_label.setText(directory)
            
            # Find all image files in the directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            image_files = []
            
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, file))
            
            image_files.sort()  # Sort files alphabetically
            
            if not image_files:
                self.log("No image files found in the selected directory")
                return
            
            self.batch_image_paths = image_files
            self.log(f"Found {len(image_files)} images in directory")
            
            # Reset batch processing state
            self.batch_results = [None] * len(image_files)
            self.current_batch_index = 0
            self.image_counter_label.setText(f"1/{len(image_files)}")
            
            # Enable translate button
            self.translate_button.setEnabled(True)
            
            # Load and display the first image
            self.output_image = cv2.imread(image_files[0])
            self.display_image(cv_img=self.output_image)
            self.current_view = "original"
            self.original_radio.setChecked(True)
            
            # Enable navigation buttons
            self.update_navigation_buttons()
            
            # Enable zoom controls
            self.zoom_in_button.setEnabled(True)
            self.zoom_out_button.setEnabled(True)
            self.zoom_reset_button.setEnabled(True)
            
            # Enable view toggle
            self.view_toggle_group.setEnabled(True)
    
    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on current index"""
        if not self.batch_image_paths:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            return
            
        self.prev_image_button.setEnabled(self.current_batch_index > 0)
        self.next_image_button.setEnabled(self.current_batch_index < len(self.batch_image_paths) - 1)
        self.image_counter_label.setText(f"{self.current_batch_index + 1}/{len(self.batch_image_paths)}")
    
    def show_previous_image(self):
        """Navigate to the previous image in batch mode"""
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.load_batch_image_at_index()
            self.update_navigation_buttons()
    
    def show_next_image(self):
        """Navigate to the next image in batch mode"""
        if self.current_batch_index < len(self.batch_image_paths) - 1:
            self.current_batch_index += 1
            self.load_batch_image_at_index()
            self.update_navigation_buttons()
    
    def load_batch_image_at_index(self):
        """Load and display the batch image at the current index"""
        image_path = self.batch_image_paths[self.current_batch_index]
        
        # Check if this image has already been processed
        result = self.batch_results[self.current_batch_index]
        
        if result:
            # This image has been processed, load all its data
            self.output_image = result["output"]
            self.translated_image = result.get("output_result")
            self.debug_image = result.get("output_debug")
            self.extracted_texts = result.get("extracted_texts")
            
            # Enable appropriate view toggles
            self.debug_radio.setEnabled(self.debug_image is not None)
            self.translated_radio.setEnabled(self.translated_image is not None)
            
            # Show the translated image if available, otherwise show original
            if self.translated_image is not None and self.translated_radio.isChecked():
                self.display_image(cv_img=self.translated_image)
                self.current_view = "translated"
            elif self.debug_image is not None and self.debug_radio.isChecked():
                self.display_image(cv_img=self.debug_image)
                self.current_view = "debug"
            else:
                self.display_image(cv_img=self.output_image)
                self.current_view = "original"
                self.original_radio.setChecked(True)
                
            # Enable save button if we have a translated image
            self.save_button.setEnabled(self.translated_image is not None)
            
        else:
            # This image hasn't been processed yet, just show the original
            self.output_image = cv2.imread(image_path)
            self.translated_image = None
            self.debug_image = None
            self.extracted_texts = None
            
            self.display_image(cv_img=self.output_image)
            self.current_view = "original"
            self.original_radio.setChecked(True)
            
            # Disable view toggles that aren't applicable
            self.debug_radio.setEnabled(False)
            self.translated_radio.setEnabled(False)
            
            # Disable save button
            self.save_button.setEnabled(False)
    
    def display_image(self, image_path=None, cv_img=None):
        if image_path is None and cv_img is None:
            return
        
        if cv_img is not None:
            # Convert OpenCV image to QPixmap
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
        else:
            # Load image from path
            pixmap = QPixmap(image_path)
        
        # Store the current pixmap for zoom operations
        self.current_pixmap = pixmap
        self.current_scale_factor = 1.0
        
        # Set the pixmap to the display label
        self.image_display_label.setPixmap(pixmap)
        self.image_display_label.setMinimumSize(1, 1)  # Allow the image to scale down
        
        # Automatically fit image to the width of the scroll area
        self.fit_image_to_width()

    def fit_image_to_width(self):
        """Adjust zoom factor to fit the image to the scroll area width."""
        if self.current_pixmap and not self.current_pixmap.isNull():
            # Get the current viewport width
            viewport_width = self.image_scroll_area.viewport().width()
            
            # Calculate the scale factor needed to fit the image to the viewport width
            if self.current_pixmap.width() > 0:  # Avoid division by zero
                scale_factor = viewport_width / self.current_pixmap.width()
                self.current_scale_factor = scale_factor
                
                # Apply the zoom with the calculated scale factor
                new_size = self.current_pixmap.size() * self.current_scale_factor
                scaled_pixmap = self.current_pixmap.scaled(
                    new_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # Update the displayed image
                self.image_display_label.setPixmap(scaled_pixmap)
                self.image_display_label.setMinimumSize(scaled_pixmap.size())
        
    def zoom_image(self, factor=1.0, reset=False):
        if self.current_pixmap is None:
            return
        
        if reset:
            self.current_scale_factor = 1.0
        else:
            self.current_scale_factor *= factor
        
        # Limit the scale factor to reasonable bounds
        self.current_scale_factor = max(0.1, min(5.0, self.current_scale_factor))
        
        # Calculate the new size while maintaining aspect ratio
        new_size = self.current_pixmap.size() * self.current_scale_factor
        
        # Create a scaled pixmap
        scaled_pixmap = self.current_pixmap.scaled(
            new_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Update the displayed image
        self.image_display_label.setPixmap(scaled_pixmap)
        
        # Ensure the label can accommodate the scaled image
        self.image_display_label.setMinimumSize(scaled_pixmap.size())

    def toggle_image_view(self):
        if self.original_radio.isChecked() and self.output_image is not None:
            self.display_image(cv_img=self.output_image)
            self.current_view = "original"
        elif self.debug_radio.isChecked() and self.debug_image is not None:
            self.display_image(cv_img=self.debug_image)
            self.current_view = "debug"
        elif self.translated_radio.isChecked() and self.translated_image is not None:
            self.display_image(cv_img=self.translated_image)
            self.current_view = "translated"
    
    def log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def start_translation(self):
        """Start translation process based on current mode"""
        # Hide manual translation fields
        self.manual_translation_group.setVisible(False)
        self.prompt_text.clear()
        self.response_text.clear()
        
        # Get translation settings
        translation_model = self.model_combo.currentText()
        is_debug = self.debug_checkbox.isChecked()
        
        if self.batch_mode:
            # Batch processing
            if not self.batch_image_paths:
                QMessageBox.warning(self, "Error", "Please select a directory with images first.")
                return
            
            # Disable UI elements during batch processing
            self.translate_button.setEnabled(False)
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            
            # Start batch translation worker
            self.batch_worker = BatchTranslationWorker(
                self.batch_image_paths, 
                translation_model, 
                is_debug
            )
            self.batch_worker.progress.connect(self.log)
            self.batch_worker.error.connect(self.handle_error)
            self.batch_worker.image_processed.connect(self.handle_single_image_processed)
            self.batch_worker.finished.connect(self.handle_batch_translation_finished)
            self.batch_worker.start()
            
        else:
            # Single image processing (existing implementation)
            if not self.image_path:
                QMessageBox.warning(self, "Error", "Please select an image file first.")
                return
                
            # Disable toggle buttons until translation is complete
            self.debug_radio.setEnabled(False)
            self.translated_radio.setEnabled(False)
            
            # Disable translate button during processing
            self.translate_button.setEnabled(False)
            
            # Start translation worker thread
            self.worker = TranslationWorker(self.image_path, translation_model, is_debug)
            self.worker.progress.connect(self.log)
            self.worker.error.connect(self.handle_error)
            self.worker.finished.connect(self.handle_translation_finished)
            self.worker.start()
    
    def handle_error(self, error_message):
        self.log(f"ERROR: {error_message}")
        self.translate_button.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)
    
    def handle_translation_finished(self, result):
        if result["status"] == "manual_translation_needed":
            self.log("Manual translation required. Please copy the prompt, translate it, and paste the result.")
            
            # Store extracted texts for later
            self.extracted_texts = result["extracted_texts"]
            self.output_image = result["output"]
            
            # Show debug image if available
            if result["output_debug"] is not None:
                self.debug_image = result["output_debug"]
                self.debug_radio.setEnabled(True)
                
                # Switch to debug view automatically if in debug mode
                if self.debug_checkbox.isChecked():
                    self.debug_radio.setChecked(True)
                    self.display_image(cv_img=self.debug_image)
                    self.current_view = "debug"
            
            # Show manual translation interface
            self.prompt_text.setPlainText(result["prompt"])
            self.manual_translation_group.setVisible(True)
            
        elif result["status"] == "completed":
            self.log("Translation completed successfully!")
            
            # Store extracted texts and images
            self.extracted_texts = result["extracted_texts"]
            self.output_image = result["output"]
            self.translated_image = result["output_result"]
            
            # Enable respective view toggles
            if result["output_debug"] is not None:
                self.debug_image = result["output_debug"]
                self.debug_radio.setEnabled(True)
            
            self.translated_radio.setEnabled(True)
            
            # Auto-switch to translated view
            self.translated_radio.setChecked(True)
            self.display_image(cv_img=self.translated_image)
            self.current_view = "translated"
            
            # Enable save button
            self.save_button.setEnabled(True)
        
        # Re-enable translate button
        self.translate_button.setEnabled(True)
    
    def handle_single_image_processed(self, result):
        """Handle the completion of a single image in batch mode"""
        # Find the index of this image
        try:
            index = self.batch_image_paths.index(result["path"])
            self.batch_results[index] = result
            
            # If this is the current image being displayed, update the view
            if index == self.current_batch_index:
                self.load_batch_image_at_index()
                
        except ValueError:
            self.log(f"Warning: Processed image not found in batch list: {result['path']}")
    
    def handle_batch_translation_finished(self, result):
        """Handle completion of batch processing"""
        self.translate_button.setEnabled(True)
        
        if result["status"] == "manual_translation_needed":
            self.log("Manual translation required for batch. Please copy the prompt, translate it, and paste the result.")
            
            # Store all extracted texts for later
            self.all_extracted_texts = result["all_extracted_texts"]
            
            # Show manual translation interface
            self.prompt_text.setPlainText(result["prompt"])
            self.manual_translation_group.setVisible(True)
            
        elif result["status"] == "completed":
            self.log("Batch translation completed successfully!")
            self.batch_save_all_button.setEnabled(True)
            
        # Re-enable navigation buttons
        self.update_navigation_buttons()
    
    def copy_prompt(self):
        prompt = self.prompt_text.toPlainText()
        pyperclip.copy(prompt)
        self.log("Prompt copied to clipboard")
    
    def submit_manual_translation(self):
        """Process manual translation for single image or batch"""
        response = self.response_text.toPlainText().strip()
        if not response:
            QMessageBox.warning(self, "Error", "Please paste the translated text first.")
            return
        
        try:
            if self.batch_mode:
                # Process batch manual translation
                translated_texts = parse_translation(response)
                
                # Count total texts
                total_texts = 0
                for item in self.all_extracted_texts:
                    total_texts += len(item["extracted_texts"])
                
                if len(translated_texts) != total_texts:
                    raise ValueError(f"Number of translated texts ({len(translated_texts)}) does not match the number of extracted texts ({total_texts}).")
                
                # Distribute translations to their respective images
                current_index = 0
                for i, item in enumerate(self.all_extracted_texts):
                    num_texts = len(item["extracted_texts"])
                    
                    # Get translations for this image
                    image_translations = translated_texts[current_index:current_index+num_texts]
                    current_index += num_texts
                    
                    # Update extracted texts with translations
                    for j, text_info in enumerate(item["extracted_texts"]):
                        text_info['translated_text'] = image_translations[j]
                    
                    # Overlay translated text
                    output_result = overlay_text(item["output"].copy(), item["extracted_texts"])
                    
                    # Store result
                    result = {
                        "path": item["path"],
                        "output": item["output"],
                        "output_result": output_result,
                        "output_debug": item["output_debug"],
                        "extracted_texts": item["extracted_texts"]
                    }
                    
                    # Find index of this image
                    try:
                        index = self.batch_image_paths.index(item["path"])
                        self.batch_results[index] = result
                    except ValueError:
                        self.log(f"Warning: Could not find batch index for: {item['path']}")
                
                # Enable save all button
                self.batch_save_all_button.setEnabled(True)
                
                # Update current image view
                self.load_batch_image_at_index()
                
                # Hide manual translation interface
                self.manual_translation_group.setVisible(False)
                
                self.log("Batch translation completed successfully!")
                
            else:
                # Process single image manual translation (existing code)
                translated_texts = parse_translation(response)
                
                if len(translated_texts) != len(self.extracted_texts):
                    raise ValueError(f"Number of translated texts ({len(translated_texts)}) does not match the number of extracted texts ({len(self.extracted_texts)}).")
                
                # Update translated text
                for i, text_info in enumerate(self.extracted_texts):
                    text_info['translated_text'] = translated_texts[i]
                
                self.log("Manual translation processed. Overlaying text on image...")
                
                # Overlay translated text
                self.translated_image = overlay_text(self.output_image.copy(), self.extracted_texts)
                
                # Enable translated view toggle
                self.translated_radio.setEnabled(True)
                
                # Auto-switch to translated view
                self.translated_radio.setChecked(True)
                self.display_image(cv_img=self.translated_image)
                self.current_view = "translated"
                
                # Enable save button
                self.save_button.setEnabled(True)
                
                # Hide manual translation interface
                self.manual_translation_group.setVisible(False)
                
                self.log("Translation completed successfully!")
                
        except Exception as e:
            self.log(f"Error processing translation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process translation: {str(e)}")
    
    def save_result(self):
        if self.translated_image is None:
            QMessageBox.warning(self, "Error", "No translated image available to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Translated Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            # Ensure file has proper extension
            if not (file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg')):
                file_path += '.png'
                
            # Save the image
            cv2.imwrite(file_path, self.translated_image)
            self.log(f"Translated image saved to: {file_path}")

    def save_all_batch_results(self):
        """Save all translated images in batch mode"""
        if not self.batch_results or not any(result is not None for result in self.batch_results):
            QMessageBox.warning(self, "Error", "No translated images available to save.")
            return
        
        # Ask for destination directory
        destination_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not destination_dir:
            return
            
        # Count successful saves
        saved_count = 0
        
        # Process each image
        for i, result in enumerate(self.batch_results):
            if result is None or "output_result" not in result:
                continue  # Skip images without translations
                
            # Generate output filename
            base_name = os.path.basename(result["path"])
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(destination_dir, f"{name}_translated.png")
            
            # Save the image
            try:
                cv2.imwrite(output_path, result["output_result"])
                saved_count += 1
            except Exception as e:
                self.log(f"Error saving {base_name}: {str(e)}")
        
        self.log(f"Saved {saved_count} translated images to: {destination_dir}")
        if saved_count > 0:
            QMessageBox.information(self, "Batch Save Complete", f"Successfully saved {saved_count} translated images.")

    def resizeEvent(self, event):
        """Handle window resize events to adjust image scaling."""
        super().resizeEvent(event)
        # Allow some time for the UI to update before fitting the image
        # Using a small delay ensures the scroll area has been resized properly
        QApplication.processEvents()
        self.fit_image_to_width()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManhwaTranslatorApp()
    window.show()
    sys.exit(app.exec_())