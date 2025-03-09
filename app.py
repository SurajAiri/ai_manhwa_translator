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
                            QRadioButton, QGroupBox)



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


class ManhwaTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Manhwa Translator")
        self.setGeometry(100, 100, 1200, 800)
        
        self.image_path = None
        self.image_directory = None
        self.image_files = []
        self.current_image_index = 0
        self.output_image = None
        self.translated_image = None
        self.debug_image = None
        self.extracted_texts = None
        self.multi_image_results = {}  # Store results for batch processing
        
        # Keep track of current image view mode
        self.current_view = "original"
        
        # Store worker threads to prevent premature destruction
        self.worker_threads = []
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # File selection
        file_layout = QVBoxLayout()
        
        select_label = QLabel("Select Image Source:")
        file_layout.addWidget(select_label)
        
        # Add radio buttons for single file vs directory
        source_layout = QHBoxLayout()
        self.source_group = QButtonGroup(self)
        self.single_file_radio = QRadioButton("Single Image")
        self.directory_radio = QRadioButton("Image Directory")
        self.single_file_radio.setChecked(True)
        self.source_group.addButton(self.single_file_radio)
        self.source_group.addButton(self.directory_radio)
        source_layout.addWidget(self.single_file_radio)
        source_layout.addWidget(self.directory_radio)
        file_layout.addLayout(source_layout)
        
        # File selection controls
        file_selection_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_source)
        file_selection_layout.addWidget(self.file_path_label)
        file_selection_layout.addWidget(browse_button)
        file_layout.addLayout(file_selection_layout)
        
        # Add to left panel
        left_panel.addLayout(file_layout)
        
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
        
        # Button layout for manual translation actions
        manual_buttons_layout = QHBoxLayout()
        
        self.submit_translation_button = QPushButton("Submit Translation")
        self.submit_translation_button.clicked.connect(self.submit_manual_translation)
        
        self.skip_translation_button = QPushButton("Skip This Image")
        self.skip_translation_button.clicked.connect(self.skip_manual_translation)
        
        manual_buttons_layout.addWidget(self.submit_translation_button)
        manual_buttons_layout.addWidget(self.skip_translation_button)
        
        manual_layout.addWidget(QLabel("Translation Prompt:"))
        manual_layout.addWidget(self.prompt_text)
        manual_layout.addWidget(copy_prompt_button)
        manual_layout.addWidget(QLabel("Translation Response:"))
        manual_layout.addWidget(self.response_text)
        manual_layout.addLayout(manual_buttons_layout)
        
        self.manual_translation_group.setVisible(False)
        
        # Add all controls to left panel
        left_panel.addLayout(file_layout)
        left_panel.addLayout(model_layout)
        left_panel.addLayout(debug_layout)
        left_panel.addWidget(self.translate_button)
        left_panel.addWidget(self.save_button)
        left_panel.addWidget(QLabel("Log:"))
        left_panel.addWidget(self.log_text)
        left_panel.addWidget(self.manual_translation_group)
        
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
        
        # Image navigation controls
        self.navigation_group = QFrame()
        nav_layout = QHBoxLayout(self.navigation_group)
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        
        self.image_counter_label = QLabel("Image 1/1")
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.image_counter_label)
        nav_layout.addWidget(self.next_button)
        
        self.navigation_group.setVisible(False)
        right_panel.addWidget(self.navigation_group)
        
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
        
        # Add multi-image view toggle
        self.multi_view_checkbox = QCheckBox("Show All Images in Sequence")
        self.multi_view_checkbox.toggled.connect(self.toggle_multi_image_view)
        controls_layout.addWidget(self.multi_view_checkbox)
        
        # Multi-image display area
        self.multi_image_scroll_area = QScrollArea()
        self.multi_image_container = QWidget()
        self.multi_image_layout = QVBoxLayout(self.multi_image_container)
        self.multi_image_scroll_area.setWidget(self.multi_image_container)
        self.multi_image_scroll_area.setWidgetResizable(True)
        self.multi_image_scroll_area.setVisible(False)
        
        right_panel.addWidget(self.multi_image_scroll_area)
        
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
        self.log("Welcome to AI Manhwa Translator. Please select an image file to begin.")
        
        # Disable view toggle and zoom controls initially
        self.view_toggle_group.setEnabled(False)
        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)
        self.zoom_reset_button.setEnabled(False)
        
    def browse_source(self):
        """Browse for either a single file or a directory based on selection"""
        if self.single_file_radio.isChecked():
            self.browse_file()
        else:
            self.browse_directory()
        
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.image_directory = None
            self.image_files = []
            self.current_image_index = 0
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
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Images")
        if directory:
            # Look for image files in the directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            image_files = []
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, file))
            
            if not image_files:
                QMessageBox.warning(self, "No Images Found", "No image files found in the selected directory.")
                return
                
            # Sort files by name to ensure proper order
            image_files.sort()
            
            self.image_directory = directory
            self.image_files = image_files
            self.current_image_index = 0
            self.image_path = image_files[0]  # Set first image as current path
            
            self.file_path_label.setText(f"Directory: {directory} ({len(image_files)} images)")
            self.translate_button.setEnabled(True)
            
            # Load and display the first image
            self.output_image = cv2.imread(self.image_path)
            self.display_image(cv_img=self.output_image)
            self.current_view = "original"
            self.original_radio.setChecked(True)
            
            # Enable zoom and navigation controls
            self.zoom_in_button.setEnabled(True)
            self.zoom_out_button.setEnabled(True)
            self.zoom_reset_button.setEnabled(True)
            
            # Add navigation buttons
            self.update_navigation_buttons_state()
            
            # Clear other images
            self.translated_image = None
            self.debug_image = None
            self.multi_image_results = {}
            
            # Disable the translated view toggle
            self.translated_radio.setEnabled(False)
            self.debug_radio.setEnabled(False)
            self.view_toggle_group.setEnabled(True)
            
            self.save_button.setEnabled(False)
            
            self.log(f"Selected directory: {directory} with {len(image_files)} images")
    
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
        if self.original_radio.isChecked():
            self.current_view = "original"
        elif self.debug_radio.isChecked() and self.debug_image is not None:
            self.current_view = "debug"
        elif self.translated_radio.isChecked() and self.translated_image is not None:
            self.current_view = "translated"
        
        # Update display based on current mode
        if self.multi_view_checkbox.isChecked():
            self.generate_multi_image_view()
        else:
            # Single image view
            if self.current_view == "original" and self.output_image is not None:
                self.display_image(cv_img=self.output_image)
            elif self.current_view == "debug" and self.debug_image is not None:
                self.display_image(cv_img=self.debug_image)
            elif self.current_view == "translated" and self.translated_image is not None:
                self.display_image(cv_img=self.translated_image)
    
    def log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def start_translation(self):
        if not self.image_path and not self.image_directory:
            QMessageBox.warning(self, "Error", "Please select an image file or directory first.")
            return
        
        # Reset and hide manual translation fields
        self.manual_translation_group.setVisible(False)
        self.prompt_text.clear()
        self.response_text.clear()
        
        # Disable toggle buttons until translation is complete
        self.debug_radio.setEnabled(False)
        self.translated_radio.setEnabled(False)
        
        # Disable translate button during processing
        self.translate_button.setEnabled(False)
        
        # Get translation settings
        translation_model = self.model_combo.currentText()
        is_debug = self.debug_checkbox.isChecked()
        
        # Check if we're processing a single file or a directory
        if self.single_file_radio.isChecked() or not self.image_directory:
            # Process single file
            self.process_single_image(self.image_path, translation_model, is_debug)
        else:
            # Process all files in directory
            self.process_image_directory(translation_model, is_debug)

    def process_single_image(self, image_path, translation_model, is_debug):
        """Process a single image file"""
        # Start translation worker thread
        self.worker = TranslationWorker(image_path, translation_model, is_debug)
        self.worker.progress.connect(self.log)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.handle_translation_finished)
        
        # Store worker in the list to prevent garbage collection
        self.worker_threads.append(self.worker)
        self.worker.finished.connect(lambda: self.cleanup_worker(self.worker))
        
        self.worker.start()

    def process_image_directory(self, translation_model, is_debug):
        """Process all images in the directory"""
        # Create a dialog to confirm batch processing
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Batch Processing")
        msg_box.setText(f"Process all {len(self.image_files)} images in directory?")
        msg_box.setInformativeText("This may take a while depending on the number of images.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        
        if msg_box.exec_() == QMessageBox.Yes:
            # Start processing the first image
            self.current_batch_index = 0
            self.batch_translation_model = translation_model
            self.batch_debug_mode = is_debug
            self.batch_manual_mode = (translation_model == "manual")
            self.batch_pending_manual = []  # Store indices of images waiting for manual translation
            
            # Begin the batch process
            self.process_next_batch_image()
        else:
            # Re-enable translate button
            self.translate_button.setEnabled(True)

    def process_next_batch_image(self):
        """Process the next image in the batch"""
        if self.current_batch_index < len(self.image_files):
            image_path = self.image_files[self.current_batch_index]
            self.log(f"Processing image {self.current_batch_index + 1}/{len(self.image_files)}: {os.path.basename(image_path)}")
            
            # Create a worker for this image
            self.worker = TranslationWorker(image_path, self.batch_translation_model, self.batch_debug_mode)
            self.worker.progress.connect(lambda msg: self.log(f"[{os.path.basename(image_path)}] {msg}"))
            self.worker.error.connect(self.handle_batch_error)
            self.worker.finished.connect(self.handle_batch_translation_finished)
            
            # Store worker in the list to prevent garbage collection
            self.worker_threads.append(self.worker)
            self.worker.finished.connect(lambda: self.cleanup_worker(self.worker))
            
            self.worker.start()
        else:
            # All images processed - check if we have pending manual translations
            if self.batch_manual_mode and self.batch_pending_manual:
                self.log(f"Batch processing completed. {len(self.batch_pending_manual)} images need manual translation.")
                
                # Show dialog to start manual translation process
                QMessageBox.information(
                    self, 
                    "Manual Translation Required", 
                    f"{len(self.batch_pending_manual)} images need manual translation.\n\n"
                    "You will now be guided through each image that requires translation."
                )
                
                # Start the manual translation process for the first image
                self.start_batch_manual_translation()
            else:
                # All done - no manual translations needed or not in manual mode
                self.log(f"Batch processing completed: {self.current_batch_index} images processed")
                self.translate_button.setEnabled(True)
                self.save_button.setEnabled(True)

    def cleanup_worker(self, worker):
        """Remove completed worker from the list to allow garbage collection"""
        if worker in self.worker_threads:
            self.worker_threads.remove(worker)

    def handle_batch_error(self, error_message):
        """Handle errors during batch processing"""
        image_path = self.image_files[self.current_batch_index]
        self.log(f"ERROR processing {os.path.basename(image_path)}: {error_message}")
        
        # Move to next image
        self.current_batch_index += 1
        self.process_next_batch_image()

    def handle_batch_translation_finished(self, result):
        """Handle completion of a single image in batch processing"""
        image_path = self.image_files[self.current_batch_index]
        
        # Store results for this image
        if result["status"] == "completed":
            # Handle completed automatic translation
            self.multi_image_results[image_path] = {
                'output': result["output"],
                'translated_image': result["output_result"],
                'debug_image': result.get("output_debug"),
                'extracted_texts': result["extracted_texts"]
            }
            self.log(f"Completed translation of {os.path.basename(image_path)}")
            
            # If this is the current displayed image, update our references
            if self.current_image_index == self.current_batch_index:
                self.output_image = result["output"]
                self.translated_image = result["output_result"]
                if result.get("output_debug") is not None:
                    self.debug_image = result["output_debug"]
                
                # Enable view toggles
                self.debug_radio.setEnabled(self.debug_image is not None)
                self.translated_radio.setEnabled(True)
                
                # Update view if we're in multi-image view mode
                if self.multi_view_checkbox.isChecked():
                    self.generate_multi_image_view()
                # Otherwise, update single view if this is the current image
                else:
                    self.translated_radio.setChecked(True)
                    self.display_image(cv_img=self.translated_image)
                    self.current_view = "translated"
                    
            # Process next image
            self.current_batch_index += 1
            self.process_next_batch_image()
            
        elif result["status"] == "manual_translation_needed":
            # For manual translation in batch mode, we store the data and move on to the next image
            self.log(f"Image {self.current_batch_index + 1} ({os.path.basename(image_path)}) needs manual translation (will be processed later)")
            
            # Store the data needed for manual translation
            self.multi_image_results[image_path] = {
                'output': result["output"],
                'debug_image': result.get("output_debug"),
                'extracted_texts': result["extracted_texts"],
                'prompt': result["prompt"],
                'needs_manual': True
            }
            
            # Add to list of images needing manual translation
            self.batch_pending_manual.append(self.current_batch_index)
            
            # Process next image
            self.current_batch_index += 1
            self.process_next_batch_image()
    
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
    
    def copy_prompt(self):
        prompt = self.prompt_text.toPlainText()
        pyperclip.copy(prompt)
        self.log("Prompt copied to clipboard")
    
    def submit_manual_translation(self):
        response = self.response_text.toPlainText().strip()
        if not response:
            QMessageBox.warning(self, "Error", "Please paste the translated text first.")
            return
        
        try:
            # Parse the translation
            translated_texts = parse_translation(response)
            
            if len(translated_texts) != len(self.extracted_texts):
                raise ValueError(f"Number of translated texts ({len(translated_texts)}) does not match the number of extracted texts ({len(self.extracted_texts)}).")
            
            # Update translated text
            for i, text_info in enumerate(self.extracted_texts):
                text_info['translated_text'] = translated_texts[i]
            
            self.log("Manual translation processed. Overlaying text on image...")
            
            # Overlay translated text
            self.translated_image = overlay_text(self.output_image.copy(), self.extracted_texts)
            
            # Store this result in multi_image_results
            if self.image_path:
                # Update the stored result
                self.multi_image_results[self.image_path].update({
                    'translated_image': self.translated_image,
                    'needs_manual': False,
                    'extracted_texts': self.extracted_texts
                })
            
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
            self.response_text.clear()
            
            self.log("Translation completed successfully!")
            
            # Check if we're in batch manual mode
            if hasattr(self, 'batch_pending_manual') and self.batch_pending_manual:
                # Remove the current image from pending list
                self.batch_pending_manual.pop(0)
                
                if self.batch_pending_manual:
                    # Show a message about moving to next image
                    QMessageBox.information(
                        self, 
                        "Next Image", 
                        f"Moving to the next image requiring manual translation.\n\n"
                        f"{len(self.batch_pending_manual)} images remaining."
                    )
                    
                    # Process the next image
                    self.start_batch_manual_translation()
                else:
                    # All done with manual translations
                    QMessageBox.information(
                        self, 
                        "Batch Translation Complete", 
                        "All manual translations have been completed successfully!"
                    )
                    
        
        except Exception as e:
            self.log(f"Error processing translation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process translation: {str(e)}")
    
    def save_result(self):
        # Check if we're working with a directory or single file
        if self.image_directory and len(self.multi_image_results) > 1:
            # Batch save dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Save Translated Images")
            msg_box.setText("Save all processed images?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            
            if msg_box.exec_() == QMessageBox.Yes:
                # Ask for output directory
                output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if output_dir:
                    self.save_batch_results(output_dir)
                    return
        
        # Single image save
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

    def save_batch_results(self, output_dir):
        """Save all translated images to the output directory"""
        saved_count = 0
        
        for image_path, result in self.multi_image_results.items():
            if 'translated_image' in result:
                # Create output filename
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"translated_{filename}")
                
                # Save image
                cv2.imwrite(output_path, result['translated_image'])
                saved_count += 1
        
        self.log(f"Batch save completed: {saved_count} images saved to {output_dir}")
        QMessageBox.information(self, "Batch Save", f"{saved_count} images were saved successfully.")

    def resizeEvent(self, event):
        """Handle window resize events to adjust image scaling."""
        super().resizeEvent(event)
        # Allow some time for the UI to update before fitting the image
        # Using a small delay ensures the scroll area has been resized properly
        QApplication.processEvents()
        self.fit_image_to_width()

    def update_navigation_buttons_state(self):
        """Update the state of navigation buttons based on current index"""
        if not self.image_files:
            self.navigation_group.setVisible(False)
            return
            
        self.navigation_group.setVisible(True)
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_files) - 1)
        self.image_counter_label.setText(f"Image {self.current_image_index + 1}/{len(self.image_files)}")

    def show_previous_image(self):
        """Show previous image in the directory"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def show_next_image(self):
        """Show next image in the directory"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def load_current_image(self):
        """Load and display the current image based on the index"""
        if not self.image_files:
            return
            
        self.image_path = self.image_files[self.current_image_index]
        
        # Update navigation buttons
        self.update_navigation_buttons_state()
        
        # Check if we already processed this image
        if self.image_path in self.multi_image_results:
            result = self.multi_image_results[self.image_path]
            self.output_image = result.get('output')
            self.translated_image = result.get('translated_image')
            self.debug_image = result.get('debug_image')
            
            # Enable appropriate view buttons
            self.translated_radio.setEnabled(self.translated_image is not None)
            self.debug_radio.setEnabled(self.debug_image is not None)
        else:
            # Load unprocessed image
            self.output_image = cv2.imread(self.image_path)
            self.translated_image = None
            self.debug_image = None
            
            # Disable view buttons for unprocessed images
            self.translated_radio.setEnabled(False)
            self.debug_radio.setEnabled(False)
        
        # Display the image based on current view preference
        if self.current_view == "translated" and self.translated_image is not None:
            self.display_image(cv_img=self.translated_image)
        elif self.current_view == "debug" and self.debug_image is not None:
            self.display_image(cv_img=self.debug_image)
        else:
            self.display_image(cv_img=self.output_image)
            self.original_radio.setChecked(True)
            
        self.log(f"Showing image {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(self.image_path)}")

    def toggle_multi_image_view(self, checked):
        """Toggle between single image view and multi-image sequence view"""
        if not self.image_files:
            self.multi_view_checkbox.setChecked(False)
            return
            
        if checked:
            # Switch to multi-image view
            self.image_scroll_area.setVisible(False)
            self.navigation_group.setVisible(False)
            self.multi_image_scroll_area.setVisible(True)
            
            # Generate the multi-image view
            self.generate_multi_image_view()
        else:
            # Switch back to single image view
            self.multi_image_scroll_area.setVisible(False)
            self.image_scroll_area.setVisible(True)
            self.navigation_group.setVisible(len(self.image_files) > 1)
            
            # Make sure current image is displayed
            self.load_current_image()

    def generate_multi_image_view(self):
        """Generate a sequential view of all images"""
        # Clear existing widgets
        while self.multi_image_layout.count():
            item = self.multi_image_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Determine which view to show (original, translated, debug)
        view_type = self.current_view
        
        # Add each image to the layout
        for index, image_path in enumerate(self.image_files):
            try:
                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                frame_layout = QVBoxLayout(frame)
                
                # Image title
                title_label = QLabel(f"Image {index + 1}: {os.path.basename(image_path)}")
                title_label.setAlignment(Qt.AlignCenter)
                frame_layout.addWidget(title_label)
                
                # Image display
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                
                # Get the appropriate image based on view type
                cv_img = None
                if image_path in self.multi_image_results:
                    result = self.multi_image_results[image_path]
                    if view_type == "translated" and result.get('translated_image') is not None:
                        cv_img = result['translated_image']
                    elif view_type == "debug" and result.get('debug_image') is not None:
                        cv_img = result['debug_image']
                    else:
                        cv_img = result.get('output')
                
                # If no processed image is found, load from file
                if cv_img is None:
                    cv_img = cv2.imread(image_path)
                    
                if cv_img is None:
                    # Handle case where image couldn't be loaded
                    error_label = QLabel("Error loading image")
                    error_label.setStyleSheet("color: red;")
                    frame_layout.addWidget(error_label)
                    self.multi_image_layout.addWidget(frame)
                    continue
                
                # Create pixmap
                height, width, channel = cv_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                # Scale to fit width
                max_width = self.multi_image_scroll_area.viewport().width() - 40  # Add padding
                scale_factor = min(1.0, max_width / width) if width > 0 else 1.0
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    int(width * scale_factor), 
                    int(height * scale_factor),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                image_label.setPixmap(scaled_pixmap)
                frame_layout.addWidget(image_label)
                
                # Add to container
                self.multi_image_layout.addWidget(frame)
            
            except Exception as e:
                # Add error info for this image
                error_frame = QFrame()
                error_frame.setFrameShape(QFrame.StyledPanel)
                error_layout = QVBoxLayout(error_frame)
                
                title_label = QLabel(f"Image {index + 1}: {os.path.basename(image_path)}")
                title_label.setAlignment(Qt.AlignCenter)
                
                error_label = QLabel(f"Error: {str(e)}")
                error_label.setStyleSheet("color: red;")
                error_label.setWordWrap(True)
                
                error_layout.addWidget(title_label)
                error_layout.addWidget(error_label)
                
                self.multi_image_layout.addWidget(error_frame)
        
        # Add stretch at the end to keep images aligned at the top
        self.multi_image_layout.addStretch()

    def closeEvent(self, event):
        """Handle application closing properly"""
        # Check if we're in the middle of manual batch translation
        if hasattr(self, 'batch_pending_manual') and self.batch_pending_manual:
            reply = QMessageBox.question(
                self, 
                "Exit Application", 
                "Manual batch translation is in progress.\nDo you want to quit anyway?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Terminate any running worker threads
        for worker in self.worker_threads:
            if worker.isRunning():
                worker.quit()
                worker.wait(500)  # Wait up to 500ms for threads to finish
        
        super().closeEvent(event)

    def reset_ui_state(self):
        """Reset UI to a clean state for new processing"""
        # Hide manual translation panel
        self.manual_translation_group.setVisible(False)
        self.prompt_text.clear()
        self.response_text.clear()
        
        # Reset image view toggles
        self.original_radio.setChecked(True)
        
        # Re-enable buttons
        self.translate_button.setEnabled(True)
        
        # Clear any existing worker threads
        for worker in self.worker_threads:
            if worker.isRunning():
                worker.quit()
                worker.wait(500)  # Wait up to 500ms for threads to finish
        
        self.worker_threads = []

    def start_batch_manual_translation(self):
        """Start the manual translation process for the next image in the batch."""
        if not self.batch_pending_manual:
            # No more images to manually translate
            self.log("All manual translations completed!")
            self.translate_button.setEnabled(True)
            self.save_button.setEnabled(True)
            return
        
        # Get the next image index that needs manual translation
        index = self.batch_pending_manual[0]
        image_path = self.image_files[index]
        
        # Set as current image
        self.current_image_index = index
        self.image_path = image_path
        
        # Load the image and its data
        result = self.multi_image_results[image_path]
        self.output_image = result['output']
        self.debug_image = result.get('debug_image')
        self.extracted_texts = result['extracted_texts']
        
        # Show the image
        self.display_image(cv_img=self.output_image)
        self.current_view = "original"
        self.original_radio.setChecked(True)
        
        # Enable debug view if available
        if self.debug_image is not None:
            self.debug_radio.setEnabled(True)
        else:
            self.debug_radio.setEnabled(False)
        
        self.translated_radio.setEnabled(False)
        
        # Show manual translation interface
        self.prompt_text.setPlainText(result['prompt'])
        self.manual_translation_group.setVisible(True)
        
        # Update navigation state
        self.update_navigation_buttons_state()
        
        # Update log
        self.log(f"Manual translation required for image {index + 1}/{len(self.image_files)}: {os.path.basename(image_path)}")

    def skip_manual_translation(self):
        """Skip the current manual translation in batch mode"""
        if not hasattr(self, 'batch_pending_manual') or not self.batch_pending_manual:
            # Not in batch mode, just close the manual translation panel
            self.manual_translation_group.setVisible(False)
            self.prompt_text.clear()
            self.response_text.clear()
            self.translate_button.setEnabled(True)
            return
            
        reply = QMessageBox.question(
            self, 
            "Skip Translation", 
            "Are you sure you want to skip translating this image?\nIt will remain untranslated.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove the current image from pending list
            current_index = self.batch_pending_manual.pop(0)
            image_path = self.image_files[current_index]
            self.log(f"Skipped translation for image: {os.path.basename(image_path)}")
            
            # Clear manual translation interface
            self.manual_translation_group.setVisible(False)
            self.prompt_text.clear()
            self.response_text.clear()
            
            if self.batch_pending_manual:
                # Move to the next image
                self.start_batch_manual_translation()
            else:
                # All done with manual translations
                QMessageBox.information(
                    self, 
                    "Batch Translation Complete", 
                    "All manual translations have been processed (some were skipped)."
                )
                self.translate_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManhwaTranslatorApp()
    window.show()
    sys.exit(app.exec_())