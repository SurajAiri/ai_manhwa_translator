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
        self.output_image = None
        self.translated_image = None
        self.debug_image = None
        self.extracted_texts = None
        
        # Keep track of current image view mode
        self.current_view = "original"
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(browse_button)
        
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
        
        self.submit_translation_button = QPushButton("Submit Translation")
        self.submit_translation_button.clicked.connect(self.submit_manual_translation)
        
        manual_layout.addWidget(QLabel("Translation Prompt:"))
        manual_layout.addWidget(self.prompt_text)
        manual_layout.addWidget(copy_prompt_button)
        manual_layout.addWidget(QLabel("Translation Response:"))
        manual_layout.addWidget(self.response_text)
        manual_layout.addWidget(self.submit_translation_button)
        
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
        self.log("Welcome to AI Manhwa Translator. Please select an image file to begin.")
        
        # Disable view toggle and zoom controls initially
        self.view_toggle_group.setEnabled(False)
        self.zoom_in_button.setEnabled(False)
        self.zoom_out_button.setEnabled(False)
        self.zoom_reset_button.setEnabled(False)
        
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
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image file first.")
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