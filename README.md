# Bolt Classification and Detection System Using YOLOv8 and ONNX

## Description
This project implements a bolt classification and detection system for industrial quality control, utilizing YOLOv8 and ONNX models. It processes images to detect bolts, classify them as "NORM" or "TVS" using a combination of ONNX-based classification, YOLOv8 classification, and YOLOv8 segmentation for enhanced accuracy. The system applies image preprocessing techniques like resizing, normalization, and brightness enhancement to improve detection performance. A majority voting mechanism ensures robust classification by combining results from multiple models and augmentation steps. Integrated with FastAPI, the system provides a RESTful API endpoint to handle base64-encoded images, returning detection results, including class labels, total bolt count, and status ("OK" or "NG"). Non-conforming images are saved for further analysis, supporting quality control workflows. The system is optimized for CPU inference, with comprehensive error handling to ensure reliability in production environments.

## Features
- **Object Detection**: Detects bolts in images using an ONNX model with high confidence thresholds.
- **Classification**: Classifies bolts as "NORM" or "TVS" using ONNX, YOLOv8, and YOLOv8 segmentation models.
- **Image Augmentation**: Enhances images with brightness adjustments for improved classification accuracy.
- **Majority Voting**: Combines results from multiple models to ensure robust classification.
- **API Integration**: Provides a FastAPI endpoint for processing base64-encoded images.
- **Quality Control**: Saves non-conforming images for further analysis.

## Tech Stack
- **Languages**: Python
- **Libraries**: YOLOv8, ONNX, FastAPI, OpenCV, PyTorch, PIL, NumPy
- **Tools**: REST API, CPU Inference, Image Processing

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rkarahul/Bolt-Classification-Detection-System.git
   cd Bolt-Classification-Detection-System
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Place the model files (`gear_bolt.onnx`, `bolt_classifier_2025_old.onnx`, `best.pt`, `Segbest.pt`, `crop.names`) in the `models/` directory.

## Usage
1. Run the FastAPI server:
   ```bash
   python src/main.py
   ```
2. Send a POST request to `http://127.0.0.1:5000/predict` with a JSON payload containing a base64-encoded image:
   ```json
   {
       "image": "base64_string_here"
   }
   ```
3. Receive the detection results, including the processed image, total bolt count, status, and detected classes.

## File Structure and Functionality
The project is organized into modular files for clarity and maintainability. Below is a breakdown of the key files and their roles:

- **`src/main.py`**: The entry point of the application. This script sets up the FastAPI app, defines endpoints (`/ServerCheck`, `/predict`), and handles incoming requests. The `/predict` endpoint processes base64-encoded images, invokes the detection and classification pipeline, and returns results, including the processed image, bolt count, status, and detected classes. It also saves non-conforming images to the `NG_Image` directory.

- **`src/detection.py`**: Contains the core detection and classification logic. Key classes and functions include:
  - `DetectionTask`: Base class for loading ONNX models and class names.
  - `TargetDetectionTask`: Manages YOLOv8 models for classification and segmentation, with methods like `perform_classifier()`, `perform_classifier_yolo()`, `nut_classifier_seg()`, `performaug()`, and `ImageEnhancer()` for brightness enhancement.
  - `CropDetectionTask`: Performs bolt detection using an ONNX model, with methods like `perform_detection()` for detecting bolts and `draw_boxes()` for visualizing results and invoking classification.
  - `process_image_with_classificationD1()`: An alternative detection method using contour detection and intensity-based classification.

- **`src/utils.py`**: Provides utility functions for image processing and saving. Key functions include:
  - `save_img()`: Saves non-conforming images to the `NG_Image` directory with a timestamped filename.
  - Defines the `transform` pipeline for resizing and normalizing images for ONNX classification.

- **`models/gear_bolt.onnx`**: ONNX model for detecting bolts in images.
- **`models/bolt_classifier_2025_old.onnx`**: ONNX model for classifying bolts as "NORM" or "TVS".
- **`models/best.pt`**: YOLOv8 model weights for bolt classification.
- **`models/Segbest.pt`**: YOLOv8 model weights for bolt segmentation.
- **`models/crop.names`**: File containing class names for the detection model.
- **`data/temp_check.bmp`**: A sample input image for testing the detection pipeline.
- **`data/NG_Image/`**: Directory for saving non-conforming images.
- **`tests/test_detection.py`**: Includes unit tests for detection and classification functions (to be implemented).
- **`requirements.txt`**: Lists all Python dependencies required to run the project.

This modular structure ensures each component is isolated, making the codebase easy to maintain and extend.

## Example Output
```json
{
    "image": "base64_string_of_processed_image",
    "Total_Nut": 16,
    "Status": "OK",
    "Detected_Classes": ["NORM", "NORM", ..., "NORM"]
}
```

## Project Structure
- `src/`: Core source code for detection, classification, and API handling.
- `models/`: ONNX and YOLOv8 model files.
- `data/`: Sample input data and output directory for non-conforming images.
- `tests/`: Unit tests for key functions.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## Contact
For questions, reach out to [your.email@example.com](mailto:your.email@example.com).
