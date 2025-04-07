

# SAM2 Segmentation Tool - User Guide

This guide provides detailed instructions for installing, running, and using the SAM2 Segmentation Tool, a graphical application for interactive image segmentation using the Segment Anything Model (SAM) from Hugging Face.

## Table of Contents

1. [Installation](#1-installation)
2. [Running the Application](#2-running-the-application)
3. [User Interface Overview](#3-user-interface-overview)
4. [Basic Workflow](#4-basic-workflow)
5. [Input Modes](#5-input-modes)
6. [Working with Labels](#6-working-with-labels)
7. [Generating and Selecting Masks](#7-generating-and-selecting-masks)
8. [Saving Results](#8-saving-results)
9. [Keyboard Shortcuts](#9-keyboard-shortcuts)
10. [Troubleshooting](#10-troubleshooting)

## 1. Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Git (optional, for cloning the repository)

### Setup Instructions

1. **Clone or download the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r CV_app/requirements.txt
   ```

   This will install all required packages including:
   - torch
   - numpy
   - Pillow
   - opencv-python
   - matplotlib
   - huggingface_hub
   - transformers
   - accelerate

   Note: The first run will download the SAM model from Hugging Face (approximately 1.2GB).

## 2. Running the Application

1. **Navigate to the application directory:**
   ```bash
   cd CV_app
   ```

2. **Launch the application:**
   ```bash
   python app.py
   ```

3. **On first run:**
   - The application will start with an empty canvas
   - The SAM model will be downloaded from Hugging Face when you first open an image

## 3. User Interface Overview

The application window is divided into two main sections:

**Left Panel:**
- Image canvas where you can view and interact with the loaded image
- Displays points, boxes, and generated masks

**Right Panel (Controls):**
- File Operations: Open images
- Input Mode: Select between different input methods
- Label Selection: Manage and select labels
- Actions: Generate masks, save results, clear data
- Status: Displays application status and messages
- Current Selections: Shows a summary of your current inputs

## 4. Basic Workflow

1. **Open an image** using the "Open Image" button
2. **Select a label** from the dropdown (or add a new one)
3. **Add prompts** to the image:
   - Add positive points (indicating what to include in the mask)
   - Optionally add negative points (indicating what to exclude)
   - Optionally add bounding boxes to constrain the mask
4. **Generate masks** by clicking the "Generate Masks" button
5. **Select the best mask** for each label from the options presented
6. **Save the results** by clicking "Generate & Save Grid"

## 5. Input Modes

The application supports four input modes:

### Positive Points (+)
- Click on areas you want to include in the mask
- Each positive point guides the model to include that region
- Best practice: Place points in the center of objects you want to segment

### Negative Points (-)
- Click on areas you want to exclude from the mask
- Useful for refining masks when the model includes unwanted regions
- Best practice: Place negative points in areas incorrectly included in previous mask attempts

### Box
- Click and drag to create a bounding box
- Constrains the mask generation to the area within the box
- Best practice: Draw a box that fully contains the object but is as tight as possible

### Fixed Points
- Special points that are assigned unique IDs (starting from 100)
- These points are preserved in the final grid with their unique IDs
- Useful for marking specific locations that need to be identified in the output

## 6. Working with Labels

### Default Labels
- The application starts with a default label "floor"

### Adding New Labels
1. Type the new label name in the text field next to "Add Label"
2. Click the "Add Label" button
3. The new label will be added to the dropdown and selected automatically

### Selecting Labels
- Use the dropdown to select the active label
- Points and boxes are associated with the currently selected label
- Each label can have its own set of positive points and fixed points
- Negative points are shared across all labels

## 7. Generating and Selecting Masks

### Generating Masks
1. Add at least one positive point for each label you want to segment
2. Click the "Generate Masks" button
3. The application will process each label separately and generate multiple mask options

### Selecting the Best Mask
1. After generation, a dialog will appear showing mask options for each label
2. For each label tab, select the best mask by clicking its radio button
3. The masks are sorted by confidence score (highest first)
4. Click "Apply Selections" to confirm your choices
5. Selected masks will be displayed on the main canvas

## 8. Saving Results

### Saving the Grid
1. After selecting masks, click "Generate & Save Grid"
2. The application will:
   - Create a timestamped directory in the `dataset` folder
   - Save the original image
   - Save individual masks (both as .npy and .png files)
   - Generate and save a multi-label grid
   - Save a visualization of the grid
   - Save metadata in JSON format

### Output Structure
```
dataset/
└── YYYYMMDD_HHMMSS/
    ├── original_image.png
    ├── label_map.json
    ├── summary.json
    ├── masks/
    │   ├── mask_label1.npy
    │   ├── mask_label1.png
    │   ├── mask_label2.npy
    │   └── mask_label2.png
    └── grid/
        ├── labeled_grid.npy
        ├── labeled_grid_vis.png
        └── grid_info.json
```

## 9. Keyboard Shortcuts

- `1`: Switch to Positive Points mode
- `2`: Switch to Negative Points mode
- `3`: Switch to Box mode
- `4`: Switch to Fixed Points mode
- `c`: Clear all data
- `g`: Generate masks
- `s`: Save grid

## 10. Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure you have a stable internet connection for the first run
- Check that you have enough disk space for the model (~1.2GB)
- If using GPU, ensure CUDA is properly installed

**Memory Issues:**
- Large images may require significant memory
- Try using a smaller image or running on a machine with more RAM
- If using CPU, be patient as mask generation may take longer

**UI Issues:**
- If the UI appears too large or small, adjust your system's display scaling
- If buttons or controls are missing, try resizing the window

**Mask Generation Problems:**
- If masks are not generating correctly, try adding more positive and negative points
- For complex objects, use a combination of points and boxes
- Try different positions for your points if the results aren't satisfactory

### Getting Help

If you encounter issues not covered in this guide:
- Check the terminal/console for error messages
- Look for error messages in the Status panel
- Refer to the project documentation or repository issues

---

This application uses the Segment Anything Model (SAM) from Meta AI, accessed through the Hugging Face Transformers library.
