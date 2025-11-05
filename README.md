# Image Processing Project 
# Image Processing Pipeline

This project contains the data preprocessing and augmentation pipeline for two ML tasks:
1.  Image-to-Image Translation (Sketch to Photo)
2.  Crowd Density Estimation

## Role: Research & Data Lead

The code in `src/` is used to process data from `data/raw/` and save the clean, augmented, and normalized results into `data/processed/`.

### How to Run

1.  Add raw datasets to the `data/raw/` subfolders.
2.  Install requirements: `pip install -r requirements.txt`
3.  Run the pipeline for a specific project:

    ```bash
    # To process the sketch-to-photo data
    python src/main.py --project img2img
    
    # To process the crowd density data
    python src/main.py --project crowd
    ```
4.  Fill out the `docs/dataset_report.md` with details on the data sources.