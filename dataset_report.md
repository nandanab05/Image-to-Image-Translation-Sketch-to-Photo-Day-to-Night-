# Dataset Documentation Report
**Author:** (Your Name)
**Role:** Research & Data Lead
**Date:** (Fill in today's date)

---

## 1. Project: Image-to-Image Translation (Sketch to Photo)

### Dataset: (e.g., facades, maps, edges2shoes)

* **Source:** (e.g., "The sketchy dataset from TU Berlin", "Internal company data")
* **URL:** (Paste the URL you downloaded from)
* **Citation:** (If from a research paper, add it here)
* **Collection Date:** (e.g., "November 4, 2025")

### Characteristics

* **Type:** Paired Images (Sketch and Photo)
* **Raw File Count:** * `trainA`: (e.g., 400 images)
    * `trainB`: (e.g., 400 images)
* **Raw Format:** (e.g., `.jpg`)
* **Raw Resolution:** (e.g., "Varies, 256x256")

### Preprocessing & Augmentation Steps

1.  **Cleaning:** (e.g., "Removed 5 corrupt images", "All images were valid")
2.  **Resizing:** All images resized to **(256, 256)** using `cv2.INTER_AREA`.
3.  **Augmentation:** * `HorizontalFlip` (p=0.5)
    * `RandomBrightnessContrast` (p=0.2)
    * Applied **5** augmentations per original image.
4.  **Normalization:** Normalized to `[-1, 1]` (for GAN compatibility).
5.  **Final Output:**
    * Total `train` images: (e.g., 2,000)
    * Format: Paired images stacked side-by-side.
    * Saved as `.npy` files in `data/processed/sketch_to_photo/train/`

---

## 2. Project: Crowd Density Estimation

### Dataset: (e.g., ShanghaiTech, UCF-QNRF)

* **Source:** (e.g., "ShanghaiTech Dataset Part B")
* **URL:** (Paste the URL)
* **Citation:** (Add citation)
* **Collection Date:** (Add date)

### Characteristics

* **Type:** Images with point-level annotations.
* **Raw File Count:** * `images`: (e.g., 300 images)
    * `annotations`: (e.g., 300 `.mat` files)
* **Raw Format:** `.jpg` and `.mat`
* **Raw Resolution:** Varies.

### Preprocessing & Augmentation Steps

1.  **Cleaning:** (e.g., "All images and annotations valid")
2.  **Resizing:** (e.g., "All images resized to (512, 512)")
3.  **Annotation Conversion:** (e.g., "Loaded `.mat` files, extracted keypoints. Generated density maps using a Gaussian kernel.")
4.  **Augmentation:** (e.g., "Applied `HorizontalFlip`, `Rotate` to both images and keypoints.")
5.  **Normalization:** (e.g., "Normalized images to `[0, 1]`.")
6.  **Final Output:**
    * (e.g., "Saved as paired `.npy` files (image and density map) in `data/processed/crowd_density/train/`")