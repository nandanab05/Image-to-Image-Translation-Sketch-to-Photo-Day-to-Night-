# src/main.py
import argparse
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

# Import our custom modules
import config
from preprocess import load_image, normalize_image
from augment import get_img2img_transform, augment_paired_images

def process_image_to_image():
    """
    Runs the full pipeline for the Sketch-to-Photo project.
    
    This version is updated to handle 'pix2pix' style datasets
    (like 'facades') where image A and B are already
    stacked side-by-side in a single file.
    """
    print("--- Starting Image-to-Image Processing (Pix2Pix Dataset) ---")
    
    # 1. Get transform pipeline
    # The get_img2img_transform will resize the final image.
    # We will feed it 256x256 images.
    transform = get_img2img_transform(config.IMG_HEIGHT, config.IMG_WIDTH)

    # 2. Define input and output directories
    # We look for images in the base folder, not trainA/trainB
    train_paths = glob(str(config.IMG2IMG_RAW_DIR / "*.*g")) # .jpg, .png
    
    if not train_paths:
        print(f"Error: Could not find images in {config.IMG2IMG_RAW_DIR}")
        print("Please download the 'facades' dataset and put the 'train' images there.")
        return
        
    # 3. Create output directories
    output_dir_train = config.IMG2IMG_PROCESSED_DIR / "train"
    output_dir_train.mkdir(parents=True, exist_ok=True)

    # 4. Loop, Process, and Save
    print(f"Processing {len(train_paths)} combined images...")
    for path in tqdm(train_paths):
        
        # Load the combined (e.g., 256x512) image
        combined_image = load_image(path)
        
        if combined_image is None:
            continue
            
        # 5. Split the combined image into Image A and Image B
        # Standard pix2pix format is [A|B] or [B|A].
        # The 'facades' dataset is [Photo|Label]
        # It's 256 high and 512 wide. We split it at 256.
        width = combined_image.shape[1]
        split_point = width // 2
        
        # Image A (the photo)
        imgA = combined_image[:, :split_point, :]
        # Image B (the sketch/label)
        imgB = combined_image[:, split_point:, :]

        # Resize to our target size before augmentation
        # (Original is 256x256, but good to be safe)
        imgA = cv2.resize(imgA, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        imgB = cv2.resize(imgB, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            
        base_name = os.path.basename(path)
        
        # 6. Apply Augmentation and Preprocessing
        for i in range(config.AUGMENTATION_COUNT):
            # The 'augment_paired_images' function applies the *same*
            # random flip/etc. to both images, which is critical.
            aug_A, aug_B = augment_paired_images(imgA, imgB, transform)
            
            # 7. Normalize
            norm_A = normalize_image(aug_A, mode=config.NORMALIZATION_MODE)
            norm_B = normalize_image(aug_B, mode=config.NORMALIZATION_MODE)
            
            # 8. Save
            # Stack them back together for the model [A|B]
            stacked_image = np.concatenate([norm_A, norm_B], axis=1)
            
            # Save as .npy to preserve float values
            save_name = f"{output_dir_train}/{base_name.split('.')[0]}_aug_{i}.npy"
            np.save(save_name, stacked_image)

    print("--- Image-to-Image Processing Complete ---")
    print(f"Processed data saved to: {output_dir_train}")

def process_crowd_density():
    """
    Runs the pipeline for the Crowd Density project.
    """
    print("--- Starting Crowd Density Processing ---")
    print("Note: This function is still a placeholder.")
    print("Your next task is to write the code here to:")
    print("1. Install 'scipy': pip install scipy")
    print("2. Loop through annotation files in 'data/raw/crowd_density/annotations/'")
    print("3. Use 'scipy.io.loadmat(file_path)' to read the .mat files.")
    print("4. Extract the 'image_info' key to get head (x,y) locations.")
    print("5. Generate a 'density map' from these (x,y) points.")
    print("6. Load the matching image, apply augmentations, and save both.")
    print("--- Crowd Density Processing (Placeholder) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Pipeline")
    parser.add_argument(
        "--project", 
        type=str, 
        required=True, 
        choices=["img2img", "crowd"],
        help="Which project to process data for."
    )
    
    args = parser.parse_args()
    
    if args.project == "img2img":
        process_image_to_image()
    elif args.project == "crowd":
        process_crowd_density()