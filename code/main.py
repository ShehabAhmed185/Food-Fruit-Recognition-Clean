import loadData
import partA
import partB
import partC
import partD
import partE
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



def read_images(image_paths, target_size=(224, 224)):
    """Read and resize images using OpenCV"""
    images = []
    for path in image_paths:
        # Read image in BGR format
        img = cv2.imread(path)
        if img is not None:
            # Resize to consistent size
            img = cv2.resize(img, target_size)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Warning: Could not read image {path}")
            # Add placeholder for missing images
            images.append(np.zeros((*target_size, 3), dtype=np.uint8))
    
    return np.array(images)

def read_masks(mask_paths, target_size=(224, 224)):
    """Read and resize masks using OpenCV"""
    masks = []
    for path in mask_paths:
        # Read mask in grayscale
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize to consistent size
            mask = cv2.resize(mask, target_size)
            masks.append(mask)
        else:
            print(f"Warning: Could not read mask {path}")
            # Add placeholder for missing masks
            masks.append(np.zeros(target_size, dtype=np.uint8))
    
    return np.array(masks)


def convert_rgb_to_bgr(food_data):
    """Convert all RGB images in food_data to BGR (for VGG16 compatibility)."""
    food_data["train_images"] = food_data["train_images"][..., ::-1]  # RGB → BGR
    food_data["val_images"] = food_data["val_images"][..., ::-1]      # RGB → BGR
    return food_data


# Load dataset paths
root_path = r"Project Data"
data = loadData.load_dataset(root_path)
# Read actual images using OpenCV
print("Loading Food images...")
food_train_images = read_images(data["food_train_images"])
food_val_images = read_images(data["food_val_images"])

print("Loading Fruit images...")
fruit_train_images = read_images(data["fruit_train_images"])
fruit_val_images = read_images(data["fruit_val_images"])

print("Loading Fruit masks...")
fruit_train_masks = read_masks(data["fruit_train_masks"])
fruit_val_masks = read_masks(data["fruit_val_masks"])

# Prepare the data dictionaries
food = {
    "train_images": food_train_images,
    "train_labels": data["food_train_labels"],
    "val_images": food_val_images,
    "val_labels": data["food_val_labels"],
    "train_cal": data["food_train_cal"],
    "val_cal": data["food_val_cal"]
}

fruit = {
    "train_images": fruit_train_images,
    "train_masks": fruit_train_masks,
    "train_labels": data["fruit_train_labels"],
    "val_images": fruit_val_images,
    "val_masks": fruit_val_masks,
    "val_labels": data["fruit_val_labels"],
    "calories": data["fruit_calories"]
}


# Data inspection
print("\n=== Data Summary ===")
print(f"Food - Training: {len(food['train_images'])} images, {len(set(food['train_labels']))} categories")
print(f"Food - Validation: {len(food['val_images'])} images")
print(f"Fruit - Training: {len(fruit['train_images'])} images, {len(set(fruit['train_labels']))} categories")
print(f"Fruit - Validation: {len(fruit['val_images'])} images")
print(f"Fruit masks - Training: {len(fruit['train_masks'])} masks")
print(f"Fruit masks - Validation: {len(fruit['val_masks'])} masks")

# Check image shapes
print(f"\nImage shape: {food['train_images'][0].shape}")
print(f"Mask shape: {fruit['train_masks'][0].shape}")

# Visualize sample data
def visualize_samples():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Food samples
    for i in range(3):
        axes[0, i].imshow(food['train_images'][i])
        axes[0, i].set_title(f"Food: {food['train_labels'][i]}")
        axes[0, i].axis('off')
    
    # Fruit samples with masks
    for i in range(3):
        axes[1, i].imshow(fruit['train_images'][i])
        axes[1, i].set_title(f"Fruit: {fruit['train_labels'][i]}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show masks separately
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        axes[i].imshow(fruit['train_masks'][i], cmap='gray')
        axes[i].set_title(f"Mask: {fruit['train_labels'][i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# # Uncomment to visualize samples
# visualize_samples()
print("\nData loading completed successfully!")

# plt.imshow(food["train_images"][5])
# plt.title(food["train_labels"][5])
# plt.axis('off')
# plt.show()
def readTestData(imgPath):
    img = cv2.imread(imgPath)
    if img is not None:
        # Resize to consistent size
        target_size=(224, 224)
        img = cv2.resize(img, target_size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING PART A TRAINING")
    print("="*60)
    # partA_model, partA_history, partA_test_script = partA.run_partA(food, fruit, use_generator=True, 
    #                                                                 batch_size=8, epochs=2)    
    #ready for testing
    imgPaths =[os.path.join("Project Data", "Food", "Validation", "ceviche", "217909.jpg"),
               os.path.join("Project Data", "Lichi.jpg"),
               os.path.join("Project Data", "Guava.jpg"),
               os.path.join("Project Data", "persimmons.jpg"),
               os.path.join("Project Data", "mango.jpg"),
               os.path.join("Project Data", "Fruit","Validation","Banana","Images","76.jpg")
               ]
    # partA_testResult = partA_test_script(imgPaths[0])
    if(  'FruitX' == 'Fruit' ):
        # partA_testResult['classification'] 
        #here should do partC

        print("it's Fruit")
        # print(f"  Confidence: {partA_testResult['confidence']:.3f}")
        # print(f"  Food Probability: {partA_testResult['probability_food']:.3f}")
        # print(f"  Fruit Probability: {partA_testResult['probability_fruit']:.3f}")
        partC_model, partC_test = partC.run_partC(fruit, epochs=1, batch_size=8)
        # Test example
        # for i in range(4):
        img = readTestData(imgPaths[5])
        result = partC_test(img)
        print(result)

        # print("in Masks PartD")
        
        # seg_model, partD_test = partD.run_partD(
        #     fruit,
        #     epochs=5,
        #     batch_size=4
        # )

        # mask = partD_test(imgPaths[5], "output_mask_partD.png")


        # print("in Masks PartE")


        # #run ppartE
        # modelPartE , testPartE = partE.run_partE(fruit,epochs= 1, batch_size= 2)
        # gray_mask, color_mask = testPartE(
        #     imgPaths[5],
        #     "mask_gray_partE.png",
        #     "mask_color_partE.png"
        # )

    else:
        # #here should do partB
        print("it's Food")
        # food_BGR =convert_rgb_to_bgr(food)
        siamese_model = partB.run_partB(food, epochs=30, batch_size=16)
        
        
        # print("\nTop 5 Predictions:")
        # for i, pred in enumerate(food_result['all_predictions'][:5], 1):
        #     print(f"  {i}. {pred['category']}: {pred['confidence']:.3f} (distance: {pred['distance']:.3f})")
        