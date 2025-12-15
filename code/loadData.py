import os
from glob import glob


# -------------------------------
# Load Calories TXT File
# -------------------------------
def load_calories(file_path):
    calories = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ":" in line:
                    food_name, cal_info = line.split(":", 1)
                    food_name = food_name.strip()
                    cal_info = cal_info.strip()

                    if "~" in cal_info:
                        cal_part = cal_info.split("~")[1].strip()
                        cal_value_str = cal_part.split(" ")[0].strip()

                        try:
                            calories[food_name] = float(cal_value_str)
                        except:
                            continue
    except:
        print(f"Warning: Could not load calories from {file_path}")

    return calories


# -------------------------------
# Normalize Folder Label
# -------------------------------
def normalize_label(label):
    return label.replace("_", " ").title()


# -------------------------------
# Main Loading Function
# -------------------------------
def load_dataset(root_path):
    root_path = os.path.normpath(root_path)

    # -------------------------------
    # FOOD
    # -------------------------------
    food_train_dir = os.path.join(root_path, "Food", "Train")
    food_val_dir   = os.path.join(root_path, "Food", "Validation")

    food_train_images, food_train_labels = [], []
    food_val_images, food_val_labels     = [], []

    # Load Food Train
    if os.path.exists(food_train_dir):
        for cat in os.listdir(food_train_dir):
            cat_path = os.path.join(food_train_dir, cat)
            if not os.path.isdir(cat_path):
                continue

            for img_path in glob(os.path.join(cat_path, "*")):
                food_train_images.append(img_path)
                food_train_labels.append(cat)

    # Load Food Val
    if os.path.exists(food_val_dir):
        for cat in os.listdir(food_val_dir):
            cat_path = os.path.join(food_val_dir, cat)
            if not os.path.isdir(cat_path):
                continue

            for img_path in glob(os.path.join(cat_path, "*")):
                food_val_images.append(img_path)
                food_val_labels.append(cat)

    # Load Food Calories
    food_train_cal = load_calories(os.path.join(root_path, "Food", "Train Calories.txt"))
    food_val_cal   = load_calories(os.path.join(root_path, "Food", "Val Calories.txt"))

    food_train_cal_normalized = {normalize_label(k): v for k, v in food_train_cal.items()}
    food_val_cal_normalized   = {normalize_label(k): v for k, v in food_val_cal.items()}

    # -------------------------------
    # FRUIT
    # -------------------------------
    fruit_train_dir = os.path.join(root_path, "Fruit", "Train")
    fruit_val_dir   = os.path.join(root_path, "Fruit", "Validation")

    fruit_train_images, fruit_train_masks, fruit_train_labels = [], [], []
    fruit_val_images, fruit_val_masks, fruit_val_labels       = [], [], []

    # Load Fruit Train
    if os.path.exists(fruit_train_dir):
        for cat in os.listdir(fruit_train_dir):
            cat_path = os.path.join(fruit_train_dir, cat)

            images_dir = os.path.join(cat_path, "Images")
            masks_dir  = os.path.join(cat_path, "Mask")

            if not os.path.isdir(images_dir):
                continue

            for img_path in glob(os.path.join(images_dir, "*")):
                name = os.path.basename(img_path).split('.')[0]
                mask_path = os.path.join(masks_dir, f"{name}_mask.png")

                fruit_train_images.append(img_path)
                fruit_train_masks.append(mask_path)
                fruit_train_labels.append(cat)

    # Load Fruit Val
    if os.path.exists(fruit_val_dir):
        for cat in os.listdir(fruit_val_dir):
            cat_path = os.path.join(fruit_val_dir, cat)

            images_dir = os.path.join(cat_path, "Images")
            masks_dir  = os.path.join(cat_path, "Mask")

            if not os.path.isdir(images_dir):
                continue

            for img_path in glob(os.path.join(images_dir, "*")):
                name = os.path.basename(img_path).split('.')[0]
                mask_path = os.path.join(masks_dir, f"{name}_mask.png")

                fruit_val_images.append(img_path)
                fruit_val_masks.append(mask_path)
                fruit_val_labels.append(cat)

    fruit_calories = load_calories(os.path.join(root_path, "Fruit", "calories.txt"))
    fruit_calories_normalized = {normalize_label(k): v for k, v in fruit_calories.items()}

    # Return Everything
    return {
        # FOOD
        "food_train_images": food_train_images,
        "food_train_labels": food_train_labels,
        "food_val_images":   food_val_images,
        "food_val_labels":   food_val_labels,
        "food_train_cal":    food_train_cal_normalized,
        "food_val_cal":      food_val_cal_normalized,

        # FRUIT
        "fruit_train_images": fruit_train_images,
        "fruit_train_masks":  fruit_train_masks,
        "fruit_train_labels": fruit_train_labels,
        "fruit_val_images":   fruit_val_images,
        "fruit_val_masks":    fruit_val_masks,
        "fruit_val_labels":   fruit_val_labels,
        "fruit_calories":     fruit_calories_normalized
    }
