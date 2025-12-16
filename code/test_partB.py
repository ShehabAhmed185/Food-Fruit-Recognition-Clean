from partB import load_partB, predict_image
import os

model = load_partB("partB")

label, conf = predict_image(
    model,
    os.path.join("Project Data", "Food", "Validation", "ceviche", "217909.jpg")
)

print(label, conf)
