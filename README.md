# Food AI Nutrition Estimator

This project is a food image segmentation and nutrition estimation system. It uses a U-Net segmentation model to detect the food region in an image, then estimates nutrition values using a food classification model and a nutrition CSV database.

The main idea is that nutrition should not be estimated from the whole image directly. Instead, the system first finds the food pixels, estimates the visible food area, and uses that area to adjust the portion size.

---

## Project Overview

The pipeline works like this:

Input food image  
→ U-Net food segmentation  
→ Food mask and overlay  
→ Food label prediction using ResNet-50  
→ Nutrition lookup from CSV  
→ Portion-adjusted calories and macros  

The system outputs:

- Predicted food name
- Food segmentation mask
- Overlay image
- Food area percentage
- Portion multiplier
- Calories
- Fat
- Carbohydrates
- Protein

---

## Features

- Food segmentation using U-Net
- Binary food mask prediction
- Image overlay visualization
- Food classification using pretrained ResNet-50
- Nutrition estimation using a CSV database
- Portion adjustment based on segmented food area
- Batch nutrition prediction for multiple test images
- Custom image upload demo in Google Colab

---

## Project Structure

```text
Food-AI-Nutrition-Estimator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Food_AI_Final_Colab.ipynb
│
├── dataset.py
├── losses.py
├── metrics.py
├── model_unet.py
├── train_segmentation.py
├── predict_segmentation.py
├── nutrition_with_segmentation.py
│
├── dish_ingredients_fixed_for_colab.csv
│
└── examples/
    ├── training_curves.png
    ├── prediction_grid.png
    └── sample_nutrition_result.png

Dataset Format

The image dataset should be organized like this:
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

Images are RGB food images.

Masks are binary images:

White pixels = food
Black pixels = background

The image and mask filenames should match.

Example:

train/images/00000145.jpg
train/masks/00000145.png

Model

The segmentation model is a vanilla U-Net.

Input:

RGB image: 3 × 256 × 256

Output:

Binary mask: 1 × 256 × 256

The U-Net model predicts which pixels belong to food.

Loss Function

The model is trained using a combined BCE-Dice loss:

Loss = 0.5 × BCE Loss + 0.5 × Dice Loss

Binary Cross Entropy helps classify each pixel as food or background.

Dice Loss helps improve overlap between the predicted mask and the ground-truth mask.

Metrics

The segmentation model is evaluated using Intersection over Union and Dice Score.

Intersection over Union:

IoU = Overlap / Union

Dice Score:

Dice = 2 × Overlap / (Predicted Area + Ground Truth Area)

A Dice score closer to 1 means the predicted mask is very close to the correct mask.

Results

The trained U-Net achieved:

Best Validation IoU: 0.8561
Test IoU: 0.8260
Test Dice: 0.8997

These results show that the model segments food regions well.

Nutrition Estimation

The nutrition estimation system uses:

U-Net to detect food area
ResNet-50 to predict the food label
A nutrition CSV file to find calories and macronutrients
A portion multiplier based on the food mask area

Example output:

Predicted dish: cucumber
Food area: 38.0%
Portion weight: 0.95×
Calories: 14 kcal
Fat: 0.1 g
Carbs: 3.4 g
Protein: 0.7 g
Serving: 95 g
Old vs New Pipeline

The old pipeline assumes every food image has a fixed serving size.

The new pipeline uses segmentation to estimate portion size.

Example:

Old pipeline:
Serving: 100 g
Calories: 15 kcal

New segmentation-based pipeline:
Serving: 95 g
Calories: 14 kcal
Food mask area: 38.0%

This makes the nutrition estimate more image-aware.

Custom Image Demo

The notebook includes a custom upload demo. A user can upload any food image, and the system will output:

Predicted dish
Food mask
Overlay
Calories
Fat
Carbs
Protein
Macronutrient chart

Example:

Predicted dish: ice cream
Food area: 24.8%
Portion weight: 0.62×
Calories: 128 kcal
Fat: 6.8 g
Carbs: 14.8 g
Protein: 2.2 g
How to Run

Open the Colab notebook:

Food_AI_Final_Colab.ipynb

Run the cells in order:

1. Environment Setup
2. Mount Google Drive and Set Paths
3. Upload Project Python Files
4. Prepare Dataset Splits
5. Load Dataset and Visualize Samples
6. Build and Verify U-Net Model
7. Test Loss Functions and Metrics
8. Train Food Segmentation Model
9. Evaluate Segmentation Model
10. Upload Fixed Nutrition CSV
11. Initialize Nutrition Estimator
12. Run Nutrition Estimation and Save Visualization
13. Compare Pipeline and Run Batch Nutrition
14. Custom Food Image Demo
Requirements

Install the required packages:

pip install -r requirements.txt

Main dependencies:

torch
torchvision
numpy
Pillow
pandas
matplotlib
tqdm
Files Not Included

The following folders are not included in the GitHub repository because they can be large:

dataset/
checkpoints/
predictions/
nutrition_output/

The trained model checkpoint is also not included by default:

best_model.pth

To reproduce results, train the model using the Colab notebook.

Limitations

The segmentation model performs well, but the food classification part uses a pretrained ImageNet ResNet-50 model. Because of this, it may sometimes predict non-food labels such as:

wooden spoon
tray

If the predicted label is not in the nutrition CSV, the nutrition values may be zero.

The portion estimate is based on 2D image area, not true food volume or weight. Camera angle, plate size, and food height can affect the estimate.
