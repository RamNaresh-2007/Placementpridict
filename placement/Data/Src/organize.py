import os
import shutil

src_dir = r"c:\Users\ravil\OneDrive\Documents\ML\placement\Data\Src"

folders = {
    "01_Datasets": [
        "placement_predict_50k Dataset (2).csv",
        "placement_sample_20.csv"
    ],
    "02_Data_Preprocessing": [
        "categorical.py",
        "categorical_encoding.py",
        "numerical_scaling.py"
    ],
    "03_Exploratory_Data_Analysis": [
        "Eda.py",
        "EDA_Outputs",
        "plots"
    ],
    "04_Models_and_Pipelines": [
        "Models.py",
        "linear regression.py",
        "logistic_regression.py",
        "decision_trees.py",
        "complete_pipeline.py",
        "PPP.py"
    ],
    "05_Web_Application": [
        "app.py",
        "dashboard.py",
        "main.py",
        "config.py",
        "static",
        "templates"
    ],
    "06_Outputs_and_Utils": [
        "outputs",
        "processed",
        "processed_data",
        "PlacementPredict_Outputs",
        "Utils.py",
        "__pycache__"
    ]
}

for folder, items in folders.items():
    folder_path = os.path.join(src_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    for item in items:
        item_path = os.path.join(src_dir, item)
        if os.path.exists(item_path):
            shutil.move(item_path, os.path.join(folder_path, item))

# Fix paths in complete_pipeline.py and decision_trees.py
pipe_file = os.path.join(src_dir, "04_Models_and_Pipelines", "complete_pipeline.py")
if os.path.exists(pipe_file):
    with open(pipe_file, "r") as f:
        content = f.read()
    content = content.replace('"placement_predict_50k Dataset (2).csv"', 'os.path.join("..", "01_Datasets", "placement_predict_50k Dataset (2).csv")')
    with open(pipe_file, "w") as f:
        f.write(content)

tree_file = os.path.join(src_dir, "04_Models_and_Pipelines", "decision_trees.py")
if os.path.exists(tree_file):
    with open(tree_file, "r") as f:
        content = f.read()
    content = content.replace('"placement_sample_20.csv"', 'os.path.join("..", "01_Datasets", "placement_sample_20.csv")')
    with open(tree_file, "w") as f:
        f.write(content)
