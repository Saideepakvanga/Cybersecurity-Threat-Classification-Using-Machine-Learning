# CICIDS2017-DATASET
Cybersecurity Threat Classification Using Machine Learning (CIC-IDS2017)


## Overview
This project focuses on **classifying cybersecurity threats** using machine learning techniques. The system is trained on the **CICIDS2017** dataset to detect network intrusions effectively. Multiple ML models are trained like e.g., Random Forest, SVM, or Neural Networks for classification.and evaluated to determine the best approach for threat detection.


## Dataset
The project utilizes the **CICIDS2017** dataset, available at:
[https://www.unb.ca/cic/datasets/ids.html](https://www.unb.ca/cic/datasets/ids.html)

If downloading the dataset is an issue, a small custom dataset with labeled attack categories may be used.

## Task Workflow
1. **Data Preprocessing:**
   Inspect Data: Understand data types, missing values, and distributions.
   Handle Missing Values:
   Identify missing data.
   Decide on deletion or imputation.
   Apply chosen technique (mean, median, etc.).
   Address Outliers: Detect and handle extreme values.
   Normalize/Standardize: Scale numerical features for consistent ranges.
   Encode Categorical Features: Convert text data to numerical form (one-hot, label, etc.).
   Feature Selection/Engineering: Choose relevant features or create new ones.
   Split Data: Divide into training and testing sets.
   Verify: Ensure data is clean and ready for modeling.
   
2. **Feature Selection:**
   - Feature selection isolates the most impactful variables for a model. It reduces dimensionality, speeds up training, and enhances model interpretability. By focusing on relevant features, it 
     eliminates noise and improves classification accuracy.

3. **Model Selection & Training:**
   -The model selection and training process involves several crucial steps:

    **Define the Problem:** Clearly understand the classification task and its objectives.
    **Select Candidate Models:** Choose at least two diverse algorithms (e.g., Random Forest, SVM, Neural Networks) based on the data characteristics and problem complexity.
    **Split Data:** Divide the preprocessed data into training, validation, and test sets.
    **Model Instantiation:** Initialize the chosen models with default or initial parameter settings.
    **Hyperparameter Tuning:** Use techniques like cross-validation or grid search to optimize model parameters on the validation set.
    **Model Training:** Train each tuned model on the training data.
    **Model Evaluation:** Assess the performance of each trained model on the validation set using appropriate metrics (accuracy, precision, recall, F1-score).
    **Model Selection:** Choose the best-performing model based on the evaluation results.
    **Final Evaluation:** Evaluate the selected model's performance on the unseen test set to estimate its generalization ability.
    **Model Deployment (Optional):** If the model meets performance requirements, deploy it for real-world use.

4. **Evaluation:**

    Accuracy: Overall correctness, but can be misleading with imbalanced datasets.   
    Precision: Proportion of correctly predicted positives out of all predicted positives, crucial when false positives are costly.   
    Recall (Sensitivity): Proportion of correctly predicted positives out of all actual positives, important when false negatives are costly.   
    F1-score: Harmonic mean of precision and recall, balancing both metrics, useful when seeking a compromise.


5. **Visualization:**
   Confusion Matrix: Displays true vs. predicted labels, revealing misclassifications and performance per class.
   Feature Importance Plots: Show the relative importance of features in a model's predictions, aiding in feature selection and model interpretation.
   ROC Curve (Receiver Operating Characteristic) & AUC (Area Under the Curve): Visualizes the trade-off between true positive rate and false positive rate, evaluating model discrimination.
   Precision-Recall Curve: Useful for imbalanced datasets, showing the trade-off between precision and recall.
   Bar/Histograms: Display feature distributions and class imbalances.
   Scatter Plots: Visualize relationships between features and the target variable.
   Heatmaps: Show correlation between features.
## Tech Stack
- **Python**
- **Libraries Used:**
  - `Scikit-learn` (for ML models and evaluation)
  - `Pandas` (for data handling)
  - `NumPy` (for numerical operations)
  - `Matplotlib` & `Seaborn` (for visualization)
  - `TensorFlow/PyTorch` (if deep learning models are used)

## Installation
1. Clone the repository:
   ```bash
  https://github.com/Saideepakvanga/Cybersecurity-Threat-Classification-Using-Machine-Learning/tree/main '''
 
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the dataset from [CICIDS2017](https://www.unb.ca/cic/datasets/ids.html).
4. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
5. Train the models:
   ```bash
   python train.py
   ```
6. Evaluate the models:
   ```bash
   python evaluate.py
   ```

## Usage
- Modify `config.py` to adjust dataset paths and model parameters.
- Run the scripts sequentially to preprocess data, train models, and evaluate performance.
- Visualize results using provided Jupyter notebooks.

## Results
- Detailed evaluation metrics and visualizations will be saved in the `results/` directory.
- The best-performing model can be used for real-time intrusion detection in a production environment.

## Submission Requirements
- A **Jupyter Notebook** or Python script with the complete implementation.
- A **brief report** (PDF or DOCX, max 3 pages) summarizing the approach, findings, and results.
- A **README file** explaining how to run the code.

## Author
vangasaideepak
[GitHub Profile](https://github.com/Saideepakvanga/Cybersecurity-Threat-Classification-Using-Machine-Learning/tree/main))

## License
This project is licensed under the MIT License.
### Outputs
Model evaluation metrics (accuracy, precision, recall, F1-score)
Performance comparison graphs
### Notes
Ensure the dataset file is in the correct directory before running the scripts.

---
Feel free to contribute or suggest improvements!
