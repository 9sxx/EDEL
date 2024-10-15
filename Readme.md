## README.md

### 1. **Project Title**
   - EDEL (Error-Driven Ensemble Learning Algorithm)

### 2. **Project Description**
   - EDEL is a machine learning algorithm designed to address the challenge of class imbalance. By dynamically reintroducing misclassified samples during training and employing a multi-perspective learning strategy, EDEL enhances the model's ability to recognize minority class instances. This project implements the EDEL algorithm and compares it with various baseline methods (such as SMOTE, SMOTE-TLNN-DEPSO, and models without any imbalance handling techniques) across multiple real-world datasets, including financial fraud detection and credit risk assessment.

### 3. **Installation**
   - **Step 1:** Clone the repository:
     ```bash
     git clone <repository-url>
     ```
   - **Step 2:** Set up the datasets. Refer to the `Datasets download link.md` file for download links to the datasets, and download them into the `dataset` folder under the respective directories.

### 4. **Datasets**
   - **Download datasets:**
     - Open the `Datasets download link.md` file to get the download links for each dataset.
     - After downloading, place the datasets in their respective folders under the `dataset/` directory (e.g., `dataset/Adult Income Dataset`, `dataset/Spambase Dataset`).
   - **Preprocessing the data:**
     - For each dataset, run the preprocessing scripts located in the corresponding subfolder under the `experiment` folder to convert the raw data into `data.csv`.
     - Example command:
       ```bash
       python <dataset-folder>/preprocess.py
       ```

### 5. **Usage**
   - **Note:** Before running the following scripts, make sure to update the dataset paths in `baseline_models.py`, `experiment_main.py`, `experiment_smote.py`, and `experiment_smote_tlnn_depso.py` according to the respective dataset locations.

   - **Baseline Model (No Imbalance Handling):**
     - To run experiments using baseline models without any imbalance handling techniques:
       ```bash
       python experiment/baseline_models.py
       ```
   - **EDEL (Error-Driven Ensemble Learning):**
     - To run experiments using the EDEL algorithm:
       ```bash
       python experiment/experiment_main.py
       ```
   - **SMOTE:**
     - To run experiments using the SMOTE algorithm for imbalance handling:
       ```bash
       python experiment/experiment_smote.py
       ```
   - **SMOTE-TLNN-DEPSO:**
     - To run experiments using the state-of-the-art SMOTE-TLNN-DEPSO method:
       ```bash
       python experiment/experiment_smote_tlnn_depso.py
       ```

### 6. **Experiment Details**
   - **Baseline Experiment:**
     - `baseline_models.py`: Runs baseline models without any imbalance handling techniques across all datasets.
   - **EDEL Experiment:**
     - `experiment_main.py`: Runs the EDEL algorithm on the datasets and records the results and metrics, such as Recall and G-mean.
   - **SMOTE Experiment:**
     - `experiment_smote.py`: Runs experiments using the SMOTE algorithm for handling imbalance.
   - **SMOTE-TLNN-DEPSO Experiment:**
     - `experiment_smote_tlnn_depso.py`: Runs experiments using the SMOTE-TLNN-DEPSO method, a state-of-the-art approach for handling class imbalance.

### 7. **Sensitivity Analysis**
   - Sensitivity analysis for each dataset can be run using the scripts in the `sensitivity_analysis/` folder. Each `.py` file corresponds to the sensitivity analysis for a specific dataset.
     - For example, to run the sensitivity analysis for the Adult Income dataset:
       ```bash
       python sensitivity_analysis/sensitivity_analysis_aid.py
       ```
