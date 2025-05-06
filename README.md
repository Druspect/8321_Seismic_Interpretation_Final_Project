# Automated Seismic Facies Classification using 3D CNN, BiLSTM, and Hybrid Deep Learning Models

## Introduction
This repository hosts a research project focused on advancing seismic data interpretation for stratigraphic analysis through the application of deep learning models. Specifically, we explore the capabilities of a 3D Convolutional Neural Network (CNN), a Bidirectional Long Short-Term Memory (BiLSTM) network, and a novel hybrid model that combines the strengths of both architectures for seismic facies classification.

## Project Overview

### Motivation
Manual seismic stratigraphic interpretation is labor-intensive, prone to human error, and heavily reliant on expert knowledge. By leveraging machine learning, this project aims to improve the efficiency, accuracy, and objectivity of seismic facies classification. We address key challenges such as large dataset handling, noise, and complex geological structures that can inhibit our ability to extract stratigraphic bedding data.

### Research Questions
1.  How effectively can a 3D Convolutional Neural Network (CNN), trained on seismic amplitude patches, classify seismic facies defined by interpreted horizons?
2.  Can sequence-based models, specifically a Bidirectional Long Short-Term Memory (BiLSTM) network analyzing individual seismic traces, provide a competitive or complementary approach to patch-based CNNs for this classification task?
3.  Does a hybrid model combining features from the 3D CNN and BiLSTM improve classification performance compared to the individual models?
4.  Will these models be able to work on unlabeled data once trained with any discernible accuracy? (Secondary question, focus on supervised first)

### Hypothesis
Our hypothesis is that a 3D CNN will excel at capturing spatial patterns from seismic patches, while a BiLSTM will be effective in modeling sequential dependencies along seismic traces. We further hypothesize that a hybrid model, integrating features from both the CNN and BiLSTM, will outperform the individual models by leveraging both spatial and sequential information, leading to more accurate and robust seismic facies classification.

### Objectives
-   Develop and evaluate a 3D CNN for patch-based seismic facies classification using the F3 block dataset.
-   Develop and evaluate a BiLSTM network for trace-based seismic facies classification using the F3 block dataset.
-   Design, implement, and evaluate a hybrid deep learning model that combines the feature extraction capabilities of the 3D CNN and BiLSTM.
-   Compare the performance of the 3D CNN, BiLSTM, and hybrid model using standard classification metrics.
-   Investigate the potential for these trained models to generalize to unlabeled data (exploratory).

## Methodology
The project adopts a structured pipeline implemented in a Jupyter notebook:
1.  **Data Collection and Preprocessing**:
    *   Utilize the open-source TerraNubis F3 block seismic dataset.
    *   Preprocess data via SEG-Y loading, coordinate scaling, bandpass filtering, Hilbert transform for envelope extraction, and normalization.
    *   Extract labeled 3D patches for the CNN and 1D traces for the BiLSTM based on interpreted horizons.
2.  **3D Convolutional Neural Network (CNN) Model**:
    *   Design and implement a custom 3D CNN architecture tailored for seismic patch classification.
    *   Train the CNN on extracted 3D patches and evaluate its performance.
3.  **Bidirectional Long Short-Term Memory (BiLSTM) Model**:
    *   Design and implement a BiLSTM network to classify facies based on individual seismic traces (sequences of amplitude values).
    *   Train the BiLSTM on extracted 1D traces and evaluate its performance.
4.  **Hybrid CNN-BiLSTM Model**:
    *   Develop a hybrid architecture that combines features extracted by the 3D CNN (from patches) and the BiLSTM (from corresponding traces).
    *   Train the hybrid model and evaluate its performance.
5.  **Comparative Analysis**:
    *   Benchmark the performance of the 3D CNN, BiLSTM, and hybrid model.
    *   Use metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
6.  **Visualization**:
    *   Generate seismic sections with predicted facies overlays for validation and presentation.
    *   Plot training history (loss and accuracy) and performance comparison charts.

## Getting Started

### Prerequisites
-   **Python**: 3.8 or higher
-   **JupyterLab**: For running the notebook
-   **PyTorch**: For deep learning model implementation
-   **Hardware**: GPU (CUDA or MPS) recommended for efficient model training.

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/druspect/druspect-8321_seismic_interpretation_final_project.git
    cd druspect-8321_seismic_interpretation_final_project
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Obtain the TerraNubis F3 seismic dataset as outlined in `data/README.md` (to be created or updated) and place them in the `data/` directory.

### Usage
The project utilizes a single Jupyter notebook (`Revised Seismic Interpretation Analysis Final.ipynb` or similar name) within a JupyterLab environment. Detailed instructions for running and modifying the notebook are provided within the notebook itself.

## Repository Structure (Proposed)
-   `data/`: Guidelines for sourcing and organizing seismic datasets (e.g., F3 block).
-   `docs/`: Theoretical background, literature resources, and project drafts.
-   `notebooks/`: Jupyter notebook(s) for the analysis (e.g., `Revised Seismic Interpretation Analysis Final.ipynb`).
-   `outputs/`: Placeholder for figures, model weights, and other generated outputs.
-   `src/` (Optional): Python scripts for helper functions or model definitions if refactored out of the notebook.
-   `README.md`: Central documentation (this file).
-   `LICENSE`: Project license.
-   `requirements.txt`: List of Python dependencies.

## Expected Challenges
-   **Data Scale**: Efficiently processing and training models on large 3D seismic datasets.
-   **Label Quality**: Mitigating the impact of any noise or inconsistencies in the interpreted horizon data used for labeling.
-   **Model Complexity**: Balancing model complexity with available computational resources and training time.
-   **Hyperparameter Tuning**: Finding optimal hyperparameters for three distinct deep learning architectures.

## Assumptions
-   Availability of the F3 block dataset with seismic data and corresponding horizon interpretations.
-   The chosen deep learning architectures (3D CNN, BiLSTM, Hybrid) are suitable for capturing relevant features for seismic facies classification.

## Documentation
-   Principles of seismic interpretation and the deep learning techniques employed.
-   Key literature and datasets informing the project.

## Obtaining Datasets
-   **TerraNubis F3 Dataset**: Download from [TerraNubis](https://terranubis.com). This is an open seismic dataset with geological context, suitable for this project.

## Setup Instructions for Data
1.  Download the `.sgy` files for the seismic volume and any associated horizon/top files.
2.  Place them in a designated data directory (e.g., `data/F3_Block/`).
3.  Update paths in the Jupyter notebook to point to these data files.

## Expected Data Format
-   Seismic Volume: SEG-Y format.
-   Horizons/Tops: Text files (e.g., `.xyt`) containing X, Y, and TWT (Two-Way Time) information, or other formats compatible with the preprocessing scripts.

