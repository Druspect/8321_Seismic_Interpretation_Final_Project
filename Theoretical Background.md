# Theoretical Background

This document outlines the theoretical foundations of the project, focusing on deep learning techniques for seismic facies classification.

## Seismic Interpretation
Seismic data interpretation involves analyzing reflected seismic waves to map subsurface geological structures, stratigraphy, and potential hydrocarbon reservoirs. A key task is **seismic facies classification**, which aims to categorize distinct zones within the seismic volume based on their reflection characteristics (e.g., amplitude, frequency, continuity, texture). These facies often correspond to different depositional environments or lithologies.

Challenges in seismic interpretation include:
-   **Noise**: Seismic data can be contaminated by various types of noise from acquisition and environmental factors.
-   **Complexity**: Subsurface geology can be highly complex, with features like faults, folds, and subtle stratigraphic traps that are difficult to delineate.
-   **Subjectivity**: Manual interpretation can be subjective and time-consuming, especially for large 3D seismic volumes.

Automated methods using machine learning, particularly deep learning, offer the potential for more objective, consistent, and efficient interpretation.

## Deep Learning for Seismic Facies Classification

### Convolutional Neural Networks (CNNs) for Seismic Data
3D Convolutional Neural Networks (CNNs) are well-suited for analyzing volumetric seismic data. They can learn hierarchical spatial features directly from 3D seismic patches.
-   **Input**: Typically, small 3D patches (e.g., 32x32x32 or 64x64x64 samples) extracted from the seismic amplitude volume.
-   **Architecture**: Consists of multiple layers of 3D convolutions, activation functions (e.g., ReLU), pooling layers (e.g., MaxPooling), and batch normalization. The final layers are usually fully connected layers for classification.
-   **Strengths**: Effective at capturing local 3D spatial patterns, textures, and structural information within the seismic data, which are crucial for distinguishing different facies.

### Recurrent Neural Networks (RNNs) and LSTMs for Seismic Data
Seismic traces (1D sequences of amplitude values along the depth/time axis) exhibit sequential dependencies. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, are designed to model such sequential data.
-   **Input**: Individual 1D seismic traces or segments of traces.
-   **Architecture**: LSTMs use gates (input, forget, output) to control the flow of information, allowing them to learn long-range dependencies within a sequence. Bidirectional LSTMs (BiLSTMs) process the sequence in both forward and backward directions, providing richer contextual information.
-   **Strengths**: Can capture vertical stratigraphic patterns and sequential relationships along traces that might be indicative of specific facies or depositional sequences.

### Hybrid Deep Learning Models
Hybrid models aim to combine the strengths of different architectures. In this project, a hybrid model integrating a 3D CNN and a BiLSTM is explored.
-   **Concept**: The CNN branch processes 3D patches to extract spatial features, while the BiLSTM branch processes the corresponding 1D central traces to extract sequential features.
-   **Feature Fusion**: Features from both branches are typically concatenated and then fed into a final classifier (e.g., fully connected layers) to make the facies prediction.
-   **Potential Benefits**: By leveraging both spatial and sequential information, hybrid models may achieve improved classification accuracy and robustness compared to using either a CNN or BiLSTM alone.

## Datasets
-   **TerraNubis F3 Block Dataset**: A publicly available 3D seismic survey from the Dutch sector of the North Sea. It includes the seismic volume (SEG-Y format) and interpreted horizons, which are used to define the ground truth facies labels for supervised learning. (Source: [https://terranubis.com](https://terranubis.com))

