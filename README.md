# High-Performance Medical Image Analysis with Numba

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Numba](https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Overview
This project demonstrates the application of **High-Performance Computing (HPC)** techniques to medical image analysis. By leveraging Python's `numba` library, the project significantly accelerates the computationally heavy task of extracting statistical features from medical images (Healthy, Tumor, and Pneumonia classes).

The extracted features are then used to train a Logistic Regression model capable of classifying the disease state with high accuracy.

## 🚀 Key Features
- **Synthetic Medical Data Generation**: Simulates datasets of medical images for demonstration purposes.
- **JIT Compilation & Parallelization**: Uses Numba's `@njit(parallel=True)` to bypass Python's Global Interpreter Lock (GIL) and run feature extraction on multiple CPU cores simultaneously.
- **Machine Learning**: Uses `scikit-learn` to train a Logistic Regression classifier on the extracted image features.
- **Data Visualization**: Uses `matplotlib` to plot disease distributions and execution benchmark comparisons.

## 📊 Performance Benchmarking

A core focus of this project is comparing traditional sequential processing against Numba-optimized parallel processing. 

During the feature extraction phase on a massive dataset of synthetic images, the heavy computational load is distributed across available CPU threads. 

**Performance Gains:**
- **Sequential Approach**: Processes one image at a time, resulting in significant bottlenecks and high latency.
- **Parallel Approach (HPC)**: Achieves a **massive speedup**, drastically reducing feature extraction time. 
- *Note: Exact execution times depend on your local CPU hardware, but `numba` commonly achieves speedups of magnitudes (e.g., 10x-50x faster) for mathematical operations over standard Python loops.*

## 🧠 Model Accuracy
The Logistic Regression model trained on these optimized features successfully distinguishes between **Healthy**, **Tumor**, and **Pneumonia** affected images. 
- **Classification Accuracy:** **100.0%** (on synthetic dataset validation)

## 📂 Project Structure
- `medical_image_analysis_using_numba.ipynb`: The main Jupyter Notebook containing all the code for data generation, sequential vs. parallel feature extraction, model training, and performance visualizations.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd medical-image-analysis
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install numpy matplotlib scikit-learn numba
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook "medical_image_analysis_using_numba (1).ipynb"
   ```

## 📈 Visualizations
The notebook generates several plots to help understand the data and processing:
1. **Sample Images**: Visual representation of the generated Healthy, Tumor, and Pneumonia images.
2. **Disease Distribution**: A pie chart showing the breakdown of the dataset classes.
3. **Performance Graph**: A bar chart directly comparing Sequential vs. Parallel feature extraction execution times.

---
*Built to showcase the raw power of Numba for High-Performance Computation in Healthcare AI.*
