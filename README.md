<h1>MIT-BIH Arrhythmia 1D-CNN Classifier</h1>

This project provides a simple and reproducible pipeline for ECG beat segmentation and arrhythmia classification using the MIT-BIH Arrhythmia Database and a 1D-CNN model.

<h2>Pipeline Overview</h2>

<ol> 
  <li>Export WFDB → CSV</li>

  <li>Preprocess ECG signals (band-pass filtering)</li>

  <li>Beat segmentation (≈600 ms windows, 216 samples)</li>

  <li>Train 1D-CNN (PyTorch)</li>

  <li>Evaluate — confusion matrix + classification report</li>
