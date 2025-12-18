# Automated Crystalline Domain Segmentation in Polycrystalline TEM Images

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Introduction
This project presents an **automated, GUI-based workflow** for analyzing High-Resolution Transmission Electron Microscopy (HRTEM) images. By integrating **Gaussian Mixture Models (GMM)** for FFT peak classification and **DBSCAN** for real-space domain extraction, this tool automates the traditionally subjective and time-consuming process of crystallographic characterization.

This solution specifically targets polycrystalline materials with weak crystallinity, such as solid electrolyte interphase (SEI) layers in lithium-metal batteries, where manual analysis is often inconsistent.

## 🎥 Demo Video
Click the image below to watch the demonstration of the tool:

[![Watch the Demo](https://img.youtube.com/vi/K2Sx-ZWBmSc/maxresdefault.jpg)](https://youtu.be/K2Sx-ZWBmSc)

> **Video Title:** AI-ML-for-Microscopy-Hackathon-2025  
> **Description:** A walkthrough of the GUI-based tool leveraging machine learning to automate FFT analysis and reduce user bias in HRTEM workflows.

## 👥 Authors
* **Woojin Bae**, Shihyun Kim, Jinho Rhee
* School of Chemical and Biological Engineering, Seoul National University

## 🚀 Key Features
* **Automated FFT Peak Detection:** Utilizes a two-component GMM to unsupervisedly classify peaks as signal (crystalline) or noise based on intensity, sharpness, and radial profile.
* **d-Spacing Calculation:** Automatically calculates d-spacing values from peak positions using user-provided pixel size calibration.
* **Domain Clustering:** Reconstructs real-space images via inverse FFT and applies DBSCAN to group crystalline domains, effectively handling irregular grain boundaries without specifying cluster counts.
* **Interactive GUI:** Users can adjust classification sensitivity via sliders, filter non-physical peaks, and visualize results immediately.

## 🛠 Methodology
The pipeline consists of four integrated stages:
1.  **GMM-Based Peak Classification:** Automatically distinguishes crystalline signals from background noise in the 2D FFT spectrum.
2.  **Automated d-Spacing Calculation:** Converts peak positions to meaningful crystallographic data.
3.  **DBSCAN-Based Domain Clustering:** Identifies spatially connected high-intensity regions in the reconstructed image while filtering isolated noise.
4.  **Visualization & Quantification:** Overlays color-coded domains on the original image and calculates individual and total crystalline areas.

## 📊 Results
* Successfully analyzes polycrystalline HRTEM images containing weak crystallinity and phase heterogeneity.
* Identifies multiple coexisting crystalline phases and measures crystallite sizes ranging from nanometers to tens of nanometers.
* Significantly reduces analysis time compared to manual methods and ensures reproducibility independent of the analyst.

## 🔮 Future Work
* Integration with crystallographic databases for automatic phase suggestion.
* Correlation with EDS/EELS data for comprehensive structure-composition relationship analysis.

## 📜 References
* Ophus, C. (2019). *Microscopy and Microanalysis*.
* Ester, M., et al. (1996). *KDD'96*.
* Reynolds, D. (2009). *Encyclopedia of Biometrics*.