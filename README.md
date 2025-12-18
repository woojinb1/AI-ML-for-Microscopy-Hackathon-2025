# Automated Crystalline Domain Segmentation in Polycrystalline TEM Images

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Introduction
[cite_start]This project presents an **automated, GUI-based workflow** for analyzing High-Resolution Transmission Electron Microscopy (HRTEM) images[cite: 8]. [cite_start]By integrating **Gaussian Mixture Models (GMM)** for FFT peak classification and **DBSCAN** for real-space domain extraction, this tool automates the traditionally subjective and time-consuming process of crystallographic characterization[cite: 8, 11].

[cite_start]It is designed to address challenges in analyzing polycrystalline materials with weak crystallinity, such as solid electrolyte interphase (SEI) layers in lithium-metal batteries[cite: 7, 13].

## 👥 Authors
* [cite_start]**Woojin Bae**, Shihyun Kim, Jinho Rhee [cite: 3]
* [cite_start]School of Chemical and Biological Engineering, Seoul National University [cite: 4]

## 🚀 Key Features
* [cite_start]**Automated FFT Peak Detection:** Uses a two-component GMM to unsupervisedly classify peaks as signal (crystalline) or noise based on intensity and sharpness[cite: 23, 24].
* [cite_start]**d-Spacing Calculation:** Automatically calculates d-spacing values from peak positions using pixel size calibration for phase identification[cite: 27].
* [cite_start]**Domain Clustering:** Reconstructs real-space images via inverse FFT and uses DBSCAN to group crystalline domains, effectively handling irregular grain boundaries[cite: 28, 29, 30].
* [cite_start]**Interactive GUI:** Allows users to adjust sensitivity, filter non-physical peaks, and visualize results without coding experience[cite: 26, 51].

## 🛠 Methodology
[cite_start]The pipeline consists of four integrated stages[cite: 23, 32]:
1.  [cite_start]**GMM-Based Peak Classification:** Distinguishes crystalline signals from background noise in the 2D FFT spectrum[cite: 23].
2.  [cite_start]**Automated d-Spacing Calculation:** Converts peak positions to crystallographic data[cite: 27].
3.  [cite_start]**DBSCAN-Based Domain Clustering:** Identifies spatially connected high-intensity regions in the reconstructed image[cite: 29].
4.  [cite_start]**Visualization & Quantification:** Overlays color-coded domains on the original image and calculates domain areas[cite: 32, 33].

## 📊 Results
* [cite_start]Successfully analyzes polycrystalline HRTEM images with weak crystallinity and phase heterogeneity[cite: 35].
* [cite_start]Identifies multiple coexisting crystalline phases and measures crystallite sizes ranging from nanometers to tens of nanometers[cite: 42].
* [cite_start]Significantly reduces analysis time compared to manual methods while ensuring reproducibility[cite: 38].

## 🔮 Future Work
* [cite_start]Integration with crystallographic databases for automatic phase suggestion[cite: 48].
* [cite_start]Correlation with EDS/EELS data for structure-composition relationship analysis[cite: 49].

## 📜 References
* Ophus, C. (2019). *Microsc. [cite_start]Microanal.* [cite: 54]
* Ester, M., et al. (1996). [cite_start]*KDD'96*. [cite: 57]
* Reynolds, D. (2009). [cite_start]*Encyclopedia of Biometrics*. [cite: 59]