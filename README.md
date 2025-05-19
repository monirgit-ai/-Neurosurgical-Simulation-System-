Preoperative Neurosurgical Training Simulator

A Python-based GUI application for simulating **tearing of the cerebral dura mater** using real medical images, force modeling, and finite element simulation.

## 🚀 Project Overview

Theseus is a research-grade simulation system designed to support neurosurgical training. The application allows users to:
- Load and segment brain scans
- Simulate surgical force interactions
- Visualize tearing effects on ROI (e.g., dura mater)
- Run stress distribution FEM simulations
- Generate analytical reports and data summaries

This system supports manual interaction through a GUI and integrates human-computer modeling, medical image processing, and biomechanical simulation.

---

## 🧩 Features

- 📁 **Medical Image Loader** (.dcm, .png, .jpg)
- 🖼️ **Manual & Auto Segmentation**
- 🛠️ **Tool-Based Force Simulation** (Scalpel, Forceps, Scissors)
- 🧮 **FEM Simulation (skfem)** for stress distribution
- 🎛️ **Material Parameter & Boundary Condition Controls**
- 📊 **Live Tearing Visualization**
- 🗃️ **SQLite Logging of Simulation Sessions**
- 📈 **Statistical Analysis and CSV Export**
- 📝 **PDF Report Generation**

---

## 🖥️ GUI Demo

| Load Image | Segment | Simulate Force | Run FEM |
|------------|---------|----------------|---------|
| ✅ DICOM Support | ✅ Auto/Manual | ✅ Tearing Thresholds | ✅ Stress Field Overlays |

---

## 📦 Installation

### 🔧 Requirements

- Python 3.10
- PyQt5
- NumPy 1.24
- SciPy 1.11
- Matplotlib
- OpenCV
- scikit-fem==7.0.0
- fpdf

### 🔄 Setup Instructions

```bash
git clone https://github.com/yourusername/theseus-neurosurgical-trainer.git
cd theseus-neurosurgical-trainer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
