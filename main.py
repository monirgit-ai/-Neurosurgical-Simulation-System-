import sys
import os
import numpy as np
import cv2
from enum import Enum
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QTextEdit, QSlider, QProgressBar, QHBoxLayout,
    QSizePolicy, QMessageBox, QGroupBox, QRadioButton, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from image_loader import load_image
from image_preprocessing import preprocess_image, auto_segment, manual_segment
from force_simulation import TearSimulator
from visual_overlay import draw_crack_line, draw_roi_boundary
from database import init_db, log_simulation
from report_generator import generate_pdf_report
from analysis import analyze_simulations
from skfem import *
from skfem.helpers import dot, sym_grad, trace
from skfem.visuals.matplotlib import draw, plot
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skfem import MeshTri




class InstrumentType(Enum):
    SCALPEL = "Scalpel"
    FORCEPS = "Forceps"
    SCISSORS = "Scissors"

TOOL_PROFILES = {
    InstrumentType.SCALPEL: {"stiffness": 6.0, "threshold": 20.0},
    InstrumentType.FORCEPS: {"stiffness": 4.0, "threshold": 25.0},
    InstrumentType.SCISSORS: {"stiffness": 5.0, "threshold": 30.0}
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Theseus Neurosurgical Trainer")
        self.resize(1200, 600)

        self.image = None
        self.metadata = {}
        self.mask = None
        self.simulator = TearSimulator()
        init_db()

        # Image viewer
        self.image_label = QLabel("No image loaded")
        self.image_label.setFixedSize(512, 512)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Controls
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        self.preprocess_button = QPushButton("Preprocess")
        self.preprocess_button.clicked.connect(self.run_preprocessing)

        self.auto_seg_button = QPushButton("Auto Segment")
        self.auto_seg_button.clicked.connect(self.run_auto_segmentation)

        self.manual_seg_button = QPushButton("Manual Segment")
        self.manual_seg_button.clicked.connect(self.run_manual_segmentation)

        self.analysis_button = QPushButton("Show Analysis")
        self.analysis_button.clicked.connect(self.show_analysis)

        self.report_button = QPushButton("Generate Report")
        self.report_button.clicked.connect(self.generate_report)

        self.stress_button = QPushButton("Simulate Stress")
        self.stress_button.clicked.connect(self.run_stress_simulation)

        self.fem_button = QPushButton("Run FEM Simulation")
        self.fem_button.clicked.connect(self.run_fem_simulation)

        # FEM Inputs
        self.material_dropdown = QComboBox()
        self.material_dropdown.addItems(["Linear Elastic", "Neo-Hookean"])

        self.youngs_input = QLineEdit("3000")
        self.poisson_input = QLineEdit("0.45")

        self.bc_dropdown = QComboBox()
        self.bc_dropdown.addItems(["Fixed bottom", "Fixed left"])

        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setFixedHeight(100)

        self.force_slider = QSlider(Qt.Horizontal)
        self.force_slider.setMinimum(0)
        self.force_slider.setMaximum(100)
        self.force_slider.setValue(0)
        self.force_slider.valueChanged.connect(self.apply_force)

        self.force_label = QLabel("Force: 0.0 N")
        self.tool_label = QLabel("Tool: Scalpel")
        self.status_label = QLabel("Status: Intact")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.tear_progress = QProgressBar()

        # Tool selection
        self.tool_group = QGroupBox("Select Tool")
        self.scalpel_radio = QRadioButton("Scalpel")
        self.forceps_radio = QRadioButton("Forceps")
        self.scissors_radio = QRadioButton("Scissors")
        self.scalpel_radio.setChecked(True)
        self.scalpel_radio.toggled.connect(self.update_tool)
        self.forceps_radio.toggled.connect(self.update_tool)
        self.scissors_radio.toggled.connect(self.update_tool)
        tool_layout = QVBoxLayout()
        for r in [self.scalpel_radio, self.forceps_radio, self.scissors_radio]:
            tool_layout.addWidget(r)
        self.tool_group.setLayout(tool_layout)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.preprocess_button)
        left_layout.addWidget(self.auto_seg_button)
        left_layout.addWidget(self.manual_seg_button)
        left_layout.addWidget(self.tool_group)
        left_layout.addWidget(self.tool_label)
        left_layout.addWidget(self.analysis_button)
        left_layout.addWidget(self.report_button)
        left_layout.addWidget(self.stress_button)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Finite Element Simulation"))
        left_layout.addWidget(QLabel("Material Model"))
        left_layout.addWidget(self.material_dropdown)
        left_layout.addWidget(QLabel("Young's Modulus (MPa)"))
        left_layout.addWidget(self.youngs_input)
        left_layout.addWidget(QLabel("Poisson's Ratio"))
        left_layout.addWidget(self.poisson_input)
        left_layout.addWidget(QLabel("Boundary Condition"))
        left_layout.addWidget(self.bc_dropdown)
        left_layout.addWidget(self.fem_button)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Image Metadata:"))
        left_layout.addWidget(self.meta_text)
        left_layout.addWidget(QLabel("\U0001F9EA Apply Force for Tear Simulation"))
        left_layout.addWidget(self.force_slider)
        left_layout.addWidget(self.force_label)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.tear_progress)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(520)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.update_tool()

    def update_tool(self):
        tool = InstrumentType.SCALPEL if self.scalpel_radio.isChecked() else (
            InstrumentType.FORCEPS if self.forceps_radio.isChecked() else InstrumentType.SCISSORS)
        profile = TOOL_PROFILES[tool]
        self.simulator.set_profile(profile["stiffness"], profile["threshold"])
        self.tool_label.setText(f"Tool: {tool.value} (Stiffness={profile['stiffness']}, Threshold={profile['threshold']})")

    def show_image(self, img_array):
        if len(img_array.shape) == 2:
            height, width = img_array.shape
            qimg = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            height, width, ch = img_array.shape
            qimg = QImage(img_array.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(512, 512, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.dcm *.png *.jpg)")
        if not path:
            return
        self.image, self.metadata = load_image(path)
        self.mask = None
        self.simulator.reset()
        self.force_slider.setValue(0)
        self.apply_force()
        self.show_image(self.image)
        self.meta_text.setText("\n".join(f"{k}: {v}" for k, v in self.metadata.items()))

    def run_preprocessing(self):
        if self.image is not None:
            self.image = preprocess_image(self.image)
            self.show_image(self.image)

    def run_auto_segmentation(self):
        if self.image is not None:
            self.mask = auto_segment(self.image)
            self.show_image(draw_roi_boundary(self.image, self.mask))

    def run_manual_segmentation(self):
        if self.image is not None:
            self.mask = manual_segment(self.image)
            self.show_image(draw_roi_boundary(self.image, self.mask))

    def apply_force(self):
        displacement = self.force_slider.value() / 10.0
        force, torn, progress = self.simulator.apply_displacement(displacement)
        self.force_label.setText(f"Force: {force:.2f} N")
        self.tear_progress.setValue(int(progress))
        if torn:
            self.status_label.setText("Status: TORN!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            overlay = draw_crack_line(self.image, self.mask) if self.mask is not None else self.image
            self.show_image(overlay)
        else:
            self.status_label.setText("Status: Intact")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            overlay = draw_roi_boundary(self.image, self.mask) if self.mask is not None else self.image
            self.show_image(overlay)

    def run_stress_simulation(self):
        from stress_simulation import compute_stress_distribution, visualize_stress
        try:
            if self.mask is not None:
                m, disp = compute_stress_distribution(self.mask, applied_force=self.simulator.force)
                visualize_stress(m, disp)
            else:
                QMessageBox.warning(self, "Error", "Please segment an ROI first.")
        except Exception as e:
            print("[DEBUG] Stress Simulation Failed:", e)
            QMessageBox.critical(self, "Stress Simulation Failed", str(e))

    def run_fem_simulation(self):
        try:
            if self.mask is None:
                QMessageBox.warning(self, "Error", "Please segment an ROI first.")
                return

            from skfem import MeshTri, ElementTriP1, ElementVector, Basis, asm, solve, condense
            from skfem.helpers import sym_grad, trace, dot
            from skfem.visuals.matplotlib import draw, plot
            import matplotlib.pyplot as plt
            import numpy as np
            import skfem
            from skfem import MeshTri
            m = MeshTri.init_symmetric()
            print(type(m))
            print(skfem.__version__)

            # --- FEM Inputs ---
            E = float(self.youngs_input.text()) * 1e6  # MPa to Pa
            nu = float(self.poisson_input.text())
            boundary = self.bc_dropdown.currentText()
            print(f"[DEBUG] E={E}, nu={nu}, boundary={boundary}")

            # --- Generate unit square mesh ---
            m = MeshTri.init_symmetric().refined(5)  # 32x32 uniform mesh

            # --- Scale mesh to image dimensions ---
            h, w = self.mask.shape
            coords = m.p.T * [w, h]  # scale unit mesh to image space

            # --- Create triangle mask using image segmentation ---
            img_mask = (self.mask > 0).astype(np.uint8)  # binary mask
            tri_pts = coords[m.t.T]  # shape: (N_tri, 3, 2)
            centers = tri_pts.mean(axis=1).astype(int)  # shape: (N_tri, 2)

            # Ensure centers are within image bounds
            centers[:, 0] = np.clip(centers[:, 0], 0, w - 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, h - 1)

            in_roi = img_mask[centers[:, 1], centers[:, 0]] > 0
            m.t = m.t[:, in_roi]

            # 🔁 Force clean by reinitializing from valid triangles and used nodes only
            used_indices = np.unique(m.t)
            p_clean = m.p[:, used_indices]
            index_map = {old: new for new, old in enumerate(used_indices)}
            t_clean = np.vectorize(index_map.get)(m.t)

            # Recreate the cleaned mesh
            from skfem import MeshTri
            m = MeshTri(p_clean, t_clean.astype(np.int32))  # Ensure int dtype

            print(f"[DEBUG] Cleaned mesh: p {m.p.shape}, t {m.t.shape}")


            print(f"[DEBUG] Filtered mesh: nodes={m.p.shape}, triangles={m.t.shape}")
            m = m.with_orientation()  #

            basis = Basis(m, ElementVector(ElementTriP1()))

            # --- Material constants ---
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))

            @BilinearForm
            def a(u, v, w):
                eps_u = sym_grad(u)
                eps_v = sym_grad(v)
                return lam * trace(eps_u) * trace(eps_v) + 2 * mu * dot(eps_u, eps_v)

            @LinearForm
            def l(v, w):
                return dot([0.0, -1e4], v)

            A = asm(a, basis)
            b = asm(l, basis)

            # --- Boundary conditions ---
            if boundary == "Fixed bottom":
                fixed = m.nodes_satisfying(lambda x: x[1] < 1e-3)
            elif boundary == "Fixed left":
                fixed = m.nodes_satisfying(lambda x: x[0] < 1e-3)
            else:
                fixed = []

            D = basis.get_dofs(fixed, components=[0, 1])
            x = solve(*condense(A, b, D=D))

            # --- Postprocessing displacement ---
            u = x.reshape((basis.N, 2))
            disp_mag = np.linalg.norm(u, axis=1)
            field = basis.interpolate(disp_mag)

            # --- Visualization ---
            fig, ax = plt.subplots()
            draw(m, ax=ax, linewidth=0.3)
            plot(m, field, ax=ax, shading='gouraud')
            ax.set_title("FEM Deformation Field (Structured Mesh)")
            plt.colorbar(ax.collections[0], ax=ax, label="Displacement")
            plt.show()

        except Exception as e:
            print(f"[DEBUG] FEM Simulation Failed: {e}")
            QMessageBox.critical(self, "FEM Simulation Error", str(e))

    def show_analysis(self):
        try:
            analyze_simulations()
            QMessageBox.information(self, "Analysis", "Exported to simulation_summary.csv")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def generate_report(self):
        if self.image is not None:
            filename = self.metadata.get("Filename", "Unknown")
            roi_area = int(np.sum(self.mask > 0)) if self.mask is not None else 0
            force = self.simulator.force
            torn = self.simulator.torn
            generate_pdf_report(filename, roi_area, force, torn)
            QMessageBox.information(self, "PDF Report", "Generated!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
