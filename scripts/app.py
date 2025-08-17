# VMD Motion Optimizer by Barış Keser (barkeser2002)
# License: GNU General Public License v3.0 (GPL-3.0)
# See LICENSE for details.

import os
import sys
import json
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets, QtGui

from optimize_vmd import optimize_vmd


DEFAULT_PROFILES = {
    "Quality": {"pos_eps": 0.02, "rot_eps_deg": 0.25, "morph_eps": 0.0005, "key_step": 1},
    "Balanced": {"pos_eps": 0.05, "rot_eps_deg": 0.5, "morph_eps": 0.001, "key_step": 1},
    "Aggressive": {"pos_eps": 0.1, "rot_eps_deg": 1.0, "morph_eps": 0.002, "key_step": 2},
}


def _resource_path(name: str) -> Optional[str]:
    # PyInstaller (MEIPASS) ve kaynak dizinlerini dene
    candidates = []
    base = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base, name))
    candidates.append(os.path.join(os.path.dirname(base), name))
    if getattr(sys, '_MEIPASS', None):
        candidates.insert(0, os.path.join(sys._MEIPASS, name))  # type: ignore[attr-defined]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _read_version() -> str:
    for rel in ("version.txt", os.path.join("..", "version.txt")):
        p = _resource_path(rel) if not os.path.isabs(rel) else rel
        if p and os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass
    return "dev"


class Worker(QtCore.QThread):
    progress_signal = QtCore.pyqtSignal(str, int, int)
    done_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            def on_progress(section: str, i: int, total: int):
                self.progress_signal.emit(section, i, total)
            out = optimize_vmd(progress=on_progress, **self.params)
            self.done_signal.emit(out)
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ver = _read_version()
        self.setWindowTitle(f"VMD Motion Optimizer by Barış Keser (barkeser2002) v{ver}")
        self.resize(700, 600)

        # Üstte logo
        logo_path = _resource_path('logo.png')
        self.logo_label = QtWidgets.QLabel()
        if logo_path:
            pix = QtGui.QPixmap(logo_path)
            if not pix.isNull():
                scaled = pix.scaledToWidth(700, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled)
                self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.logo_label.setMinimumWidth(700)
        
        # Inputs
        self.input_edit = QtWidgets.QLineEdit()
        self.in_btn = QtWidgets.QPushButton("VMD Seç...")
        self.output_edit = QtWidgets.QLineEdit()
        self.out_btn = QtWidgets.QPushButton("Çıktı...")

        self.pos_eps = QtWidgets.QDoubleSpinBox(); self.pos_eps.setRange(0.0, 10.0); self.pos_eps.setValue(0.05); self.pos_eps.setSingleStep(0.01)
        self.rot_eps = QtWidgets.QDoubleSpinBox(); self.rot_eps.setRange(0.0, 45.0); self.rot_eps.setValue(0.5); self.rot_eps.setSingleStep(0.1)
        self.morph_eps = QtWidgets.QDoubleSpinBox(); self.morph_eps.setRange(0.0, 1.0); self.morph_eps.setDecimals(5); self.morph_eps.setValue(0.001); self.morph_eps.setSingleStep(0.0005)
        self.key_step = QtWidgets.QSpinBox(); self.key_step.setRange(1, 10); self.key_step.setValue(1)

        # Depth alignment removal
        self.depth_check = QtWidgets.QCheckBox("Depth (Z) hizasını kaldır")
        self.depth_smooth = QtWidgets.QSpinBox(); self.depth_smooth.setRange(0, 100); self.depth_smooth.setValue(0)
        self.depth_scale = QtWidgets.QDoubleSpinBox(); self.depth_scale.setRange(0.0, 10.0); self.depth_scale.setValue(1.0); self.depth_scale.setSingleStep(0.1)
        # Ground stabilization
        self.ground_check = QtWidgets.QCheckBox("Ground stabilization (Y)")
        self.ground_target = QtWidgets.QDoubleSpinBox(); self.ground_target.setRange(-1000.0, 1000.0); self.ground_target.setValue(0.0)
        self.ground_smooth = QtWidgets.QSpinBox(); self.ground_smooth.setRange(0, 200); self.ground_smooth.setValue(0)
        self.ground_scale = QtWidgets.QDoubleSpinBox(); self.ground_scale.setRange(0.0, 10.0); self.ground_scale.setValue(1.0); self.ground_scale.setSingleStep(0.1)
        self.ground_all_bones = QtWidgets.QCheckBox("Tüm kemikleri kullan (varsayılan: ayak)")

        # Profiles
        self.profile_combo = QtWidgets.QComboBox(); self.profile_combo.addItems(DEFAULT_PROFILES.keys())
        self.save_profile_btn = QtWidgets.QPushButton("Profili Kaydet")
        self.load_profile_btn = QtWidgets.QPushButton("Profilleri Dışa Aktar/İçe Al")

        self.start_btn = QtWidgets.QPushButton("Optimize Et")
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)

        # Credit etiketi
        self.credit = QtWidgets.QLabel(f"VMD Motion Optimizer by Barış Keser (barkeser2002) — GPL-3.0 — v{ver}")
        self.credit.setStyleSheet("color: gray; font-size: 11px;")

        form = QtWidgets.QFormLayout()
        form.addRow("Girdi VMD:", self._with_btn(self.input_edit, self.in_btn))
        form.addRow("Çıktı VMD:", self._with_btn(self.output_edit, self.out_btn))
        form.addRow("Pozisyon eps:", self.pos_eps)
        form.addRow("Rotasyon eps (deg):", self.rot_eps)
        form.addRow("Morph eps:", self.morph_eps)
        form.addRow("Key step:", self.key_step)
        form.addRow(self.depth_check)
        form.addRow("Depth smooth window:", self.depth_smooth)
        form.addRow("Depth scale:", self.depth_scale)
        form.addRow(self.ground_check)
        form.addRow("Ground target Y:", self.ground_target)
        form.addRow("Ground smooth window:", self.ground_smooth)
        form.addRow("Ground scale:", self.ground_scale)
        form.addRow(self.ground_all_bones)
        form.addRow("Profil:", self._with_btn(self.profile_combo, self.save_profile_btn))
        form.addRow(self.load_profile_btn)

        v = QtWidgets.QVBoxLayout(self)
        if logo_path:
            v.addWidget(self.logo_label)
        v.addLayout(form)
        v.addWidget(self.start_btn)
        v.addWidget(self.progress)
        v.addWidget(self.log)
        v.addWidget(self.credit)

        # signals
        self.in_btn.clicked.connect(self.select_input)
        self.out_btn.clicked.connect(self.select_output)
        self.start_btn.clicked.connect(self.start)
        self.profile_combo.currentTextChanged.connect(self.apply_profile)
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.load_profile_btn.clicked.connect(self.import_export_profiles)

        self._profiles = dict(DEFAULT_PROFILES)
        self.apply_profile(self.profile_combo.currentText())

        self.worker: Optional[Worker] = None
        self._section_total = 0
        self._section_done = 0

    def _with_btn(self, widget, btn):
        h = QtWidgets.QHBoxLayout(); w = QtWidgets.QWidget()
        h.addWidget(widget); h.addWidget(btn); w.setLayout(h)
        return w

    def log_text(self, s: str):
        self.log.appendPlainText(s)

    def select_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "VMD seç", str(Path.cwd()), "VMD Files (*.vmd)")
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text():
                out = os.path.splitext(path)[0] + "_optimized.vmd"
                self.output_edit.setText(out)

    def select_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Çıktı yolu", self.output_edit.text() or str(Path.cwd()), "VMD Files (*.vmd)")
        if path:
            self.output_edit.setText(path)

    def apply_profile(self, name: str):
        p = self._profiles.get(name)
        if not p:
            return
        self.pos_eps.setValue(float(p.get("pos_eps", self.pos_eps.value())))
        self.rot_eps.setValue(float(p.get("rot_eps_deg", self.rot_eps.value())))
        self.morph_eps.setValue(float(p.get("morph_eps", self.morph_eps.value())))
        self.key_step.setValue(int(p.get("key_step", self.key_step.value())))

    def save_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Profil adı", "Ad")
        if not ok or not name:
            return
        self._profiles[name] = {
            "pos_eps": self.pos_eps.value(),
            "rot_eps_deg": self.rot_eps.value(),
            "morph_eps": self.morph_eps.value(),
            "key_step": self.key_step.value(),
        }
        self.profile_combo.clear(); self.profile_combo.addItems(self._profiles.keys())
        self.profile_combo.setCurrentText(name)

    def import_export_profiles(self):
        menu = QtWidgets.QMenu()
        act_export = menu.addAction("Dışa aktar JSON")
        act_import = menu.addAction("İçe al JSON")
        action = menu.exec(self.cursor().pos())
        if action == act_export:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Profilleri dışa aktar", str(Path.cwd()/"profiles.json"), "JSON (*.json)")
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self._profiles, f, ensure_ascii=False, indent=2)
                self.log_text("Profiller kaydedildi: " + path)
        elif action == act_import:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Profilleri içe al", str(Path.cwd()), "JSON (*.json)")
            if path:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._profiles = json.load(f)
                    self.profile_combo.clear(); self.profile_combo.addItems(self._profiles.keys())
                    self.log_text("Profiller yüklendi: " + path)
                except Exception as e:
                    self.log_text("Hata: " + str(e))

    def start(self):
        inp = self.input_edit.text().strip()
        outp = self.output_edit.text().strip()
        if not inp or not os.path.exists(inp):
            self.log_text("Geçerli giriş seçin")
            return
        if not outp:
            outp = os.path.splitext(inp)[0] + "_optimized.vmd"
            self.output_edit.setText(outp)

        params = dict(
            input_path=inp,
            output_path=outp,
            pos_eps=self.pos_eps.value(),
            rot_eps_deg=self.rot_eps.value(),
            morph_eps=self.morph_eps.value(),
            key_step=self.key_step.value(),
            preserve_end_keys=True,
            remove_depth=self.depth_check.isChecked(),
            depth_smooth_window=self.depth_smooth.value(),
            depth_scale=self.depth_scale.value(),
            stabilize_ground_flag=self.ground_check.isChecked(),
            ground_target_y=self.ground_target.value(),
            ground_use_feet_only=not self.ground_all_bones.isChecked(),
            ground_smooth_window=self.ground_smooth.value(),
            ground_scale=self.ground_scale.value(),
        )

        self.progress.setValue(0)
        self.log_text("Başladı...")

        self.worker = Worker(params)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.done_signal.connect(self.on_done)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    @QtCore.pyqtSlot(str, int, int)
    def on_progress(self, section: str, i: int, total: int):
        # Basit yüzdelik: Bones %50 + Morphs %50
        if section == 'Bones':
            pct = int((i / max(total, 1)) * 50)
        else:
            pct = 50 + int((i / max(total, 1)) * 50)
        self.progress.setValue(max(0, min(100, pct)))

    @QtCore.pyqtSlot(str)
    def on_done(self, out: str):
        self.progress.setValue(100)
        self.log_text("Bitti. Kaydedildi: " + out)

    @QtCore.pyqtSlot(str)
    def on_error(self, msg: str):
        self.log_text("Hata: " + msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("VMD Motion Optimizer by Barış Keser (barkeser2002)")
    # Uygulama ikonu: önce icon.ico, yoksa logo.png
    icon_path = _resource_path('icon.ico')
    if not icon_path:
        icon_path = _resource_path('logo.png')
    if icon_path:
        app.setWindowIcon(QtGui.QIcon(icon_path))
    w = MainWindow()
    # Pencere ikonu da aynı şekilde
    if icon_path:
        w.setWindowIcon(QtGui.QIcon(icon_path))
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
