from PySide6.QtCore import QEvent, QModelIndex, Qt, Signal, Slot
from PySide6.QtWidgets import QWidget

from src.utils.Datas import Transform

from .casia_tree import INodeInfoDisplay, PointCloudNode
from .PointCloudInfoWidget_ui import Ui_PointCloudInfoWidget


class PointCloudInfoWidget(QWidget, Ui_PointCloudInfoWidget, INodeInfoDisplay):
    pc_data_changed = Signal(Transform)
    POS_STEP = 1.0
    POS_FINE_STEP = 0.1
    ROT_STEP = 0.1
    ROT_FINE_STEP = 0.01

    @Slot(QModelIndex)
    def update_from_tree_selection(self, index):
        node = index.internalPointer()
        self.show_node_info(node)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._updating = False
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.pc_x_spin.setSingleStep(self.POS_STEP)
        self.pc_y_spin.setSingleStep(self.POS_STEP)
        self.pc_z_spin.setSingleStep(self.POS_STEP)
        self.pc_rz_spin.setSingleStep(self.ROT_STEP)
        self.pc_ry_spin.setSingleStep(self.ROT_STEP)
        self.pc_rx_spin.setSingleStep(self.ROT_STEP)
        self.pc_x_spin.installEventFilter(self)
        self.pc_y_spin.installEventFilter(self)
        self.pc_z_spin.installEventFilter(self)
        self.pc_rz_spin.installEventFilter(self)
        self.pc_ry_spin.installEventFilter(self)
        self.pc_rx_spin.installEventFilter(self)

    def _connect_signals(self):
        self.pc_x_spin.valueChanged.connect(self.on_value_changed)
        self.pc_y_spin.valueChanged.connect(self.on_value_changed)
        self.pc_z_spin.valueChanged.connect(self.on_value_changed)
        self.pc_rz_spin.valueChanged.connect(self.on_value_changed)
        self.pc_ry_spin.valueChanged.connect(self.on_value_changed)
        self.pc_rx_spin.valueChanged.connect(self.on_value_changed)

    def show_node_info(self, node: PointCloudNode):
        if node is None:
            return
        self._updating = True
        try:
            t = Transform.from_SE3(node.transform)
            trans = t.translation
            z, y, x = t.rotation.as_zyx()
            self.pc_index_label.setText(node.display_name)
            self.pc_x_spin.setValue(trans.x)
            self.pc_y_spin.setValue(trans.y)
            self.pc_z_spin.setValue(trans.z)
            self.pc_rz_spin.setValue(z)
            self.pc_ry_spin.setValue(y)
            self.pc_rx_spin.setValue(x)
        finally:
            self._updating = False  # 更新完成

    def get_widget(self) -> QWidget:
        return self

    def on_value_changed(self, value):
        if self._updating:  # 如果是内部更新，直接返回
            return
        x = self.pc_x_spin.value()
        y = self.pc_y_spin.value()
        z = self.pc_z_spin.value()
        rz = self.pc_rz_spin.value()
        ry = self.pc_ry_spin.value()
        rx = self.pc_rx_spin.value()
        new_t = Transform.from_list([x, y, z, rz, ry, rx])
        self.pc_data_changed.emit(new_t)

    def eventFilter(self, watched, event):
        if watched in {
            self.pc_x_spin,
            self.pc_y_spin,
            self.pc_z_spin,
            self.pc_rz_spin,
            self.pc_ry_spin,
            self.pc_rx_spin,
        } and event.type() == QEvent.Type.Wheel:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                step_num = int(event.angleDelta().y() / 120)
                if step_num != 0:
                    fine_step = self.POS_FINE_STEP if watched in {
                        self.pc_x_spin,
                        self.pc_y_spin,
                        self.pc_z_spin,
                    } else self.ROT_FINE_STEP
                    watched.setValue(float(watched.value()) + float(step_num) * float(fine_step))
                    event.accept()
                    return True
        return super().eventFilter(watched, event)
