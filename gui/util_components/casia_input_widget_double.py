import sys

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CasiaInputWidgetDouble(QWidget):
    """浮点数触屏输入窗口。

    Notes
    -----
    该控件支持两种使用方式：

    1. 独立使用
       - 外部自行调用 ``open_with_value()`` 或 ``open_for_button()``

    2. 即插即用绑定按钮
       - 通过 ``bind_target_button()`` 或 ``create_and_attach()``
       - 点击按钮时自动弹出
       - 确认后自动回写按钮文本

    Parameters
    ----------
    parent : QWidget | None
        父对象。
    min_val : float
        最小值。
    max_val : float
        最大值。
    decimals : int
        小数位数。
    title : str
        标题文本。
    clear_on_first_input : bool
        窗口打开后，第一次输入数字/小数点时是否先清空已有文本。
    auto_apply_to_bound_button : bool
        确认后是否自动回写到绑定按钮。
    """

    valueConfirmed = Signal(float)

    @staticmethod
    def _get_touch_scale() -> float:
        app = QApplication.instance()
        if app is None:
            return 1.0
        try:
            scale = float(app.property("touchScale") or 1.0)
        except (TypeError, ValueError):
            return 1.0
        return max(1.0, scale)

    def _scaled(self, value: int) -> int:
        return int(round(value * self._touch_scale))

    def __init__(
        self,
        parent: QWidget | None = None,
        min_val: float = 0.0,
        max_val: float = 100.0,
        decimals: int = 2,
        title: str = "请输入数字",
        clear_on_first_input: bool = True,
        auto_apply_to_bound_button: bool = True,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.title_text = title

        self.clear_on_first_input = clear_on_first_input
        self.auto_apply_to_bound_button = auto_apply_to_bound_button

        self.input_str = ""
        self._first_input_pending = False

        self._bound_button: QPushButton | None = None
        self._button_text_format: str = ""
        self._button_title_override: str = ""

        self._touch_scale = self._get_touch_scale()
        self._button_width = self._scaled(100)
        self._button_height = self._scaled(74)
        self._grid_spacing = self._scaled(10)

        self._apply_styles()
        self._build_ui()
        self._sync_display()

    # region 便捷工厂

    @classmethod
    def create_and_attach(
        cls,
        button: QPushButton,
        parent: QWidget | None = None,
        min_val: float = 0.0,
        max_val: float = 100.0,
        decimals: int = 2,
        title: str = "请输入数字",
        clear_on_first_input: bool = True,
        auto_apply_to_bound_button: bool = True,
        button_text_format: str = "",
    ) -> "CasiaInputWidgetDouble":
        """创建输入窗口并直接绑定到目标按钮。

        Parameters
        ----------
        button : QPushButton
            目标按钮。
        parent : QWidget | None
            父对象。
        min_val : float
            最小值。
        max_val : float
            最大值。
        decimals : int
            小数位数。
        title : str
            默认标题。
        clear_on_first_input : bool
            首次输入是否先清空。
        auto_apply_to_bound_button : bool
            确认后是否自动写回按钮。
        button_text_format : str
            按钮显示格式，例如 ``"{:.2f}"``。

        Returns
        -------
        CasiaInputWidgetDouble
            创建好的输入窗口实例。
        """
        widget = cls(
            parent=parent,
            min_val=min_val,
            max_val=max_val,
            decimals=decimals,
            title=title,
            clear_on_first_input=clear_on_first_input,
            auto_apply_to_bound_button=auto_apply_to_bound_button,
        )
        widget.set_button_text_format(button_text_format)
        widget.bind_target_button(button)
        return widget

    # endregion

    # region 样式与界面

    def _apply_styles(self) -> None:
        """应用整窗统一样式。"""
        title_font_pt = self._scaled(16)
        display_font_pt = self._scaled(24)
        key_font_pt = self._scaled(16)
        cancel_font_pt = self._scaled(18)
        key_min = self._scaled(54)
        cancel_min = self._scaled(48)

        self.setObjectName("CasiaInputWidgetDouble")
        self.setStyleSheet(
            """
            QWidget#CasiaInputWidgetDouble {{
                background-color: palette(window);
                border: 1px solid palette(mid);
            }}

            QLabel#casia_title_label {{
                font-size: {title_font_pt}pt;
                font-weight: bold;
                color: palette(window-text);
                border: none;
            }}

            QLineEdit#casia_display_edit {{
                background: palette(base);
                font-size: {display_font_pt}pt;
                padding: 5px;
                border: 2px solid #0078D7;
                border-radius: 4px;
                color: palette(text);
            }}

            QPushButton[role="digit"] {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 4px;
                font-size: {key_font_pt}pt;
                color: palette(button-text);
            }}

            QPushButton[role="digit"]:pressed {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: #DDE7F4;
            }}

            QPushButton[role="secondary"] {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 4px;
                font-size: {key_font_pt}pt;
                color: palette(button-text);
            }}

            QPushButton[role="secondary"]:pressed {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: #D8D8D8;
            }}

            QPushButton[role="confirm"] {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: #90EE90;
                border: 1px solid #7CBF7C;
                border-radius: 4px;
                font-size: {key_font_pt}pt;
                font-weight: bold;
                color: #1a1a1a;
            }}

            QPushButton[role="confirm"]:pressed {{
                min-height: {key_min}px;
                min-width: {key_min}px;
                background-color: #7FDD7F;
            }}

            QPushButton#casia_cancel_button {{
                min-height: {cancel_min}px;
                min-width: {cancel_min}px;
                background: transparent;
                border: none;
                color: palette(window-text);
                font-size: {cancel_font_pt}pt;
                font-weight: bold;
            }}

            QPushButton#casia_cancel_button:hover {{
                min-height: {cancel_min}px;
                min-width: {cancel_min}px;
                color: red;
            }}
            """
            .format(
                title_font_pt=title_font_pt,
                display_font_pt=display_font_pt,
                key_font_pt=key_font_pt,
                cancel_font_pt=cancel_font_pt,
                key_min=key_min,
                cancel_min=cancel_min,
            )
        )

    def _build_ui(self) -> None:
        """构建界面。"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(self._scaled(20), self._scaled(14), self._scaled(20), self._scaled(20))
        self.main_layout.setSpacing(self._scaled(12))

        self._build_title_bar()
        self._build_display()
        self._build_keypad()

        total_width = self._scaled(20) + (self._button_width * 4) + (self._grid_spacing * 3) + self._scaled(20)
        total_height = (
            self._scaled(14)
            + self._scaled(40)
            + self._scaled(12)
            + self._scaled(70)
            + self._scaled(12)
            + (self._button_height * 4)
            + (self._grid_spacing * 3)
            + self._scaled(20)
        )
        self.setFixedSize(total_width, total_height)

    def _build_title_bar(self) -> None:
        """构建标题栏。"""
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(self._scaled(8))

        self.title_label = QLabel(self.title_text)
        self.title_label.setObjectName("casia_title_label")

        self.cancel_btn = QPushButton("✕")
        self.cancel_btn.setObjectName("casia_cancel_button")
        self.cancel_btn.setFixedSize(self._scaled(42), self._scaled(42))
        self.cancel_btn.clicked.connect(self.hide)

        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.cancel_btn)

        self.main_layout.addLayout(title_layout)

    def _build_display(self) -> None:
        """构建显示框。"""
        self.display = QLineEdit()
        self.display.setObjectName("casia_display_edit")
        self.display.setReadOnly(True)
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.display.setFixedHeight(self._scaled(70))

        self.main_layout.addWidget(self.display)

    def _build_keypad(self) -> None:
        """构建键盘区。"""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(self._grid_spacing)
        grid.setVerticalSpacing(self._grid_spacing)

        for column in range(4):
            grid.setColumnStretch(column, 1)

        for row in range(4):
            grid.setRowStretch(row, 1)

        grid.addWidget(self._create_digit_button("7"), 0, 0)
        grid.addWidget(self._create_digit_button("8"), 0, 1)
        grid.addWidget(self._create_digit_button("9"), 0, 2)
        grid.addWidget(self._create_backspace_button(), 0, 3)

        grid.addWidget(self._create_digit_button("4"), 1, 0)
        grid.addWidget(self._create_digit_button("5"), 1, 1)
        grid.addWidget(self._create_digit_button("6"), 1, 2)
        grid.addWidget(self._create_clear_button(), 1, 3)

        grid.addWidget(self._create_digit_button("1"), 2, 0)
        grid.addWidget(self._create_digit_button("2"), 2, 1)
        grid.addWidget(self._create_digit_button("3"), 2, 2)
        grid.addWidget(self._create_confirm_button(), 2, 3, 2, 1)

        grid.addWidget(self._create_zero_button(), 3, 0, 1, 2)
        grid.addWidget(self._create_digit_button("."), 3, 2)

        self.main_layout.addLayout(grid)

    # endregion

    # region 按钮工厂

    def _create_digit_button(self, text: str) -> QPushButton:
        """创建普通数字按钮。"""
        btn = QPushButton(text)
        btn.setProperty("role", "digit")
        btn.setFixedSize(self._button_width, self._button_height)
        btn.clicked.connect(lambda checked=False, value=text: self._handle_input(value))
        return btn

    def _create_zero_button(self) -> QPushButton:
        """创建占两列宽度的 0 按钮。"""
        btn = QPushButton("0")
        btn.setProperty("role", "digit")
        btn.setFixedSize(self._button_width * 2 + self._grid_spacing, self._button_height)
        btn.clicked.connect(lambda checked=False: self._handle_input("0"))
        return btn

    def _create_backspace_button(self) -> QPushButton:
        """创建退格按钮。"""
        btn = QPushButton("⌫")
        btn.setProperty("role", "secondary")
        btn.setFixedSize(self._button_width, self._button_height)

        font = QFont("Segoe UI Symbol")
        font.setPointSize(self._scaled(16))
        btn.setFont(font)

        btn.clicked.connect(lambda checked=False: self._handle_input("⌫"))
        return btn

    def _create_clear_button(self) -> QPushButton:
        """创建清除按钮。"""
        btn = QPushButton("清除")
        btn.setProperty("role", "secondary")
        btn.setFixedSize(self._button_width, self._button_height)
        btn.clicked.connect(lambda checked=False: self._handle_input("C"))
        return btn

    def _create_confirm_button(self) -> QPushButton:
        """创建占两行高度的确认按钮。"""
        btn = QPushButton("确认")
        btn.setProperty("role", "confirm")
        btn.setFixedSize(self._button_width, self._button_height * 2 + self._grid_spacing)
        btn.clicked.connect(lambda checked=False: self._handle_input("确认"))
        return btn

    # endregion

    # region 对外 API

    def bind_target_button(
        self,
        button: QPushButton,
        title: str = "",
        button_text_format: str = "",
    ) -> None:
        """绑定目标按钮，实现即插即用。

        Parameters
        ----------
        button : QPushButton
            目标按钮。
        title : str
            绑定该按钮时使用的标题。为空则使用默认标题。
        button_text_format : str
            写回按钮文本格式，例如 ``"{:.2f}"``。
            为空时默认按 ``decimals`` 格式化。
        """
        if self._bound_button is not None:
            try:
                self._bound_button.clicked.disconnect(self._on_bound_button_clicked)
            except (TypeError, RuntimeError):
                pass

        self._bound_button = button
        self._button_title_override = title
        if button_text_format:
            self._button_text_format = button_text_format

        button.clicked.connect(self._on_bound_button_clicked)

    def open_for_button(self, button: QPushButton) -> None:
        """根据按钮当前文本打开输入窗口。

        Parameters
        ----------
        button : QPushButton
            目标按钮。
        """
        current_value = self._parse_float_from_text(button.text())
        self.open_with_value(current_value)

        title = self._button_title_override if self._button_title_override else self.title_text
        self.set_title(title)
        self._move_near_button(button)

    def open_with_value(self, value: float) -> None:
        """用给定初始值打开窗口。

        Parameters
        ----------
        value : float
            初始值。
        """
        self.set_initial_value(value)
        self._first_input_pending = self.clear_on_first_input

        self.show()
        self.raise_()
        self.activateWindow()

    def set_clear_on_first_input(self, enable: bool) -> None:
        """设置首次输入是否先清空原文本。

        Parameters
        ----------
        enable : bool
            是否启用。
        """
        self.clear_on_first_input = enable

    def set_auto_apply_to_bound_button(self, enable: bool) -> None:
        """设置确认后是否自动回写绑定按钮。"""
        self.auto_apply_to_bound_button = enable

    def set_button_text_format(self, fmt: str) -> None:
        """设置按钮回写格式。

        Parameters
        ----------
        fmt : str
            例如 ``"{:.2f}"``。传空字符串则回退为默认格式。
        """
        self._button_text_format = fmt

    def set_value_range(self, min_val: float, max_val: float) -> None:
        """设置数值范围。"""
        self.min_val = min_val
        self.max_val = max_val

    def set_decimals(self, decimals: int) -> None:
        """设置小数位数。"""
        self.decimals = decimals

    def set_initial_value(self, value: float) -> None:
        """设置初始值。"""
        self.input_str = str(round(value, self.decimals))
        self._sync_display()

    def set_title(self, title: str) -> None:
        """设置标题。"""
        self.title_text = title
        self.title_label.setText(title)

    # endregion

    # region 内部逻辑

    def _on_bound_button_clicked(self) -> None:
        """绑定按钮点击后的内部入口。"""
        if self._bound_button is None:
            return
        self.open_for_button(self._bound_button)

    def _parse_float_from_text(self, text: str) -> float:
        """从文本中解析浮点数。"""
        stripped = text.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0

    def _format_value_for_button(self, value: float) -> str:
        """格式化回写到按钮的文本。"""
        if self._button_text_format:
            return self._button_text_format.format(value)
        return f"{value:.{self.decimals}f}"

    def _move_near_button(self, button: QPushButton) -> None:
        """将窗口移动到按钮附近。"""
        global_pos = button.mapToGlobal(button.rect().bottomLeft())
        x = global_pos.x()
        y = global_pos.y() + 6
        self.move(QPoint(x, y))

    def _sync_display(self) -> None:
        """同步显示框文本。"""
        self.display.setText(self.input_str)

    def _consume_first_input_flag_if_needed(self) -> None:
        """如有需要，在首次数字输入前清空旧文本。"""
        if self._first_input_pending:
            self.input_str = ""
            self._first_input_pending = False

    def _handle_input(self, char: str) -> None:
        """处理输入字符。

        Parameters
        ----------
        char : str
            输入字符或控制命令。
        """
        if char == "确认":
            self._on_confirm()
            return

        if char == "C":
            self.input_str = ""
            self._first_input_pending = False
            self._sync_display()
            return

        if char == "⌫":
            self.input_str = self.input_str[:-1]
            self._first_input_pending = False
            self._sync_display()
            return

        if char == ".":
            self._consume_first_input_flag_if_needed()
            if "." not in self.input_str:
                if self.input_str == "":
                    self.input_str = "0."
                else:
                    self.input_str += "."
            self._sync_display()
            return

        # 剩余情况为数字
        self._consume_first_input_flag_if_needed()
        self.input_str += char
        self._sync_display()

    def _on_confirm(self) -> None:
        """确认输入。"""
        try:
            if self.input_str == "":
                self.hide()
                return

            value = float(self.input_str)
            value = max(self.min_val, min(self.max_val, value))
            value = round(value, self.decimals)

            if self.auto_apply_to_bound_button and self._bound_button is not None:
                self._bound_button.setText(self._format_value_for_button(value))

            self.valueConfirmed.emit(value)
            self.hide()
        except ValueError:
            self.input_str = ""
            self.display.setText("Error")

    # endregion

    # region 键盘支持

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """键盘输入支持。"""
        key = event.key()

        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            self._handle_input(event.text())
            return

        if key in (Qt.Key.Key_Period, Qt.Key.Key_Comma):
            self._handle_input(".")
            return

        if key == Qt.Key.Key_Backspace:
            self._handle_input("⌫")
            return

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_confirm()
            return

        if key == Qt.Key.Key_Escape:
            self.hide()
            return

        super().keyPressEvent(event)

    # endregion


if __name__ == "__main__":
    import os
    import sys

    from PySide6.QtWidgets import QMainWindow

    # 设置 QT_QPA_PLATFORM_PLUGIN_PATH 环境变量
    pyside6_dir = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6")
    os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins", "platforms")

    class TestWindow(QMainWindow):
        """测试窗口。"""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("CasiaInputWidgetDouble 测试")
            self.setGeometry(100, 100, 520, 260)

            self.button_a = QPushButton("12.50", self)
            self.button_a.setGeometry(40, 60, 180, 60)

            self.button_b = QPushButton("100.00", self)
            self.button_b.setGeometry(260, 60, 180, 60)

            # 即插即用：一行创建 + 绑定
            self.input_widget_a = CasiaInputWidgetDouble.create_and_attach(
                button=self.button_a,
                parent=self,
                min_val=0.0,
                max_val=999.99,
                decimals=2,
                title="请输入速度",
                clear_on_first_input=True,
                auto_apply_to_bound_button=True,
                button_text_format="{:.2f}",
            )

            # 第二个示例：首次输入不清空，而是接着输入
            self.input_widget_b = CasiaInputWidgetDouble.create_and_attach(
                button=self.button_b,
                parent=self,
                min_val=-1000.0,
                max_val=1000.0,
                decimals=3,
                title="请输入偏移",
                clear_on_first_input=False,
                auto_apply_to_bound_button=True,
                button_text_format="{:.3f}",
            )

            self.input_widget_a.valueConfirmed.connect(self._on_value_a_changed)
            self.input_widget_b.valueConfirmed.connect(self._on_value_b_changed)

        def _on_value_a_changed(self, value: float) -> None:
            print(f"A confirmed: {value}")

        def _on_value_b_changed(self, value: float) -> None:
            print(f"B confirmed: {value}")

    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
