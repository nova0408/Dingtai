from dataclasses import dataclass


@dataclass(frozen=True)
class CasiaQss:
    """Casia 样式 QSS 类"""

    SPIN_BOX = """
            QSpinBox {
            padding-top: 2px;
            padding-bottom: 2px;
            padding-left: 4px;
            padding-right: 15px;
            border: 1px solid rgb(64,64,64);
            border-radius: 3px;
            color: rgb(200,200,200);
            background-color: rgb(44,44,44);
            selection-color: rgb(235,235,235);
            selection-background-color: rgb(83,121,180);
            font-family: "Microsoft Yahei";
            font-size: 14pt;
            min-height: 48px;
        }

        QSpinBox:hover {
            background-color: rgb(59,59,59);
        }

        QSpinBox::up-button { /* 向上按钮 */
            subcontrol-origin: border; /* 起始位置 */
            subcontrol-position: top right; /* 居于右上角 */
            border: none;
            width: 20px;
            margin-top: 2px;
            margin-right: 1px;
            margin-bottom: 0px;
        }

        QSpinBox::up-button:hover {
            border: none;
        }

        QSpinBox::up-arrow { /* 向上箭头，图片大小为 8x8 */
            image: url(:/Resources/up.png);
        }

        QSpinBox::up-arrow:hover {
            image: url(:/Resources/up.png);
        }

        QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off {
            image: url(:/Resources/up.png);
        }

        QSpinBox::down-button { /* 向下按钮 */
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            border: none;
            width: 20px;
            margin-top: 0px;
            margin-right: 1px;
            margin-bottom: 2px;
        }

        QSpinBox::down-button:hover {
            border: none;
        }

        QSpinBox::down-arrow { /* 向下箭头 */
            image: url(:/Resources/down.png);
        }

        QSpinBox::down-arrow:hover {
            image: url(:/Resources/down.png);
        }

        QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off {
            image: url(:/Resources/down.png);
        }
    """

    TREE_VIEW = """
    QTreeView {
        show-decoration-selected: 1;
        outline: 0;
        border: none;
        font-size: 16px;
    }
    QTreeView::item {
        min-height: 42px;
    }
    QTreeView::branch {
        width: 28px;
    }
    """
