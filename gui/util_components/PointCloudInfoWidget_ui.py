# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PointCloudInfoWidget.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QGridLayout, QLabel,
    QSizePolicy, QWidget)

class Ui_PointCloudInfoWidget(object):
    def setupUi(self, PointCloudInfoWidget):
        if not PointCloudInfoWidget.objectName():
            PointCloudInfoWidget.setObjectName(u"PointCloudInfoWidget")
        PointCloudInfoWidget.resize(250, 118)
        self.gridLayout = QGridLayout(PointCloudInfoWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_16 = QLabel(PointCloudInfoWidget)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout.addWidget(self.label_16, 0, 0, 1, 1)

        self.pc_index_label = QLabel(PointCloudInfoWidget)
        self.pc_index_label.setObjectName(u"pc_index_label")

        self.gridLayout.addWidget(self.pc_index_label, 0, 1, 1, 1)

        self.label_13 = QLabel(PointCloudInfoWidget)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout.addWidget(self.label_13, 1, 0, 1, 1)

        self.pc_x_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_x_spin.setObjectName(u"pc_x_spin")
        self.pc_x_spin.setMinimum(-99999.000000000000000)
        self.pc_x_spin.setMaximum(99999.000000000000000)

        self.gridLayout.addWidget(self.pc_x_spin, 1, 1, 1, 1)

        self.label_12 = QLabel(PointCloudInfoWidget)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout.addWidget(self.label_12, 1, 2, 1, 1)

        self.pc_rz_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_rz_spin.setObjectName(u"pc_rz_spin")
        self.pc_rz_spin.setMinimum(-99999.000000000000000)
        self.pc_rz_spin.setMaximum(99999.000000000000000)
        self.pc_rz_spin.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.pc_rz_spin, 1, 3, 1, 1)

        self.label_9 = QLabel(PointCloudInfoWidget)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 2, 0, 1, 1)

        self.pc_y_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_y_spin.setObjectName(u"pc_y_spin")
        self.pc_y_spin.setMinimum(-99999.000000000000000)
        self.pc_y_spin.setMaximum(99999.000000000000000)

        self.gridLayout.addWidget(self.pc_y_spin, 2, 1, 1, 1)

        self.label_15 = QLabel(PointCloudInfoWidget)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout.addWidget(self.label_15, 2, 2, 1, 1)

        self.pc_ry_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_ry_spin.setObjectName(u"pc_ry_spin")
        self.pc_ry_spin.setMinimum(-99999.000000000000000)
        self.pc_ry_spin.setMaximum(99999.000000000000000)
        self.pc_ry_spin.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.pc_ry_spin, 2, 3, 1, 1)

        self.label_7 = QLabel(PointCloudInfoWidget)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)

        self.pc_z_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_z_spin.setObjectName(u"pc_z_spin")
        self.pc_z_spin.setMinimum(-99999.000000000000000)
        self.pc_z_spin.setMaximum(99999.000000000000000)

        self.gridLayout.addWidget(self.pc_z_spin, 3, 1, 1, 1)

        self.label_14 = QLabel(PointCloudInfoWidget)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout.addWidget(self.label_14, 3, 2, 1, 1)

        self.pc_rx_spin = QDoubleSpinBox(PointCloudInfoWidget)
        self.pc_rx_spin.setObjectName(u"pc_rx_spin")
        self.pc_rx_spin.setMinimum(-99999.000000000000000)
        self.pc_rx_spin.setMaximum(99999.000000000000000)
        self.pc_rx_spin.setSingleStep(0.100000000000000)

        self.gridLayout.addWidget(self.pc_rx_spin, 3, 3, 1, 1)


        self.retranslateUi(PointCloudInfoWidget)

        QMetaObject.connectSlotsByName(PointCloudInfoWidget)
    # setupUi

    def retranslateUi(self, PointCloudInfoWidget):
        PointCloudInfoWidget.setWindowTitle(QCoreApplication.translate("PointCloudInfoWidget", u"Form", None))
        self.label_16.setText(QCoreApplication.translate("PointCloudInfoWidget", u"\u7d22\u5f15", None))
        self.pc_index_label.setText(QCoreApplication.translate("PointCloudInfoWidget", u"-1", None))
        self.label_13.setText(QCoreApplication.translate("PointCloudInfoWidget", u"X", None))
        self.label_12.setText(QCoreApplication.translate("PointCloudInfoWidget", u"RZ", None))
        self.label_9.setText(QCoreApplication.translate("PointCloudInfoWidget", u"Y", None))
        self.label_15.setText(QCoreApplication.translate("PointCloudInfoWidget", u"RY", None))
        self.label_7.setText(QCoreApplication.translate("PointCloudInfoWidget", u"Z", None))
        self.label_14.setText(QCoreApplication.translate("PointCloudInfoWidget", u"RX", None))
    # retranslateUi

