# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UtilDoFWidget.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(363, 194)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.dof_label = QLabel(Form)
        self.dof_label.setObjectName(u"dof_label")

        self.horizontalLayout.addWidget(self.dof_label)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.min_label = QLabel(Form)
        self.min_label.setObjectName(u"min_label")

        self.horizontalLayout_2.addWidget(self.min_label)

        self.value_HSlider = QSlider(Form)
        self.value_HSlider.setObjectName(u"value_HSlider")
        self.value_HSlider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_2.addWidget(self.value_HSlider)

        self.max_label = QLabel(Form)
        self.max_label.setObjectName(u"max_label")

        self.horizontalLayout_2.addWidget(self.max_label)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.forward_button = QPushButton(Form)
        self.forward_button.setObjectName(u"forward_button")

        self.horizontalLayout_3.addWidget(self.forward_button)

        self.backward_button = QPushButton(Form)
        self.backward_button.setObjectName(u"backward_button")

        self.horizontalLayout_3.addWidget(self.backward_button)


        self.verticalLayout.addLayout(self.horizontalLayout_3)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText(QCoreApplication.translate("Form", u"DOF:", None))
        self.dof_label.setText(QCoreApplication.translate("Form", u"NaN", None))
        self.min_label.setText(QCoreApplication.translate("Form", u"min", None))
        self.max_label.setText(QCoreApplication.translate("Form", u"max", None))
        self.forward_button.setText(QCoreApplication.translate("Form", u"-", None))
        self.backward_button.setText(QCoreApplication.translate("Form", u"+", None))
    # retranslateUi

