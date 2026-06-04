# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'TestRobotTab.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize)
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1233, 934)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_4 = QGroupBox(Form)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setVisible(False)
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_4 = QLabel(self.groupBox_4)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_5.addWidget(self.label_4)

        self.base_enable = QWidget(self.groupBox_4)
        self.base_enable.setObjectName(u"base_enable")
        self.base_enable.setMinimumSize(QSize(44, 44))
        self.base_enable.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.horizontalLayout_5.addWidget(self.base_enable)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_5)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.dof_base_x = QWidget(self.groupBox_4)
        self.dof_base_x.setObjectName(u"dof_base_x")
        self.dof_base_x.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout_4.addWidget(self.dof_base_x, 0, 0, 1, 1)

        self.dof_base_y = QWidget(self.groupBox_4)
        self.dof_base_y.setObjectName(u"dof_base_y")
        self.dof_base_y.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout_4.addWidget(self.dof_base_y, 0, 1, 1, 1)

        self.dof_base_yaw = QWidget(self.groupBox_4)
        self.dof_base_yaw.setObjectName(u"dof_base_yaw")
        self.dof_base_yaw.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout_4.addWidget(self.dof_base_yaw, 0, 2, 1, 1)


        self.verticalLayout_5.addLayout(self.gridLayout_4)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)

        self.body_enable = QWidget(self.groupBox)
        self.body_enable.setObjectName(u"body_enable")
        self.body_enable.setMinimumSize(QSize(44, 44))
        self.body_enable.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.horizontalLayout_4.addWidget(self.body_enable)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.dof_body_ry = QWidget(self.groupBox)
        self.dof_body_ry.setObjectName(u"dof_body_ry")
        self.dof_body_ry.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout.addWidget(self.dof_body_ry, 0, 1, 1, 1)

        self.dof_body_z = QWidget(self.groupBox)
        self.dof_body_z.setObjectName(u"dof_body_z")
        self.dof_body_z.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout.addWidget(self.dof_body_z, 0, 0, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout)


        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_head = QGroupBox(Form)
        self.groupBox_head.setObjectName(u"groupBox_head")
        self.verticalLayout_head = QVBoxLayout(self.groupBox_head)
        self.verticalLayout_head.setObjectName(u"verticalLayout_head")
        self.horizontalLayout_head = QHBoxLayout()
        self.horizontalLayout_head.setObjectName(u"horizontalLayout_head")
        self.label_head = QLabel(self.groupBox_head)
        self.label_head.setObjectName(u"label_head")

        self.horizontalLayout_head.addWidget(self.label_head)

        self.head_enable = QWidget(self.groupBox_head)
        self.head_enable.setObjectName(u"head_enable")
        self.head_enable.setMinimumSize(QSize(44, 44))
        self.head_enable.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.horizontalLayout_head.addWidget(self.head_enable)

        self.horizontalSpacer_head = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_head.addItem(self.horizontalSpacer_head)

        self.verticalLayout_head.addLayout(self.horizontalLayout_head)

        self.gridLayout_head = QGridLayout()
        self.gridLayout_head.setObjectName(u"gridLayout_head")
        self.dof_head_yaw = QWidget(self.groupBox_head)
        self.dof_head_yaw.setObjectName(u"dof_head_yaw")
        self.dof_head_yaw.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(255, 55, 168);\n"
"}")

        self.gridLayout_head.addWidget(self.dof_head_yaw, 0, 0, 1, 1)

        self.verticalLayout_head.addLayout(self.gridLayout_head)

        self.verticalLayout.addWidget(self.groupBox_head)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.left_enable = QWidget(self.groupBox_2)
        self.left_enable.setObjectName(u"left_enable")
        self.left_enable.setMinimumSize(QSize(44, 44))
        self.left_enable.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.horizontalLayout_2.addWidget(self.left_enable)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.dof_l_1 = QWidget(self.groupBox_2)
        self.dof_l_1.setObjectName(u"dof_l_1")
        self.dof_l_1.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_1, 0, 0, 1, 1)

        self.dof_l_2 = QWidget(self.groupBox_2)
        self.dof_l_2.setObjectName(u"dof_l_2")
        self.dof_l_2.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_2, 0, 1, 1, 1)

        self.dof_l_3 = QWidget(self.groupBox_2)
        self.dof_l_3.setObjectName(u"dof_l_3")
        self.dof_l_3.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_3, 0, 2, 1, 1)

        self.dof_l_4 = QWidget(self.groupBox_2)
        self.dof_l_4.setObjectName(u"dof_l_4")
        self.dof_l_4.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_4, 1, 0, 1, 1)

        self.dof_l_5 = QWidget(self.groupBox_2)
        self.dof_l_5.setObjectName(u"dof_l_5")
        self.dof_l_5.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_5, 1, 1, 1, 1)

        self.dof_l_6 = QWidget(self.groupBox_2)
        self.dof_l_6.setObjectName(u"dof_l_6")
        self.dof_l_6.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_2.addWidget(self.dof_l_6, 1, 2, 1, 1)


        self.verticalLayout_3.addLayout(self.gridLayout_2)


        self.horizontalLayout.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(Form)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.right_enable = QWidget(self.groupBox_3)
        self.right_enable.setObjectName(u"right_enable")
        self.right_enable.setMinimumSize(QSize(44, 44))
        self.right_enable.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.horizontalLayout_3.addWidget(self.right_enable)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)


        self.verticalLayout_4.addLayout(self.horizontalLayout_3)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.dof_r_1 = QWidget(self.groupBox_3)
        self.dof_r_1.setObjectName(u"dof_r_1")
        self.dof_r_1.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_1, 0, 0, 1, 1)

        self.dof_r_2 = QWidget(self.groupBox_3)
        self.dof_r_2.setObjectName(u"dof_r_2")
        self.dof_r_2.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_2, 0, 1, 1, 1)

        self.dof_r_3 = QWidget(self.groupBox_3)
        self.dof_r_3.setObjectName(u"dof_r_3")
        self.dof_r_3.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_3, 0, 2, 1, 1)

        self.dof_r_4 = QWidget(self.groupBox_3)
        self.dof_r_4.setObjectName(u"dof_r_4")
        self.dof_r_4.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_4, 1, 0, 1, 1)

        self.dof_r_5 = QWidget(self.groupBox_3)
        self.dof_r_5.setObjectName(u"dof_r_5")
        self.dof_r_5.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_5, 1, 1, 1, 1)

        self.dof_r_6 = QWidget(self.groupBox_3)
        self.dof_r_6.setObjectName(u"dof_r_6")
        self.dof_r_6.setStyleSheet(u"[_q_custom_style_disabled=\"true\"] QObject {\n"
"	background-color: rgb(85, 170, 255)\n"
"}")

        self.gridLayout_3.addWidget(self.dof_r_6, 1, 2, 1, 1)


        self.verticalLayout_4.addLayout(self.gridLayout_3)


        self.horizontalLayout.addWidget(self.groupBox_3)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Form", u"AGV", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u4f7f\u80fd:", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"body", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u4f7f\u80fd:", None))
        self.groupBox_head.setTitle(QCoreApplication.translate("Form", u"head", None))
        self.label_head.setText(QCoreApplication.translate("Form", u"\u4f7f\u80fd:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"left arm", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u4f7f\u80fd:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"right arm", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u4f7f\u80fd:", None))
    # retranslateUi

