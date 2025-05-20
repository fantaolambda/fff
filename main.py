import pickle

import torch
import torch.nn as nn
import sys

from PyQt5.QtGui import QFont, QDoubleValidator, QTextCharFormat, QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QLineEdit, QTextEdit, QMessageBox)


class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        # 调用父类的初始化方法
        super(DNN, self).__init__()
        self.x1 = 64
        self.x2 = 32
        self.x3 = 16

        # 定义第一个全连接层，将输入层连接到隐藏层
        self.fc1 = nn.Linear(input_size, self.x1)
        self.fc2 = nn.Linear(self.x1, self.x2)
        self.fc3 = nn.Linear(self.x2, self.x3)
        self.fc4 = nn.Linear(self.x3, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# 使用模型进行预测的函数
def predict(model, x, train_x_min, train_x_max):
    # 将输入数据转换为torch张量
    x_tensor = x.clone().detach().double()
    # 对输入数据进行归一化
    x_tensor, _, _ = min_max_normalize2(x_tensor, train_x_min, train_x_max)
    # 在不计算梯度的情况下进行预测
    with torch.no_grad():
        prediction = model(x_tensor)

    return prediction.numpy()


def min_max_normalize2(data, min_val, max_val):
    # 进行最小 - 最大归一化操作，将数据映射到[0, 1]区间
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = normalized_data.double()
    return normalized_data, min_val, max_val


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.validator = None
        self.input_box = ''
        self.result_text = None
        self.result_label = None
        self.input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.combo1 = None
        self.combo2 = None
        self.combo3 = None
        self.combo4 = None
        self.combo5 = None
        self.combo6 = None
        self.combo7 = None
        self.combo8 = None
        self.combo9 = None
        self.combo10 = None
        self.combo11 = None
        self.combo12 = None
        self.combo13 = None
        self.option1_label = None
        self.option2_label = None
        self.option3_label = None
        self.option4_label = None
        self.option5_label = None
        self.option6_label = None
        self.option7_label = None
        self.option8_label = None
        self.option9_label = None
        self.option10_label = None
        self.option11_label = None
        self.option12_label = None
        self.option13_label = None
        self.option1_layout = None
        self.option2_layout = None
        self.option3_layout = None
        self.option4_layout = None
        self.option5_layout = None
        self.option6_layout = None
        self.option7_layout = None
        self.option8_layout = None
        self.option9_layout = None
        self.option10_layout = None
        self.option11_layout = None
        self.option12_layout = None
        self.option13_layout = None
        self.initUI()

    def initUI(self):

        font = QFont("微软雅黑", 12, QFont.Bold)  # 字体格式

        self.setWindowTitle('高温翘曲数据预测界面')
        self.setGeometry(800, 800, 400, 800)  # 调整窗口高度以容纳结果区域

        # 创建主垂直布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 创建选项部分的水平布局
        options_layout = QHBoxLayout()

        # 创建13个下拉菜单和一个输入框
        self.combo1 = QComboBox()
        self.combo2 = QComboBox()
        self.combo3 = QComboBox()
        self.combo4 = QComboBox()
        self.combo5 = QComboBox()
        self.combo6 = QComboBox()
        self.combo7 = QComboBox()
        self.combo8 = QComboBox()
        self.combo9 = QComboBox()
        self.combo10 = QComboBox()
        self.combo11 = QComboBox()
        self.combo12 = QComboBox()
        self.combo13 = QComboBox()
        self.combo2.setStyleSheet("QComboBox { width: 100px; }")
        self.combo4.setStyleSheet("QComboBox { width: 120px; }")
        self.input_box = QLineEdit()
        self.validator = QDoubleValidator()
        self.input_box.setValidator(self.validator)

        # 设置初始选项
        self.combo1.addItems(['请选择产品编码', '03016FSY', '03016SYS', '03017edv', '03017EDW', '03017fkw', '03120779'])
        self.combo2.addItems(['请选择子项'])
        self.combo3.addItems(['请选择子项'])
        self.combo4.addItems(['请选择子项'])
        self.combo5.addItems(['请选择子项'])
        self.combo6.addItems(['请选择子项'])
        self.combo7.addItems(['请选择子项'])
        self.combo8.addItems(['请选择子项'])
        self.combo9.addItems(['请选择子项'])
        self.combo10.addItems(['请选择子项'])
        self.combo11.addItems(['请选择子项'])
        self.combo12.addItems(['请选择子项'])
        self.combo13.addItems(['请选择子项'])
        self.combo1.setFont(font)
        self.combo2.setFont(font)
        self.combo3.setFont(font)
        self.combo4.setFont(font)
        self.combo5.setFont(font)
        self.combo6.setFont(font)
        self.combo7.setFont(font)
        self.combo8.setFont(font)
        self.combo9.setFont(font)
        self.combo10.setFont(font)
        self.combo11.setFont(font)
        self.combo12.setFont(font)
        self.combo13.setFont(font)

        # 创建每个选项的垂直布局，并调整间距
        # 选项1
        self.option1_layout = QVBoxLayout()
        self.option1_label = QLabel('产品编码')
        self.option1_layout.addWidget(self.option1_label)
        self.option1_layout.addWidget(self.combo1)
        self.option1_layout.setSpacing(5)  # 调整标签和选项框之间的间距

        # 选项2
        self.option2_layout = QVBoxLayout()
        self.option2_label = QLabel('PCB厂家')
        self.option2_layout.addWidget(self.option2_label)
        self.option2_layout.addWidget(self.combo2)
        self.option2_layout.setSpacing(5)


        # 选项3
        self.option3_layout = QVBoxLayout()
        self.option3_label = QLabel('纯压/混压')
        self.option3_layout.addWidget(self.option3_label)
        self.option3_layout.addWidget(self.combo3)
        self.option3_layout.setSpacing(5)

        # 选项4
        self.option4_layout = QVBoxLayout()
        self.option4_label = QLabel('材料')
        self.option4_layout.addWidget(self.option4_label)
        self.option4_layout.addWidget(self.combo4)
        self.option4_layout.setSpacing(5)

        # 选项5
        self.option5_layout = QVBoxLayout()
        self.option5_label = QLabel('BGA长度mm')
        self.option5_layout.addWidget(self.option5_label)
        self.option5_layout.addWidget(self.combo5)
        self.option5_layout.setSpacing(5)

        # 选项6
        self.option6_layout = QVBoxLayout()
        self.option6_label = QLabel('BGA宽度mm')
        self.option6_layout.addWidget(self.option6_label)
        self.option6_layout.addWidget(self.combo6)
        self.option6_layout.setSpacing(5)

        # 选项7
        self.option7_layout = QVBoxLayout()
        self.option7_label = QLabel('BGA面积mm2')
        self.option7_layout.addWidget(self.option7_label)
        self.option7_layout.addWidget(self.combo7)
        self.option7_layout.setSpacing(5)

        # 选项8
        self.option8_layout = QVBoxLayout()
        self.option8_label = QLabel('总板厚mm')
        self.option8_layout.addWidget(self.option8_label)
        self.option8_layout.addWidget(self.combo8)
        self.option8_layout.setSpacing(5)

        # 选项9
        self.option9_layout = QVBoxLayout()
        self.option9_label = QLabel('叠层')
        self.option9_layout.addWidget(self.option9_label)
        self.option9_layout.addWidget(self.combo9)
        self.option9_layout.setSpacing(5)

        # 选项10
        self.option10_layout = QVBoxLayout()
        self.option10_label = QLabel('总层数')
        self.option10_layout.addWidget(self.option10_label)
        self.option10_layout.addWidget(self.combo10)
        self.option10_layout.setSpacing(5)

        # 选项11
        self.option11_layout = QVBoxLayout()
        self.option11_label = QLabel('面积mm2')
        self.option11_layout.addWidget(self.option11_label)
        self.option11_layout.addWidget(self.combo11)
        self.option11_layout.setSpacing(5)

        # 选项12
        self.option12_layout = QVBoxLayout()
        self.option12_label = QLabel('PCB长度mm')
        self.option12_layout.addWidget(self.option12_label)
        self.option12_layout.addWidget(self.combo12)
        self.option12_layout.setSpacing(5)

        # 选项13
        self.option13_layout = QVBoxLayout()
        self.option13_label = QLabel('PCB宽度mm')
        self.option13_layout.addWidget(self.option13_label)
        self.option13_layout.addWidget(self.combo13)
        self.option13_layout.setSpacing(5)

        # 输入框
        input_layout = QVBoxLayout()
        input_label = QLabel('常温数据:')
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_box)
        input_layout.setSpacing(5)

        self.option1_label.setFont(font)
        self.option2_label.setFont(font)
        self.option3_label.setFont(font)
        self.option4_label.setFont(font)
        self.option5_label.setFont(font)
        self.option6_label.setFont(font)
        self.option7_label.setFont(font)
        self.option8_label.setFont(font)
        self.option9_label.setFont(font)
        self.option10_label.setFont(font)
        self.option11_label.setFont(font)
        self.option12_label.setFont(font)
        self.option13_label.setFont(font)
        input_label.setFont(font)
        self.input_box.setFont(font)

        # 将所有垂直布局添加到选项部分的水平布局
        options_layout.addLayout(self.option1_layout)
        options_layout.addLayout(self.option2_layout)
        options_layout.addLayout(self.option3_layout)
        options_layout.addLayout(self.option4_layout)
        options_layout.addLayout(self.option5_layout)
        options_layout.addLayout(self.option6_layout)
        options_layout.addLayout(self.option7_layout)
        options_layout.addLayout(self.option8_layout)
        options_layout.addLayout(self.option9_layout)
        options_layout.addLayout(self.option10_layout)
        options_layout.addLayout(self.option11_layout)
        options_layout.addLayout(self.option12_layout)
        options_layout.addLayout(self.option13_layout)
        options_layout.addLayout(input_layout)

        # 创建预测按钮
        button = QPushButton('预测')
        button.clicked.connect(self.handleSubmit)
        options_layout.addWidget(button)

        # 创建结果显示区域
        out_layout = QVBoxLayout()
        self.result_label = QLabel('\n预测数据记录:')
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(800)  # 设置结果区域的高度

        text_format = QTextCharFormat()
        text_format.setForeground(QColor("red"))  # 设置文本颜色为红色
        text_format.setFont(font)

        cursor = self.result_text.textCursor()
        cursor.insertText(
            "当前版本1.4\n\n训练样本数：520        测试样本数：130\n\n"
            "优化：优化了界面显示，增加文本提示信息，温度与数值更加对齐\n\n"
            "基于提供数据， 本模型最大差值为±50左右\n\n"
            "当前版本预测最高温数据准确率在85%的置信度下会>=70%\n\n"
            "注意：本模型基于历史数据进行训练，且warpage的测量本身存在波动，预测会存在一定误差，预测值仅供大家参考！！！\n",
            text_format)

        out_layout.addWidget(self.result_label)
        out_layout.addWidget(self.result_text)
        out_layout.setSpacing(5)

        button.setFont(font)
        self.result_label.setFont(font)

        # 将选项部分和结果区域添加到主布局
        main_layout.addLayout(options_layout)
        main_layout.addLayout(out_layout)

        # 连接信号和槽
        self.combo1.currentIndexChanged.connect(self.updateCombo2)
        self.combo2.currentIndexChanged.connect(self.updateCombo3)
        self.combo3.currentIndexChanged.connect(self.updateCombo4)
        self.combo4.currentIndexChanged.connect(self.updateCombo5)
        self.combo5.currentIndexChanged.connect(self.updateCombo6)
        self.combo6.currentIndexChanged.connect(self.updateCombo7)
        self.combo7.currentIndexChanged.connect(self.updateCombo8)
        self.combo8.currentIndexChanged.connect(self.updateCombo9)
        self.combo9.currentIndexChanged.connect(self.updateCombo10)
        self.combo10.currentIndexChanged.connect(self.updateCombo11)
        self.combo11.currentIndexChanged.connect(self.updateCombo12)
        self.combo12.currentIndexChanged.connect(self.updateCombo13)

        # 窗口整体样式
        self.setStyleSheet("""
                                QMainWindow {
                                    background-color: #f0f0f0;
                                }
                                QPushButton {
                                    background-color: #007acc;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    margin: 10px;
                                    border-radius: 5px;
                                }
                                QPushButton:hover {
                                    background-color: #005ca3;
                                }
                                QPushButton:pressed {
                                    background-color: #004581;
                                }
                            """)

    def updateCombo2(self):
        mainCategory = self.combo1.currentText()
        self.combo2.clear()
        if mainCategory == '03016FSY':
            self.input[0] = 1
            self.combo2.addItems(['生益电子', '方正F6', '沪士-昆山C3'])
        elif mainCategory == '03016SYS':
            self.input[0] = 2
            self.combo2.addItems(['沪士-昆山C3', '生益电子'])
        elif mainCategory == '03017edv':
            self.input[0] = 5
            self.combo2.addItems(['沪士-昆山C3'])
        elif mainCategory == '03017EDW':
            self.input[0] = 6
            self.combo2.addItems(['方正F6'])
        elif mainCategory == '03017fkw':
            self.input[0] = 7
            self.combo2.addItems(['沪士-昆山C3'])
        elif mainCategory == '03120779':
            self.input[0] = 10
            self.combo2.addItems(['方正F3'])
        else:
            self.combo2.addItems(['请选择子项'])
        self.updateCombo3()
        self.updateCombo4()
        self.updateCombo5()
        self.updateCombo6()
        self.updateCombo7()
        self.updateCombo8()
        self.updateCombo9()
        self.updateCombo10()
        self.updateCombo11()
        self.updateCombo12()
        self.updateCombo13()

    def updateCombo3(self):
        sub1 = self.combo1.currentText()
        sub2 = self.combo2.currentText()
        self.combo3.clear()
        if sub2 == '生益电子':
            self.input[1] = 1
        elif sub2 == '方正F6':
            self.input[1] = 2
        elif sub2 == '沪士-昆山C3':
            self.input[1] = 3
        elif sub2 == '方正F3':
            self.input[1] = 5
        if sub1 == '03016FSY':
            self.combo3.addItems(['无'])
            self.input[2] = 1
        elif sub1 == '03016SYS':
            self.combo3.addItems(['纯压'])
            self.input[2] = 2
        elif sub1 == '03017edv':
            self.combo3.addItems(['纯压'])
            self.input[2] = 2
        elif sub1 == '03017EDW':
            self.combo3.addItems(['纯压'])
            self.input[2] = 2
        elif sub1 == '03017fkw':
            self.combo3.addItems(['混压'])
            self.input[2] = 3
        elif sub1 == '03120779':
            self.combo3.addItems(['混压'])
            self.input[2] = 3
        else:
            self.combo3.addItems(['请选择子项'])
        self.updateCombo4()
        self.updateCombo5()
        self.updateCombo6()
        self.updateCombo7()
        self.updateCombo8()
        self.updateCombo9()
        self.updateCombo10()
        self.updateCombo11()
        self.updateCombo12()
        self.updateCombo13()

    def updateCombo4(self):
        sub1 = self.combo1.currentText()
        sub2 = self.combo2.currentText()
        self.combo4.clear()
        if sub1 == '03016FSY':
            self.combo4.addItems(['无'])
            self.combo4.setCurrentIndex(0)
            self.input[3] = 1
        elif sub1 == '03016SYS':
            if sub2 == '沪士-昆山C3':
                self.combo4.addItems(['TU-933E'])
                self.combo4.setCurrentIndex(0)
                self.input[3] = 2
            if sub2 == '生益电子':
                self.combo4.addItems(['Megtron7 GE'])
                self.combo4.setCurrentIndex(0)
                self.input[3] = 3
        elif sub1 == '03017edv':
            self.combo4.addItems(['TU-933V'])
            self.combo4.setCurrentIndex(0)
            self.input[3] = 7
        elif sub1 == '03017EDW':
            self.combo4.addItems(['TU-862HF'])
            self.combo4.setCurrentIndex(0)
            self.input[3] = 8
        elif sub1 == '03017fkw':
            self.combo4.addItems(['S6(H3X)'])
            self.combo4.setCurrentIndex(0)
            self.input[3] = 9
        elif sub1 == '03120779':
            self.combo4.addItems(['NY-P4P'])
            self.combo4.setCurrentIndex(0)
            self.input[3] = 11
        else:
            self.combo4.addItems(['请选择子项'])

    def updateCombo5(self):
        sub1 = self.combo1.currentText()
        self.combo5.clear()
        if sub1 == '03016FSY':
            self.combo5.addItems(['无'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 1
        elif sub1 == '03016SYS':
            self.combo5.addItems(['77.5'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 2
        elif sub1 == '03017edv':
            self.combo5.addItems(['65'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 3
        elif sub1 == '03017EDW':
            self.combo5.addItems(['无'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 1
        elif sub1 == '03017fkw':
            self.combo5.addItems(['无'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 1
        elif sub1 == '03120779':
            self.combo5.addItems(['无'])
            self.combo5.setCurrentIndex(0)
            self.input[4] = 1
        else:
            self.combo5.addItems(['请选择子项'])

    def updateCombo6(self):
        sub1 = self.combo1.currentText()
        self.combo6.clear()
        if sub1 == '03016FSY':
            self.combo6.addItems(['无'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 1
        elif sub1 == '03016SYS':
            self.combo6.addItems(['69.5'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 2
        elif sub1 == '03017edv':
            self.combo6.addItems(['55'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 3
        elif sub1 == '03017EDW':
            self.combo6.addItems(['无'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 1
        elif sub1 == '03017fkw':
            self.combo6.addItems(['无'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 1
        elif sub1 == '03120779':
            self.combo6.addItems(['无'])
            self.combo6.setCurrentIndex(0)
            self.input[5] = 1
        else:
            self.combo6.addItems(['请选择子项'])

    def updateCombo7(self):
        sub1 = self.combo1.currentText()
        self.combo7.clear()
        if sub1 == '03016FSY':
            self.combo7.addItems(['无'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 1
        elif sub1 == '03016SYS':
            self.combo7.addItems(['5386.25'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 2
        elif sub1 == '03017edv':
            self.combo7.addItems(['3575'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 3
        elif sub1 == '03017EDW':
            self.combo7.addItems(['无'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 1
        elif sub1 == '03017fkw':
            self.combo7.addItems(['无'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 1
        elif sub1 == '03120779':
            self.combo7.addItems(['无'])
            self.combo7.setCurrentIndex(0)
            self.input[6] = 1
        else:
            self.combo7.addItems(['请选择子项'])

    def updateCombo8(self):
        sub1 = self.combo1.currentText()
        self.combo8.clear()
        if sub1 == '03016FSY':
            self.combo8.addItems(['无'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 1
        elif sub1 == '03016SYS':
            self.combo8.addItems(['4'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 2
        elif sub1 == '03017edv':
            self.combo8.addItems(['4'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 2
        elif sub1 == '03017EDW':
            self.combo8.addItems(['4'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 2
        elif sub1 == '03017fkw':
            self.combo8.addItems(['4'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 2
        elif sub1 == '03120779':
            self.combo8.addItems(['2.5'])
            self.combo8.setCurrentIndex(0)
            self.input[7] = 3
        else:
            self.combo8.addItems(['请选择子项'])

    def updateCombo9(self):
        sub1 = self.combo1.currentText()
        self.combo9.clear()
        if sub1 == '03016FSY':
            self.combo9.addItems(['无'])
            self.combo9.setCurrentIndex(0)
            self.input[8] = 1
        elif sub1 == '03016SYS':
            self.combo9.addItems(['14+14'])
            self.combo9.setCurrentIndex(0)
            self.input[8] = 2
        elif sub1 == '03017edv':
            self.combo9.addItems(['22'])
            self.combo8.setCurrentIndex(0)
            self.input[8] = 5
        elif sub1 == '03017EDW':
            self.combo9.addItems(['28'])
            self.combo9.setCurrentIndex(0)
            self.input[8] = 6
        elif sub1 == '03017fkw':
            self.combo9.addItems(['16+12'])
            self.combo9.setCurrentIndex(0)
            self.input[8] = 7
        elif sub1 == '03120779':
            self.combo9.addItems(['3+2+3'])
            self.combo9.setCurrentIndex(0)
            self.input[8] = 9
        else:
            self.combo9.addItems(['请选择子项'])

    def updateCombo10(self):
        sub1 = self.combo1.currentText()
        self.combo10.clear()
        if sub1 == '03016FSY':
            self.combo10.addItems(['无'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 1
        elif sub1 == '03016SYS':
            self.combo10.addItems(['28'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 2
        elif sub1 == '03017edv':
            self.combo10.addItems(['22'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 4
        elif sub1 == '03017EDW':
            self.combo10.addItems(['28'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 2
        elif sub1 == '03017fkw':
            self.combo10.addItems(['28'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 2
        elif sub1 == '03120779':
            self.combo10.addItems(['22'])
            self.combo10.setCurrentIndex(0)
            self.input[9] = 4
        else:
            self.combo10.addItems(['请选择子项'])

    def updateCombo11(self):
        sub1 = self.combo1.currentText()
        self.combo11.clear()
        if sub1 == '03016FSY':
            self.combo11.addItems(['无'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 1
        elif sub1 == '03016SYS':
            self.combo11.addItems(['25200'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 2
        elif sub1 == '03017edv':
            self.combo11.addItems(['226590.94'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 5
        elif sub1 == '03017EDW':
            self.combo11.addItems(['230656'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 4
        elif sub1 == '03017fkw':
            self.combo11.addItems(['226632'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 6
        elif sub1 == '03120779':
            self.combo11.addItems(['22400'])
            self.combo11.setCurrentIndex(0)
            self.input[10] = 8
        else:
            self.combo11.addItems(['请选择子项'])

    def updateCombo12(self):
        sub1 = self.combo1.currentText()
        self.combo12.clear()
        if sub1 == '03016FSY':
            self.combo12.addItems(['无'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 1
        elif sub1 == '03016SYS':
            self.combo12.addItems(['140'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 2
        elif sub1 == '03017edv':
            self.combo12.addItems(['515.8'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 5
        elif sub1 == '03017EDW':
            self.combo12.addItems(['424'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 4
        elif sub1 == '03017fkw':
            self.combo12.addItems(['426'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 3
        elif sub1 == '03120779':
            self.combo12.addItems(['160'])
            self.combo12.setCurrentIndex(0)
            self.input[11] = 6
        else:
            self.combo12.addItems(['请选择子项'])

    def updateCombo13(self):
        sub1 = self.combo1.currentText()
        self.combo13.clear()
        if sub1 == '03016FSY':
            self.combo13.addItems(['无'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 1
        elif sub1 == '03016SYS':
            self.combo13.addItems(['180'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 2
        elif sub1 == '03017edv':
            self.combo13.addItems(['439.3'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 5
        elif sub1 == '03017EDW':
            self.combo13.addItems(['544'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 4
        elif sub1 == '03017fkw':
            self.combo13.addItems(['532'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 6
        elif sub1 == '03120779':
            self.combo13.addItems(['140'])
            self.combo13.setCurrentIndex(0)
            self.input[12] = 7
        else:
            self.combo13.addItems(['请选择子项'])

    def handleSubmit(self):
        font = QFont("微软雅黑", 12, QFont.Bold)  # 字体格式

        sub1 = self.combo1.currentText()
        if sub1 == '请选择产品编码':
            msg = QMessageBox()
            # 设置对话框的图标为警告图标
            msg.setIcon(QMessageBox.Warning)
            # 设置对话框的标题
            msg.setWindowTitle("警告")
            # 设置对话框的内容文本
            msg.setText("未选择编码信息， 请选择编码信息！")
            # 添加一个按钮，用户可以点击该按钮关闭对话框
            msg.setStandardButtons(QMessageBox.Ok)
            # 显示对话框，并等待用户响应
            msg.exec_()
            return

        initt = self.input_box.text()

        if initt == '':
            msg = QMessageBox()
            # 设置对话框的图标为警告图标
            msg.setIcon(QMessageBox.Warning)
            # 设置对话框的标题
            msg.setWindowTitle("警告")
            # 设置对话框的内容文本
            msg.setText("初始温度未输出，请输入！")
            # 添加一个按钮，用户可以点击该按钮关闭对话框
            msg.setStandardButtons(QMessageBox.Ok)
            # 显示对话框，并等待用户响应
            msg.exec_()
            return
        else:
            self.input[13] = float(initt)
            initt = str(self.input[13])
            initt = initt

        current_text1 = self.combo1.currentText()
        current_text2 = self.combo2.currentText()
        current_text3 = self.combo3.currentText()
        current_text4 = self.combo4.currentText()
        current_text5 = self.combo5.currentText()
        current_text6 = self.combo6.currentText()
        current_text7 = self.combo7.currentText()
        current_text8 = self.combo8.currentText()
        current_text9 = self.combo9.currentText()
        current_text10 = self.combo10.currentText()
        current_text11 = self.combo11.currentText()
        current_text12 = self.combo12.currentText()
        current_text13 = self.combo13.currentText()

        tip_str = "&nbsp;&nbsp;&nbsp;&nbsp;".join(['编码：' + current_text1, '厂家：' + current_text2,
                                                   '压缩：' + current_text3, '材料：' + current_text4,
                                                   'BGA长度：' + current_text5, 'BGA宽度：' + current_text6,
                                                   'BGA面积：' + current_text7, '总板厚：' + current_text8,
                                                   '叠层：' + current_text9, '总层数：' + current_text10,
                                                   '面积：' + current_text11, 'PCB长度：' + current_text12,
                                                   'PCB宽度：' + current_text13, '常温数据：' + initt])

        # 生成数据
        temperature_labels = self.generateRandomData()

        with open('arrays.pkl', 'rb') as file:
            x_min, x_max, x_cols, y_cols = pickle.load(file)

        # 创建DNN模型实例
        model = DNN(x_cols, y_cols)
        model.double()
        # 加载保存的模型参数
        model.load_state_dict(torch.load('dnn_model.pth'))
        # 将模型设置为评估模式
        model.eval()

        data = self.input

        input_data = torch.tensor(data, dtype=torch.float64).unsqueeze(0)

        with torch.no_grad():
            # 使用模型进行预测，并反归一化预测结果
            prediction = predict(model, input_data, x_min, x_max)
            prediction = prediction[0].tolist()
            prediction = [format(num, '.2f') for num in prediction]
            prediction = list(map(str, prediction))

        for i, x in enumerate(prediction):
            if len(x) < 7:
                prediction[i] = prediction[i] + '0'*(7 - len(x))

        for i, x in enumerate(temperature_labels):
            if len(x) < 10:
                temperature_labels[i] = temperature_labels[i] + '&nbsp;'*(10 - len(x))

        self.result_text.append("<span style='color: red;'>预测情况：</span>")
        self.result_text.append(f"<span style='color: green;'>预测输入：{tip_str}</span>")

        # 将所有温度标签连接成一行
        temperature_labels_str = "&nbsp;&nbsp;&nbsp;&nbsp;".join(temperature_labels)
        self.result_text.append(f"<span style='color: orange;'>温度标签：\n{temperature_labels_str}</span>")

        prediction_num = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(
            x for i, x in enumerate(prediction))
        self.result_text.append(f"<span style='color: blue;'> 预测输出：\n{prediction_num}</span>")
        self.result_text.append('')
        self.result_text.setFont(font)

    def generateRandomData(self):
        # 定义温度前缀
        temperatures = [' 60', ' 90', '125', '150', '180', '200', '220', '235', '245', '235', '220', '200', '180',
                        '150', '125', '90', '60']
        temperature_labels = []

        for temp in temperatures:
            temperature_labels.append(f"{temp}℃")

        return temperature_labels


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
