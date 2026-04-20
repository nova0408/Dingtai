from abc import ABCMeta, abstractmethod

import numpy as np
from PySide6.QtCore import QAbstractItemModel, QMimeData, QModelIndex, QObject, Qt
from PySide6.QtGui import QAction, QContextMenuEvent
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QTreeView, QWidget

from .casia_qss import CasiaQss

# --- 1. CasiaQss 样式封装 ---
# --- 2. 元类冲突解决 ---
QObjectMeta = type(QObject)


class CombinedMeta(QObjectMeta, ABCMeta):
    pass


class INodeInfoDisplay(metaclass=CombinedMeta):
    @abstractmethod
    def show_node_info(self, node):
        pass

    @abstractmethod
    def get_widget(self) -> QWidget:
        pass


# --- 3. 核心数据结构 ---
class PointCloudNode:
    def __init__(self, display_name: str, index: int, transform: np.ndarray, is_branch: bool = False):
        self.display_name = display_name
        self.index = index
        self.transform = transform
        self.is_branch = is_branch
        self.parent: PointCloudNode | None = None
        self.children = []

    def add_child(self, child, index=-1):
        child.parent = self
        if index == -1 or index >= len(self.children):
            self.children.append(child)
        else:
            self.children.insert(index, child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False

    def row(self):
        if self.parent:
            return self.parent.children.index(self)
        return 0


# --- 4. 树模型 (逻辑统一版) ---
class PointCloudTreeModel(QAbstractItemModel):
    def __init__(self):
        super().__init__(parent=None)
        self._root_node = PointCloudNode("InvisibleRoot", -1, np.eye(4))

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return "标记树"
        return None

    def index(self, row, column, parent=QModelIndex()):
        p_node = self.get_node(parent)
        if 0 <= row < len(p_node.children):
            return self.createIndex(row, column, p_node.children[row])
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if node.parent == self._root_node or node.parent is None:
            return QModelIndex()
        return self.createIndex(node.parent.row(), 0, node.parent)

    def rowCount(self, parent=QModelIndex()):
        return len(self.get_node(parent).children)

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node: PointCloudNode = index.internalPointer()
        if role == Qt.ItemDataRole.DisplayRole:
            return node.display_name
        if role == Qt.ItemDataRole.UserRole:
            return node
        return None

    def setData(self, index: QModelIndex, value, /, role=Qt.ItemDataRole.EditRole) -> bool:
        return super().setData(index, value, role)

    def get_node(self, index: QModelIndex) -> PointCloudNode:
        if index.isValid():
            return index.internalPointer()
        return self._root_node

    def index_of_node(self, node):
        if node == self._root_node or node.parent is None:
            return QModelIndex()
        return self.createIndex(node.parent.children.index(node), 0, node)

    # --- 核心：统一的逻辑操作方法 ---

    def create_node_at_index(
        self, name, id: int, transform, parent_idx: QModelIndex | None = None, is_branch=False, row=-1
    ):
        """基础方法：在指定 parent 的指定 row 插入"""
        if not parent_idx:
            parent_idx = QModelIndex()
        p_node = self.get_node(parent_idx)
        self.beginResetModel()
        new_node = PointCloudNode(name, id, transform, is_branch=is_branch)
        p_node.add_child(new_node, row)
        self._reorder_branches(p_node)
        self.endResetModel()
        return self.index_of_node(new_node)

    def add_child(self, target_idx: QModelIndex, name: str):
        """
        统一逻辑：向 target 节点添加子内容。
        如果 target 是普通节点，则自动存入其 (分支) 容器。
        """
        target_node = self.get_node(target_idx)

        # 1. 确定要把节点放进哪个容器
        container_node = self._get_or_create_branch_container(target_node)

        # 2. 调用基础插入方法
        container_idx = self.index_of_node(container_node)
        return self.create_node_at_index(container_idx, name)

    def _get_or_create_branch_container(self, node):
        """内部工具：寻找或创建分支节点"""
        if node.is_branch:
            return node
        if node == self._root_node:
            return node  # 根层级直接返回自己

        branch_name = f"{node.display_name}(分支)"
        p_node = node.parent
        container = next((n for n in p_node.children if n.display_name == branch_name), None)
        if not container:
            container = PointCloudNode(branch_name, -1, np.eye(4), is_branch=True)
            p_node.add_child(container)
        return container

    def _reorder_branches(self, parent_node):
        if parent_node:
            parent_node.children.sort(key=lambda x: 1 if x.is_branch else 0)

    def _cleanup_empty_branches(self, node):
        if not node or node == self._root_node:
            return
        if node.is_branch and not node.children:
            p = node.parent
            if p:
                p.remove_child(node)
                self._cleanup_empty_branches(p)

    # --- 拖拽重写 ---
    def flags(self, index):
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not index.isValid():
            return f | Qt.ItemFlag.ItemIsDropEnabled
        return f | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return ["application/x-transform-node"]

    def mimeData(self, indexes):
        mime_data = QMimeData()
        self._dragged_node = indexes[0].internalPointer()
        mime_data.setData("application/x-transform-node", b"data")
        return mime_data

    def dropMimeData(self, data, action, row, column, parent_index):
        if not data.hasFormat("application/x-transform-node"):
            return False
        source_node = self._dragged_node
        old_parent = source_node.parent
        target_node = self.get_node(parent_index)

        self.beginResetModel()
        # 提取逻辑：如果是分支容器，取子项，删容器
        nodes_to_move = source_node.children[:] if source_node.is_branch else [source_node]
        if old_parent:
            old_parent.remove_child(source_node)

        if row == -1 and parent_index.isValid():
            # 拖到节点上 -> 走统一的分支逻辑
            container = self._get_or_create_branch_container(target_node)
            for n in nodes_to_move:
                container.add_child(n)
        else:
            # 拖到缝隙 -> 走链式排序逻辑
            insert_row = row if row != -1 else len(target_node.children)
            for n in reversed(nodes_to_move):
                target_node.add_child(n, insert_row)

        self._cleanup_empty_branches(old_parent)
        self._reorder_branches(target_node)
        self.endResetModel()
        return True

    # --- 业务方法 ---
    def rename_node(self, index, new_name):
        if not index.isValid():
            return
        node = self.get_node(index)
        old_branch_name = f"{node.display_name}(分支)"
        node.display_name = new_name
        # 同步重命名分支容器
        p_node = node.parent if node.parent else self._root_node
        for child in p_node.children:
            if child.is_branch and child.display_name == old_branch_name:
                child.display_name = f"{new_name}(分支)"
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])

    def remove_node(self, index):
        if not index.isValid():
            return
        node = self.get_node(index)
        p_node = node.parent
        self.beginResetModel()
        if p_node:
            p_node.remove_child(node)
        self._cleanup_empty_branches(p_node)
        self.endResetModel()

    def promote_children(self, index):
        node = self.get_node(index)
        if not node.is_branch:
            return
        self.beginResetModel()
        p_node = node.parent
        for child in node.children[:]:
            node.remove_child(child)
            p_node.add_child(child)
        p_node.remove_child(node)
        self._reorder_branches(p_node)
        self.endResetModel()


# --- 5. 自定义 View ---
class PointCloudTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeView.DragDropMode.InternalMove)
        self.setIndentation(15)
        self.setHeaderHidden(True)  # 隐藏标题
        self.setStyleSheet(CasiaQss.TREE_VIEW)

    def contextMenuEvent(self, event: QContextMenuEvent):
        index = self.indexAt(event.pos())
        model = self.model()
        menu = QMenu(self)
        style = QApplication.style()

        if index.isValid():
            act_rename = QAction(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), "重命名", self)
            act_rename.triggered.connect(lambda: self.window().rename_node_dialog(index))
            act_fol = QAction(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder), "添加同级后续", self)
            act_fol.triggered.connect(lambda: self.window().add_node_dialog(model.parent(index)))
            act_chi = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), "添加侧向分支", self)
            act_chi.triggered.connect(lambda: self.window().handle_branch_logic(index))
            act_del = QAction(style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon), "删除", self)
            act_del.triggered.connect(lambda: model.remove_node(index))
            menu.addActions([act_rename, act_fol, act_chi, act_del])
        else:
            act_new = QAction(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon), "添加根链节点", self)
            act_new.triggered.connect(lambda: self.window().add_node_dialog(QModelIndex()))
            menu.addAction(act_new)
        menu.exec(event.globalPos())


# class MainWindow(QMainWindow):
#     def __init__(self, info_panel: INodeInfoDisplay):
#         super().__init__()
#         self.setWindowTitle("标记树编辑器")
#         self.resize(1000, 600)
#         self.info_panel = info_panel
#         splitter = QSplitter(Qt.Orientation.Horizontal)
#         self.setCentralWidget(splitter)
#         self.tree_view = TransformTreeView()
#         self.model = TransformTreeModel()
#         self.tree_view.setModel(self.model)
#         splitter.addWidget(self.tree_view)
#         splitter.addWidget(self.info_panel.get_widget())
#         self.tree_view.clicked.connect(lambda idx: self.info_panel.show_node_info(self.model.get_node(idx)))

#     def rename_node_dialog(self, index):
#         node = self.model.get_node(index)
#         name, ok = QInputDialog.getText(self, "重命名", "名称：", text=node.display_name)
#         if ok and name:
#             self.model.rename_node(index, name)

#     def add_node_dialog(self, parent_idx):
#         name, ok = QInputDialog.getText(self, "添加", "名称：")
#         if ok and name:
#             self.model.create_node_at_index(parent_idx, name)
#             self.tree_view.expandAll()

#     def handle_branch_logic(self, index):
#         name, ok = QInputDialog.getText(self, "分支", "侧向子节点名称：")
#         if ok and name:
#             self.model.add_child(index, name)
#             self.tree_view.expandAll()
