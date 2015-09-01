from PySide import QtCore, QtGui, QtSvg

import xml.etree.ElementTree as ET
import pywr.core

def load_model(filename=None, data=None):
    '''Load a test model and check it'''
    if data is None:
        with open(filename, 'r') as f:
            data = f.read()
    xml = ET.fromstring(data)
    model = pywr.core.Model.from_xml(xml)
    model.check()
    return model

class MainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.resize(QtCore.QSize(600, 400))

        self.setWindowTitle('Pywr')

        action_open = QtGui.QAction('&Open...', self)
        action_open.triggered.connect(self.dialog_open)
        action_export_svg = QtGui.QAction('Export to SVG...', self)
        action_export_svg.triggered.connect(self.export_svg)
        action_export_png = QtGui.QAction('Export to PNG...', self)
        action_export_png.triggered.connect(self.export_png)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(action_open)
        file_menu.addAction(action_export_svg)
        file_menu.addAction(action_export_png)

        self.schematic = Schematic()
        self.setCentralWidget(self.schematic)

        self.show()
        self.raise_()

    def dialog_open(self):
        filename, filter_text = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.', '*.xml')
        if filename:
            self.schematic.load_model(filename)

    def export_svg(self):
        filename, selected_filter = QtGui.QFileDialog.getSaveFileName(self, 'Save SVG', '.', '*.svg')
        if filename:
            self.schematic.export_svg(path=filename)

    def export_png(self):
        filename, selected_filter = QtGui.QFileDialog.getSaveFileName(self, 'Save PNG', '.', '*.png')
        if filename:
            self.schematic.export_png(path=filename)


class Schematic(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super(Schematic, self).__init__(*args, **kwargs)

        self.scene = QtGui.QGraphicsScene(self)

        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.view.scale(1, 1)

        # placeholder text
        text = QtGui.QGraphicsTextItem()
        text.setPlainText('To open a model: File > Open...')
        text.setDefaultTextColor(QtGui.QColor('#aaa'))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        font.setFamily('Helvetica')
        font.setPointSize(12)
        text.setFont(font)
        text.setPos(0,0)
        self.scene.addItem(text)

        # zoom buttons
        button_zoom_in = QtGui.QPushButton('+')
        button_zoom_in.clicked.connect(self.zoom_in)
        button_zoom_out = QtGui.QPushButton('-')
        button_zoom_out.clicked.connect(self.zoom_out)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.view)

        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(button_zoom_in)
        hbox.addWidget(button_zoom_out)

        self.setLayout(vbox)

    def load_model(self, filename):
        model = load_model(filename)
        self.set_model(model)

    def set_model(self, model):
        self.scene.clear()

        nodes = {}
        for node in model.nodes():
            if node.parent is None:
                N = Node(node)
                nodes[node] = N

                # Node labels
                N_label = QtGui.QGraphicsTextItem(node.name, parent=N)
                N_label.setPos(11, -22)
                self.scene.addItem(N)

        for edge in model.edges():
            E = Edge(edge)
            for node in E.edge:
                node.schematic_item.edges.add(E)
            self.scene.addItem(E)
            E.refresh()

        # center view on model
        rect = self.scene.itemsBoundingRect()
        self.view.setSceneRect(rect)

    def export_svg(self, path):
        buf = QtCore.QBuffer()
        generator = QtSvg.QSvgGenerator()
        generator.setOutputDevice(buf);
        generator.setFileName(path)
        generator.setTitle('Pywr')

        # TODO: this doesn't work as expected
        rect = self.scene.itemsBoundingRect()
        generator.setSize(self.size())
        generator.setResolution(300)
        #generator.setSize(QtCore.QSize(600, 400))
        #generator.setViewBox(QtCore.QRect(0, 0, rect.width(), rect.height()))

        # paint the scene
        painter = QtGui.QPainter()
        painter.begin(generator)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.scene.render(painter)
        painter.end()

    def export_png(self, path):
        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect)
        img = QtGui.QImage(self.scene.sceneRect().size().toSize(),
                            QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(img)
        self.scene.render(painter)
        img.save(path)

    scale_factor = 1.25
    def zoom_in(self):
        self.view.scale(self.scale_factor, self.scale_factor)
    def zoom_out(self):
        self.view.scale(1/self.scale_factor, 1/self.scale_factor)

class Node(QtGui.QGraphicsItem):
    def __init__(self, node, *args, **kwargs):
        super(Node, self).__init__(*args, **kwargs)
        self.node = node
        self.node.schematic_item = self
        self.edges = set()

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setVisible(node.visible)

        x, y = node.position
        self.setPos(x * 100, y * 100)
        self.snap()

    def paint(self, painter, option, widget):
        # Domain circle
        pen = QtGui.QPen()
        painter.setPen(QtCore.Qt.NoPen)
        brush = QtGui.QBrush(QtGui.QColor(self.node.domain.color))
        painter.setBrush(brush)
        painter.setOpacity(0.25)
        painter.drawEllipse(-25, -25, 50, 50)
        # Draw Node
        painter.setOpacity(1.0)
        pen = QtGui.QPen(QtGui.QBrush(QtCore.Qt.black), 2)
        painter.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(self.node.color))
        painter.setBrush(brush)
        painter.drawRoundedRect(-10, -10, 20, 20, 2, 2)


    def boundingRect(self):
        return QtCore.QRectF(-26, -26, 54, 54)

    def itemChange(self, change, value):
        if change is QtGui.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # node has moved, refresh edges
            for edge in self.edges:
                edge.refresh()
        return super(Node, self).itemChange(change, value)

    def mouseReleaseEvent(self, event):
        button = event.button()
        if button == QtCore.Qt.LeftButton:
            self.snap()
        elif button == QtCore.Qt.RightButton:
            pass
        return(super(Node, self).mouseReleaseEvent(event))

    def snap(self):
        '''Snap to grid'''
        center = self.pos()
        base = 20
        x = int(base * round(center.x()/base))
        y = int(base * round(center.y()/base))
        snapped_center = QtCore.QPointF(x, y)
        position = self.pos()
        position += (snapped_center-center)
        self.setPos(position)

class Edge(QtGui.QGraphicsLineItem):
    def __init__(self, edge, *args, **kwargs):
        super(Edge, self).__init__(*args, **kwargs)

        node1, node2 = edge
        if node1.parent:
            node1 = node1.parent
        if node2.parent:
            node2 = node2.parent

        self.edge = (node1, node2)

        # ensure edges draw below nodes
        self.setZValue(-1)

        # line style
        pen = QtGui.QPen()
        pen.setWidth(1)
        # TODO: this is crude - use an attribute on the node instead?
        # TODO: Should use domain setting.
        if False:
            color = '#0892D0'  # blue
        else:
            color = '#000'
        pen.setColor(color)
        self.setPen(pen)

    def refresh(self):
        """Update location of edge (start and end points)"""
        node1, node2 = self.edge
        pos1 = self.mapFromScene(node1.schematic_item.scenePos())
        pos2 = self.mapFromScene(node2.schematic_item.scenePos())
        centre = (pos1 + pos2)/2
        self.toward_point = (pos2 + centre)/2
        self.setLine(QtCore.QLineF(pos1, pos2))

    def paint(self, painter, option, widget):
        QtGui.QGraphicsLineItem.paint(self, painter, option, widget)
        pen = QtGui.QPen(QtGui.QBrush(QtCore.Qt.black), 5)
        painter.setPen(pen)
        painter.drawPoint(self.toward_point)


if __name__ == '__main__':
    app = QtGui.QApplication([])

    mainwindow = MainWindow()

    app.exec_()
