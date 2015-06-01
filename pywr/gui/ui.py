#!/usr/bin/env python

from PySide import QtCore, QtGui

from .. import core, xmlutils

class Node(QtGui.QGraphicsItem):
    def __init__(self, node, scene, *args, **kwargs):
        self.node = node
        self.node.schematic = self
        self.edges = []
        
        ret = super(Node, self).__init__(*args, **kwargs)
        
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, True)
        scene.addItem(self)
        
        # node label
        text = self.text = QtGui.QGraphicsTextItem(parent=self)
        text.setPlainText(node.name)
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        font.setFamily('Arial')
        text.setFont(font)
        text.setPos(10,0)
        text.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        
        # position
        x, y = node.position
        self.setPos(x * 100, -y * 100)
        self.snap()
        
        return ret
    
    def paint(self, painter, option, widget):
        pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor('black')), 2)
        painter.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(self.node.color))
        painter.setBrush(brush)
        painter.drawRoundedRect(-10, -10, 20, 20, 2, 2)
    
    def boundingRect(self):
        return QtCore.QRectF(-12, -12, 24, 24)
    
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
            self.show_menu()
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
    
    def show_menu(self):
        '''Display contextual menu'''
        cls_name = self.node.__class__.__name__
        menu = self.menu = QtGui.QMenu()
        action = menu.addAction(cls_name)
        action.setEnabled(False)
        action = menu.addAction('Properties')
        menu.popup(QtGui.QCursor.pos())
    
    def set_label(self, text):
        self.text.setPlainText('{}\n{}'.format(self.node.name, text))

class Edge(QtGui.QGraphicsLineItem):
    def __init__(self, edge, *args, **kwargs):
        self.edge = edge
        super(Edge, self).__init__(*args, **kwargs)
        self.setZValue(-1)
        node1, node2 = self.edge
        # style
        pen = QtGui.QPen()
        pen.setWidth(2)
        # TODO: this is crude - use an attribute on the node instead?
        if(isinstance(node1, (core.River, core.Catchment, core.Terminator)) \
         and isinstance(node2, (core.River, core.Catchment, core.Terminator))):
            color = '#0892D0'  # blue
        else:
            color = '#000'
        pen.setColor(color)
        self.setPen(pen)
        # position
        self.refresh()
    def refresh(self):
        node1, node2 = self.edge
        pos1 = self.mapFromScene(node1.schematic.scenePos())
        pos2 = self.mapFromScene(node2.schematic.scenePos())
        self.setLine(QtCore.QLineF(pos1, pos2))

class PywrSchematic(QtGui.QDialog):
    def __init__(self, filename):
        super(PywrSchematic, self).__init__()
        
        self.scene = QtGui.QGraphicsScene(self)
        
        # read model from xml
        with open(filename, 'r') as f:
            data = f.read()
        self.model = xmlutils.parse_xml(data)
        
        graph = self.model.graph
        nodes = graph.nodes()
        edges = graph.edges()
        
        # add nodes
        for node in nodes:
            n = Node(node, self.scene)
        
        # add edges
        for edge in edges:
            e = Edge(edge)
            for node in edge:
                node.schematic.edges.append(e)
            self.scene.addItem(e)
            
        # fix the area shown by the scene
        rect = self.scene.sceneRect()
        self.scene.setSceneRect(rect)
        
        # create a view into the scene
        self.view = QtGui.QGraphicsView(self.scene, self)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.view.scale(1, 1)
        
        # add view to the dialog
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.view)
        
        # TODO: move this into Qt Designer
        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        spacer1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        spacer2 = QtGui.QSpacerItem(10, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        button_step = self.button_step = QtGui.QPushButton('Step')
        button_step.clicked.connect(self.step)
        timestamp_label = self.timestamp_label = QtGui.QLabel('')
        hbox.addItem(spacer1)
        hbox.addWidget(timestamp_label)
        hbox.addItem(spacer2)
        hbox.addWidget(button_step)
        vbox.addItem(hbox)
        
        self.setLayout(vbox)
        
        self.resize(700, 500)
        self.setWindowTitle('Pywr schematic')
        
        self.show()
        self.raise_()
    
    def step(self):
        result = self.model.step()
        for node, amount in result[4].items():
            node.schematic.set_label('{:.3f}'.format(amount))
        self.timestamp_label.setText((self.model.timestamp-self.model.parameters['timestep']).strftime('%Y-%m-%d'))

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication([])
    dialog = PywrSchematic(sys.argv[-1])
    app.exec_()
