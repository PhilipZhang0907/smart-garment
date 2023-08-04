import sys
from PyQt5 import QtWidgets
import vtk
import numpy as np
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import class_garment

class Ui(QtWidgets.QWidget):
    def SetUpUi(self, main_window: QtWidgets.QMainWindow):
        '''
        
        setting up ui of vtk window
        :param MainWindow: serve as central widget
        '''
        main_window.setObjectName("MainWindow")
        main_window.resize(603, 553)
        self.main_window = main_window
        self.central_widget = QtWidgets.QWidget(main_window)
        self.grid_layout = QtWidgets.QGridLayout(self.central_widget)
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.grid_layout.addWidget(self.vtk_widget, 0, 0, 3, 1)
        main_window.setCentralWidget(self.central_widget)

class View(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        '''
        
        set ui layout
        '''
        self.ui = Ui()
        self.ui.SetUpUi(self)
        '''
        
        set vtk renderer
        '''
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.5, 0.5, 0.5)
        '''
        
        set camera focus
        '''
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(0, 100, 500)
        camera.SetFocalPoint(0, 100, 0)
        self.ui.vtk_widget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.ui.vtk_widget.GetRenderWindow().GetInteractor()
        '''
        
        connect vtk, .obj file and texture
        '''
        self.reader = vtk.vtkOBJReader()
        self.reader.SetFileName(class_garment.OBJ_FILE_PATH)
        self.reader.Update()
        self.texture = self.GetTexture(class_garment.TEXTURE_FILE_PATH)
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        lut = vtk.vtkLookupTable()
        lut.SetAlphaRange(1, 1)
        lut.SetHueRange(0.67, 0)
        lut.SetNumberOfColors(256)
        lut.Build()
        '''
        
        value 0 is inserted to distinguish area cover and not covered by garment
        area covered by garment is colored from blue to red as pressure grows
        area not covered by garment shows original texture
        '''
        lut.SetTableValue(0, [128,128,128,1])
        self.mapper.SetLookupTable(lut)
        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        actor.SetTexture(self.texture)
        self.ren.AddActor(actor)
        self.iren.Initialize()
    def GetTexture(self, texture_path: str):
        '''
        
        load texture for OBJ model
        :param texture_path: file path to load .bmp texture
        '''
        texReader = vtk.vtkBMPReader()
        texReader.SetFileName(texture_path)
        texReader.Update()
        texture = vtk.vtkTexture()
        texture.SetInputConnection(texReader.GetOutputPort())
        texture.Update()
        return texture

if __name__ == '__main__':
    smart_garment = class_garment.SMART_GARMENT(class_garment.CONFIG_FILE_PATH)
    smart_garment.set_pressure_data(np.load(class_garment.CLOTHS_EXAMPLE), 
                                    np.load(class_garment.PANTS_EXAMPLE))
    scalar = smart_garment.make_vtk_scalar()
    app = QtWidgets.QApplication(sys.argv)
    win = View()
    win.reader.GetOutput().GetPointData().SetScalars(scalar)
    win.mapper.SetScalarRange(0, 600)
    win.iren.Initialize()
    win.show()
    sys.exit(app.exec_())
