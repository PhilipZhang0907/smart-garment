import class_garment
import class_qt

import sys
import numpy as np
from PyQt5 import QtWidgets

if __name__ == '__main__':

    cloths_example, pants_example = sys.argv[1:]

    smart_garment = class_garment.SMART_GARMENT(config_file_path=f'./config/config.json', 
                                                obj_file_path=f'./config/20230108_man_2.obj')
    smart_garment.set_pressure_data(np.load(cloths_example), 
                                    np.load(pants_example))
    scalar = smart_garment.make_vtk_scalar()

    app = QtWidgets.QApplication(sys.argv)
    win = class_qt.View()
    win.reader.GetOutput().GetPointData().SetScalars(scalar)
    win.mapper.SetScalarRange(0, 600)
    win.iren.Initialize()
    win.show()
    sys.exit(app.exec_())