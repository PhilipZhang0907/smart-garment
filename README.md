# Smart Garment: A long-term feasible, whole-body textile pressure sensing system  
This repository holds the python demo of pressure data visualization system by Dongquan Zhang.  
## Quick start  
The demo is implemented using package PyQt5 v5.15.6, vtk v9.1.0 and OpenCV (opencv-contrib-python v4.6.0.66).  
You can check out the visualizing result by running class_qt.py.  
We list 32 kinds of pressure distribution under different actions. To see other pressure data, change code in file class_garment.py line 15 and 16.  
``` python
18: CLOTHS_EXAMPLE = DIR + "/example/stand-4-cloths.npy"
19: PANTS_EXAMPLE = DIR + "/example/stand-4-pants.npy"
```
Default visualizing procedure upsamples the pressure image. If upsampling is not wanted, change code in class_garment.py:  
``` python
29: self.config = json.load(file)["upsample"]
44: self.cloths_data = cv2.pyrUp(cloths_data)
45: self.pants_data = cv2.pyrUp(pants_data)
```
to  
``` python
29: self.config = json.load(file)["normal"]
44: self.cloths_data = cloths_data
45: self.pants_data = pants_data
```