# Learn to Infer Human Poses Using a Full-Body Pressure-Sensing Garment  
The dataset should only be used for research purpose.
If you would like to have access to the dataset, please send an email to philipzhang@mail.ustc.edu.cn with your intention, we would record and send you the url.

# Smart Garment: A Long-Term Feasible, Whole-Body Textile Pressure-Sensing System  
This repository holds the python demo of pressure data visualization system by Dongquan Zhang.  
## Quick start  
The demo is implemented using package PyQt5 v5.15.6, vtk v9.1.0 and OpenCV (opencv-contrib-python v4.6.0.66).  
You can check out the visualizing result by running demo_script.py using the following command.
```
python demo_script.py ./example/[cloths_data_file_name].npy ./example/[pants_data_file_name].npy
```

## Configuration file
Configuration file **./config/config.json** helps the system adapt to different data arrangements and sizes.  
![](./visualization_demo/fig/configuration_file.png)
The .json file records the location of horizontal/vertical sensing stripes on the human body. To make localization easier, the human body is viewed as 5 cylinders (left/right arms, left/right legs and torso). The sensing stripes cover the cylinders equally in horizontal and vertical directions.

Thus, the arrays recorded in **./config/config.json** represent the order of stripes on the cylinders. E.g., for cylinder left_arm **(class ARM (mode='left'))**, horizontal order starts from variable **self.reference** and vertical order starts from **self.start**.

For more detailed algorithm, please refer to our work “Smart Garment: A Long-Term Feasible, Whole-Body Textile Pressure-Sensing System”.

## Uses in publicated works
“A Single-Ply and Knit-Only Textile Sensing Matrix for Mapping Body Surface Pressure,” _IEEE Sensors Journal_, 2024.  
![](./visualization_demo/fig/use_in_IEEE_Sensors_2024.png)

“Smart Garment: A Long-Term Feasible, Whole-Body Textile Pressure-Sensing System,” _IEEE Sensors Journal_, 2023.  
![](./visualization_demo/fig/use_in_IEEE_Sensors_2023.png)
