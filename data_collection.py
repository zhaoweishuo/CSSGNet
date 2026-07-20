
import time
import point_generate
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas

client = RemoteAPIClient()
sim = client.require('sim')


points = point_generate.generate_cube_point(side_length=0.2, radian_range=1, num_samples=10000)

target_handle = sim.getObject('/UR5/target')
tip_handle = sim.getObject('/UR5/tip')
origin_position = sim.getObjectPosition(tip_handle)
origin_orientation = sim.getObjectOrientation(tip_handle)


sensor_handle = sim.getObject('/UR5/Vision_sensor')

count = 1

for i in points:


    new_position = [origin_position[0] + i[0], origin_position[1] + i[1], origin_position[2] + i[2]]
    new_orientation = [origin_orientation[0] + i[3], origin_orientation[1] + i[4], origin_orientation[2] + i[5]]


    buildmatrix = sim.buildMatrix(new_position, new_orientation)
    sim.setObjectMatrix(target_handle, buildmatrix)
    time.sleep(0.4)

    image, resolution = sim.getVisionSensorImg(sensor_handle)
    image = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    image = np.flipud(image)
    image = Image.fromarray(image)


    real_position = sim.getObjectPosition(tip_handle)
    real_orientation = sim.getObjectOrientation(tip_handle)

    image.save("./dataset/10000/Image/"+str(count)+".png")  # 保存图片
    label = {
        "name": str(count)+".png",
        "x": real_position[0],
        "y": real_position[1],
        "z": real_position[2],
        "a": real_orientation[0],
        "b": real_orientation[1],
        "g": real_orientation[2]
    }
    data = pandas.DataFrame(data=label, index=[0])  #
    data.to_csv("./dataset/10000/label.csv", mode='a', index=False, header=False)  # 保存数据




    print("Image Name: {}".format(str(count)+".png"))
    count += 1


sim.setObjectPosition(target_handle, origin_position)
sim.setObjectOrientation(target_handle, origin_orientation)

