import cv2
from math import cos, degrees, sin, radians
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import open3d as o3d

GINGERBREAD = 1
SIMPLE = 2
model = GINGERBREAD

if model == GINGERBREAD:
    # reading point cloud into numpy array
    gb_file = "C:\\Users\\rdgrlcl9\\Documents\\STE_Python\\gb_house\\gbhouse.txt"
    point_cloud = np.loadtxt(gb_file).astype(np.float)

    fname_stub = 'output/gingerbread'

    # first three columms are point positions
    world_points = point_cloud[:, :3].astype(np.float)

    # last three columns are point colors
    world_colors = point_cloud[:, 3:6]

    # setting known values
    focal_length = 400.0
    image_i = 600
    image_j = 400

    # separating x, y, z values into their own respective arrays
    x_points = point_cloud[:, 0]
    y_points = point_cloud[:, 1]
    z_points = point_cloud[:, 2]

    t_vec = np.array([0, 0, 10.0]).astype(np.float)

elif model == SIMPLE:
    fname_stub = 'output/simple'
    focal_length = 5
    image_i = 50
    image_j = 50

    world_points = np.array([[0, 0, 0],  # origin
                            [1.0, 0, 0],  # x axis, red
                            [0, 1.0, 0],  # y axis, green
                            [0, 0, 1.0]])  # z axis, blue

    world_colors = np.array([[0, 0, 0],  # black
                            [255, 0, 0],  # red
                            [0, 255, 0],  # green
                            [0, 0, 255]])  # blue

    x_points = [i[0] for i in world_points]
    y_points = [i[1] for i in world_points]
    z_points = [i[2] for i in world_points]

    t_vec = np.array([0, 0, 0]).astype(np.float)

# getting the midpoints from each array
xMid = np.mean([np.min(x_points), np.max(x_points)])
yMid = np.mean([np.min(y_points), np.max(y_points)])
zMid = np.mean([np.min(z_points), np.max(z_points)])

offsets = [xMid, yMid, zMid]

# intrinsics matrix
# f_x skew=0 c_x
# 0   f_y  c_y
# 0   0    1

# f_x = f_y for this model
# assume no skew?
# c_x and c_y make up the principal point, set to 0

intrinsics = np.array([[focal_length, 0, image_j/2],
                      [0, focal_length, image_i/2],
                      [0, 0, 1.0]]).astype(np.float)

# getting rotation matrix
rx = -90
ry = 0
rz = 0

for i in range(0, 89, 30):
    ry = i

    Rx = np.array([[1, 0, 0],
                  [0, cos(radians(rx)), -sin(radians(rx))],
                  [0, sin(radians(rx)), cos(radians(rx))]])

    Ry = np.array([[cos(radians(ry)), 0, sin(radians(ry))],
                   [0, 1, 0],
                   [-sin(radians(ry)), 0, cos(radians(ry))]])

    Rz = np.array([[cos(radians(rz)), -sin(radians(rz)), 0],
                   [sin(radians(rz)), cos(radians(rz)), 0],
                   [0, 0, 1.0]])

    R = np.matmul(np.matmul(Rz, Ry), Rx)

    # sample rotation matrix = I
    # R = np.identity(3)

    # Get camera position to use in the code below to calculate depth.
    camPos = (np.matmul(R, t_vec))

    # get rotation vector
    r_vec = cv2.Rodrigues(R)[0].astype(np.float)

    # This is the money function.  Project world points to image points.
    # world_points = np.swapaxes(world_points, 0, 1)
    projected_pts, jacobian = cv2.projectPoints(world_points, r_vec, t_vec, intrinsics, distCoeffs=0)
    projected_pts = projected_pts[0:, 0, 0:]

    # Find good points inside the window.  These will be points with:
    # 1 <= i <= image_i *and* 1 <= j <= image_j.
    # Find indices greater than 1 for i,j
    keep1 = np.where(np.min(projected_pts, axis=1) >= 1.0)

    # Find indices with i <= image_i
    keep2 = np.where(projected_pts[0:, 1] <= image_i)

    # Find indices with j <= image_j
    keep3 = np.where(projected_pts[0:, 0] <= image_j)

    # Logical AND the three sets above to get our keeper indicies
    keep = np.intersect1d(keep1, np.intersect1d(keep2, keep3))

    # Get keeper XYZ
    wptsKeep = world_points[keep]

    # Get keeper projected i,j points
    projectedPoints = projected_pts[keep]

    # Get keeper colors
    colorsKeep = world_colors[keep]

    x_list = projectedPoints[0:, 0]
    y_list = projectedPoints[0:, 1]
    # fig = plt.figure()
    # ax = fig.add_subplot()
    ax = plt.axes()

    ax.scatter(x_list, y_list, c=colorsKeep/255, s=0.01)

    plt.show()

