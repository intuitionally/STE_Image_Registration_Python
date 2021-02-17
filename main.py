import cv2
from PIL import Image
from math import cos, degrees, sin, radians
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import open3d as o3d
import tifffile as tiff

GINGERBREAD = 1
SIMPLE = 2
model = GINGERBREAD

write_depth = True
write_xyz= True
write_az = True


if model == GINGERBREAD:
    # reading point cloud into numpy array
    # gb_file = "C:\\Users\\rdgrlcl9\\Documents\\STE_Python\\gb_house\\gbhouse.txt"
    gb_file = "C:\\Users\\rdgrldkb\\Documents\\gbhouse.txt"
    point_cloud = np.loadtxt(gb_file).astype(np.float)

    fname_stub = 'output/gingerbread'

    # first three columms are point positions
    world_points = point_cloud[:, :3].astype(np.float64)

    # last three columns are point colors
    world_colors = point_cloud[:, 3:6].astype(np.float64)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(world_points)
    # pcd.colors = o3d.utility.Vector3dVector(world_colors/255)
    # o3d.visualization.draw_geometries([pcd])

    # setting known values
    focal_length = 400.0
    image_i = 600
    image_j = 400

    # separating x, y, z values into their own respective arrays
    x_points = point_cloud[:, 0]
    y_points = point_cloud[:, 1]
    z_points = point_cloud[:, 2]

    t_vec = np.array([[0, 0, 10.0]]).astype(np.float)

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

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(world_points)
    # pcd.colors = o3d.utility.Vector3dVector(world_colors)
    # o3d.visualization.draw_geometries([pcd])

    x_points = [i[0] for i in world_points]
    y_points = [i[1] for i in world_points]
    z_points = [i[2] for i in world_points]

    t_vec = np.array([[0, 0, 0]]).astype(np.float)

# field of view
fov = np.degrees(np.arctan2(image_j/2, focal_length))
print(f'fov: {fov}')

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


# make sure array shape is nx3
def convert_polar(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    return np.degrees(np.arctan2(y, x))  # theta probably


def test_plot(num, pts, colors):
    pts = pts[:, 0, :]
    x_list = pts[:, 0]
    y_list = pts[:, 1]
    colors = colors[:, 0, :]

    plt.figure(num)
    ax = plt.axes()
    ax.scatter(x_list, y_list, c=colors/255, s=0.01)
    # plt.show()


def main():
    for i in range(0, 359, 30):
        # base filename
        fname_base = f'{fname_stub}_{i}'

        # synthetic image file name
        fname_syn_img = f'{fname_base}_rgb.tif'

        # Image name to contain XYZ coordinates for each pixel in the
        # synthetic image.
        fname_xyz_img = f'{fname_base}_xyz.tif'

        # Image name to contain depth values.
        fname_depth_img = f'{fname_base}_depth.tif'

        # Image name to contain azimuth values.
        fname_az_img = f'{fname_base}_az.tif'

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
        camPos = np.linalg.pinv(np.matmul(R, np.negative(np.linalg.pinv(t_vec))))

        # get rotation vector
        r_vec = cv2.Rodrigues(R)[0].astype(np.float)

        # This is the money function.  Project world points to image points.
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

        # find the intersection of the keep indices
        keep = np.intersect1d(keep1, np.intersect1d(keep2, keep3))

        # Get keeper XYZ
        wptsKeep = world_points[keep]

        # Get keeper projected i,j points
        projectedPoints = projected_pts[keep]

        # Get keeper colors
        colorsKeep = world_colors[keep]

        # test_plot(projectedPoints, colorsKeep)

        # POLAR STUFFFFF
        wpts_keep_az = convert_polar(wptsKeep) - 90  # just getting theta
        print(f'length of wpts_keep_az: {len(wpts_keep_az)}')

        # convert negative to positive
        wpts_keep_az[wpts_keep_az < 0] += 360

        # fov extents
        fov_low = i - fov
        fov_high = i + fov

        # monkey business if low is negative
        if fov_low < 0:
            wpts_keep_az[wpts_keep_az >= 360 + fov_low] -= 360

        # monkey business if high is over 360
        if fov_high > 360:
            wpts_keep_az[wpts_keep_az <= fov_high - 360] += 360
        print(f'fov_low: {fov_low}, fov_high: {fov_high}, {wpts_keep_az >= fov_low}, {wpts_keep_az <= fov_high}')
        keep_indices = np.argwhere(np.logical_and((wpts_keep_az >= fov_low), (wpts_keep_az <= fov_high)))
        # TODO: talk to Jeff about smol slice
        print(f'length of keep_indices: {len(keep_indices)}')

        wpts_keep_az = wpts_keep_az[keep_indices]
        wptsKeep = wptsKeep[keep_indices]
        projected_points = projectedPoints[keep_indices]
        colorsKeep = colorsKeep[keep_indices]
        print(f'shape of projected_points: {projected_points.shape}')
        test_plot(i, projected_points, colorsKeep)

        colorsKeep = colorsKeep[:, 0, :]

        # print(wpts_keep_az[0:20])
        projected_points = projected_points[:, 0, :]
        x_list = projected_points[:, 0]
        y_list = projected_points[:, 1]
        # fig = plt.figure()
        # ax = fig.add_subplot()

        # ax = plt.axes()

        # ax.scatter(x_list, y_list, c=colorsKeep/255, s=0.01)
        # ax.scatter(projected_pts[:, 0], projected_pts[:, 1], c=world_colors / 255, s=0.01)

        # plt.show()



        # tifffile.imsave

        # Write out images.
        # imwrite(img,fnameImg);
        # if write_xyz_img:
        #     tiff.imwrite('imgXYZ',fnameXYZImg)
        # if write_depth:
        #     tiff.imwrite('depth',fnameDepthImg)
        # if write_az:
        #     tiff.imwrite('imgAz',fnameAzImg)

        # This function writes out a TIFF with 64-bit floating point numbers.
        # The standard Matlab image write functions can't do this.
        # function writeXYZ(data,fname)
        # t = Tiff(fname, 'w');
        # tagstruct.ImageLength = size(data, 1);
        # tagstruct.ImageWidth = size(data, 2);
        # tagstruct.Compression = Tiff.Compression.None;
        # tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
        # tagstruct.Photometric = Tiff.Photometric.LinearRaw;
        # tagstruct.BitsPerSample = 64;
        # tagstruct.SamplesPerPixel = size(data,3);
        # tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        # t.setTag(tagstruct);
        # t.write(data);
        # t.close();

    plt.show()


if __name__ == '__main__':
    main()
