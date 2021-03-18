import cv2
from PIL import Image
from math import cos, dist, sin, radians
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import open3d as o3d
from pathlib import Path
import tifffile as tiff

GINGERBREAD = 1
SIMPLE = 2
SIMPLE2 = 3
model = GINGERBREAD

write_depth = True
write_xyz = True
write_az = False


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

    t_vec = np.array([[0, 0, 10]]).astype(np.float)

elif model == SIMPLE:
    fname_stub = 'output/simple'
    focal_length = 5
    image_i = 50
    image_j = 50

    world_points = np.array([[0, 0, 0],  # origin
                            [1.0, 0, 0],  # x axis, red
                            [0, 1.0, 0],  # y axis, green
                            [0, 0, 1.0]])  # z axis, blue

    world_colors = np.array([[128, 0, 128],  # purple
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

    t_vec = np.array([[0, 0, 10]]).astype(np.float)

elif model == SIMPLE2:
    fname_stub = 'output/simple2'

    world_points = np.array([[1.0, 1, 1],  # purple
                             [1, -1, 1],   # bright red
                             [-1, -1, 1],  # light green
                             [-1, 1, 1],   # light blue
                             [1, 1, -1],   # gray
                             [1, -1, -1],  # maroon
                             [-1, -1, -1], # dark green
                             [-1, 1, -1]]) # dark blue

    world_colors = np.array([[128, 0, 128],  # purple
                      [255, 0, 0],     # bright red
                      [0, 255, 0],     # light green
                      [0, 0, 255],     # light blue
                      [127, 127, 127],  # gray
                      [127, 0, 0],     # maroon
                      [0, 127, 0],     # dark green
                      [0, 0, 127]])    # dark blue

    focal_length = 10
    image_i = 50
    image_j = 50

    x_points = [i[0] for i in world_points]
    y_points = [i[1] for i in world_points]
    z_points = [i[2] for i in world_points]

    t_vec = np.array([[0, 0, 0.5]]).astype(np.float)

# field of view
fov = np.degrees(np.arctan2(image_j/2, focal_length))
print(f'fov: {fov}')

# getting the midpoints from each array
xMid = np.mean([np.min(x_points), np.max(x_points)])
yMid = np.mean([np.min(y_points), np.max(y_points)])
zMid = np.mean([np.min(z_points), np.max(z_points)])

offsets = [xMid, yMid, zMid]

world_points -= offsets


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
rx = 0
# An initial rotation matrix somehow changes which axis gets rotated around!!!
# world axes don't change when model is rotated? is this expected?
ry = 0
rz = 0


# make sure array shape is nx3
def convert_polar(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    return np.degrees(np.arctan2(y, x))  # theta probably


def test_plot(num, pts, colors):
    # pts = pts[:, 0, :]
    x_list = pts[:, 0]
    y_list = pts[:, 1]
    # colors = colors[:, 0, :]

    plt.figure(num)
    ax = plt.axes()
    ax.set_facecolor
    ax.scatter(x_list, y_list, c=colors/255)
    # plt.show()


def main():
    for i in range(0, 90, 90):
        # base filename
        fname_base = f'{fname_stub}_{i}'
        fname_img = Path(f'{fname_base}.tif')

        # Image name to contain XYZ coordinates for each pixel in the
        # synthetic image.
        fname_xyz_img = Path(f'{fname_base}_xyz.tif')

        # Image name to contain depth values.
        fname_depth_img = Path(f'{fname_base}_depth.tif')

        # Image name to contain azimuth values.
        fname_az_img = Path(f'{fname_base}_az.tif')

        # remove files if they exist
        fname_img.unlink(missing_ok=True)
        fname_xyz_img.unlink(missing_ok=True)
        fname_depth_img.unlink(missing_ok=True)
        fname_az_img.unlink(missing_ok=True)

        img = np.zeros(shape=(image_i, image_j, 3), dtype=np.uint8)
        imgXYZ = np.empty(shape=(image_i, image_j, 3))
        img_depth = np.array(np.ones((image_i, image_j))*500)
        img_az = np.empty(shape=(image_i, image_j))

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

        # Get camera position to use in the code below to calculate depth.
        # camPos = np.linalg.pinv(np.matmul(R, np.negative(np.linalg.pinv(t_vec))))
        camPos = np.linalg.pinv(np.matmul(R, np.linalg.pinv(np.negative(t_vec))))
        print(f'camera position: {camPos}')
        # camPos = np.matmul(np.negative(np.transpose(R)), np.transpose(t_vec))

        # get rotation vector
        r_vec, _ = cv2.Rodrigues(R)

        # This is the money function.  Project world points to image points.
        projected_pts = cv2.projectPoints(world_points, r_vec, t_vec, intrinsics, distCoeffs=0)[0]
        projected_pts = projected_pts[0:, 0, 0:]
        print(f'projected: {projected_pts}')
        # test_plot(i, projected_pts, world_colors)

        # Find good points inside the window.  These will be points with:
        # 1 <= i <= image_i *and* 1 <= j <= image_j.
        # Find indices greater than 1 for i,j
        # keep1 = np.where(np.min(projected_pts, axis=1) >= 1.0)
        #
        # # Find indices with i <= image_i
        # keep2 = np.where(projected_pts[0:, 1] <= image_i)
        #
        # # Find indices with j <= image_j
        # keep3 = np.where(projected_pts[0:, 0] <= image_j)
        #
        # # find the intersection of the keep indices
        # keep = np.intersect1d(keep1, np.intersect1d(keep2, keep3))
        #
        # # Get keeper XYZ
        # wptsKeep = world_points[keep]
        #
        # # Get keeper projected i,j points
        # projected_pts = projected_pts[keep]
        # # print(f'post keeps {len(projected_pts)}')
        #
        # # Get keeper colors
        # colorsKeep = world_colors[keep]

        # test_plot(i, projected_pts, colorsKeep)

        # POLAR STUFFFFF
        wpts_keep_az = convert_polar(wptsKeep, camPos) - 90  # just getting theta
        # print(f'length of wpts_keep_az: {len(wpts_keep_az)}')
        # print(f'wpts_keep_az: {wpts_keep_az}')

        # convert negative to positive
        wpts_keep_az[wpts_keep_az < 0] += 360

        # fov extents
        fov_low = i - fov

        fov_high = i + fov

        # monkey business if low is negative
        if fov_low < 0:
            wpts_keep_az[wpts_keep_az >= (360 + fov_low)] -= 360

        # monkey business if high is over 360
        if fov_high > 360:
            wpts_keep_az[wpts_keep_az <= (fov_high - 360)] += 360
        # print(f'fov_low: {fov_low}, fov_high: {fov_high}, {wpts_keep_az >= fov_low}, {wpts_keep_az <= fov_high}')

        keep_indices = np.argwhere(np.logical_and((wpts_keep_az >= fov_low), (wpts_keep_az <= fov_high)))
        # keep_indices = np.argwhere(wpts_keep_az >= fov_low)
        # keep_indices = np.argwhere(wpts_keep_az <= fov_high)

        wpts_keep_az = wpts_keep_az[keep_indices]

        # plt.hist(wpts_keep_az, 90)
        # plt.title(f'Angle={i}, fovlow={fov_low}, fovhigh={fov_high}')
        # plt.show()

        wptsKeep = wptsKeep[keep_indices]
        projected_pts = projected_pts[keep_indices]
        colorsKeep = colorsKeep[keep_indices]

        # wptsKeep = world_points
        # colorsKeep = world_colors
        projected_pts = projected_pts[:, 0, :]
        # print(f'size of projected_pts {len(projected_pts)}')
        colorsKeep = colorsKeep[:, 0, :]
        # test_plot(i, projected_pts, colorsKeep)

        for n in range(0, (np.size(projected_pts, 0))):
            # Get current pixel i,j
            ii = round(projected_pts[n, 1])

            jj = round(projected_pts[n, 0])

            # Compute delta XYZ from the camera to the current point.

            d = np.linalg.norm(camPos[0] - wptsKeep[n, :])

            # If the current depth is greater than the depth map for i,j,
            # continue to next point.

            if d > img_depth[ii, jj]:
                # print(f'greater i: {ii}, j: {jj}, depth: {d}')
                continue
            # print(f'i: {ii}, j: {jj}, depth: {d}')
            # If we get here we have a good point.
            # Update depth map.
            img_depth[ii, jj] = d

            # Update RGB image.
            img[ii, jj, :] = colorsKeep[n, :]

            # Update XYZ image.
            imgXYZ[ii, jj, :] = wptsKeep[n, :] + offsets

            # Update azimuth image.
            # img_az[ii,jj] = wptsKeepAz[n]

        # tifffile.imsave

        img = np.flip(img, 0)
        imgXYZ = np.flip(imgXYZ, 0)
        img_depth = np.flip(img_depth, 0)
        img_az = np.flip(img_az, 0)

        # Write out images.
        tiff.imwrite(fname_img, img)
        if write_xyz:
            tiff.imwrite(fname_xyz_img, imgXYZ, dtype=np.float64)
        if write_depth:
            tiff.imwrite(fname_depth_img, img_depth, dtype=np.float64)
        if write_az:
            tiff.imwrite(fname_az_img, img_az, dtype=np.float64)


    # plt.show()


if __name__ == '__main__':
    main()
