import cv2
from math import cos, sin, radians
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pathlib import Path
import tifffile as tiff

# TODO multiprocess

GINGERBREAD = 1
APARTMENTS = 2
RANCH = 3
SIMPLE = 4
SIMPLE2 = 5


model = GINGERBREAD

write_depth = False
write_xyz = True
write_az = False


if model == GINGERBREAD:
    # reading point cloud into numpy array
    gb_file = "C:\\Users\\rdgrldkb\\Documents\\gbhouse.txt"
    point_cloud = np.loadtxt(gb_file).astype(np.float)

    fname_stub = 'output/gingerbread'

    # first three columms are point positions
    world_points = point_cloud[:, :3].astype(np.float64)

    # last three columns are point colors
    world_colors = point_cloud[:, 3:6].astype(np.float64)

    # setting known values
    focal_length = 500
    image_i = 600
    image_j = 400

    # separating x, y, z values into their own respective arrays
    x_points = point_cloud[:, 0]
    y_points = point_cloud[:, 1]
    z_points = point_cloud[:, 2]

    # camera views the scene from this position
    t_vec = np.array([[0, 0, 10]]).astype(np.float)


elif model == APARTMENTS:
    # reading point cloud into numpy array
    apartment_file = "C:\\Users\\rdgrldkb\\Documents\\Apartments_C10_Local.ply"
    point_cloud = o3d.io.read_point_cloud(apartment_file)

    fname_stub = 'output/apartments'

    world_points = np.asarray(point_cloud.points)

    # last three columns are point colors
    world_colors = np.asarray(point_cloud.colors) * 255
    print(f'colors {world_colors[0]}')

    # setting known values
    focal_length = 10000
    image_i = 500
    image_j = 500

    # separating x, y, z values into their own respective arrays
    x_points = world_points[:, 0]
    y_points = world_points[:, 1]
    z_points = world_points[:, 2]

    # camera views the scene from this position
    t_vec = np.array([[0, 0, 500]]).astype(np.float)

elif model == RANCH:
    # reading point cloud into numpy array
    ranch_file = "C:\\Users\\rdgrldkb\\Documents\\OberRanchRTC360_barn.ply"
    point_cloud = o3d.io.read_point_cloud(ranch_file)

    fname_stub = 'output/ranch'

    world_points = np.asarray(point_cloud.points)

    # last three columns are point colors
    world_colors = np.asarray(point_cloud.colors) * 255
    print(f'colors {world_colors[0]}')

    # setting known values
    focal_length = 8000
    image_i = 200
    image_j = 400

    # separating x, y, z values into their own respective arrays
    x_points = world_points[:, 0]
    y_points = world_points[:, 1]
    z_points = world_points[:, 2]

    # camera views the scene from this position
    t_vec = np.array([[0, 2, 900]]).astype(np.float)

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

    x_points = [i[0] for i in world_points]
    y_points = [i[1] for i in world_points]
    z_points = [i[2] for i in world_points]

    t_vec = np.array([[0, 10, 0]]).astype(np.float)

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


# name: convert_polar
# inputs: xyz - an array of xy points (nx32)
#         camPos - current camera position
# description: returns a list of thetas corresponding to the
# spherical polar coordinates for each xyz point
def convert_polar(xy, camPos):
    x = xy[:, 0] + camPos[0]
    y = xy[:, 1] + camPos[1]

    return np.degrees(np.arctan2(y, x))  # theta probably


# name: test_plot
# inputs: num - figure number
#         pts - points to plot
#         colors - colors corresponding to points in pts
# description: test function to plot
def test_plot(num, pts, colors):
    x_list = pts[:, 1]
    y_list = pts[:, 0]

    plt.figure(num)
    ax = plt.axes()
    ax.scatter(x_list, y_list, c=colors/255, s=0.01)


# name: keep_in_frame
# inputs: projected_pts - set of points projected to an image frame
# description: removes points outside of the viewing window
def keep_in_frame(projected_pts):
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
    return keep


# name: keep_in_az
# inputs: wptsKeep - already reduced set of projected points
#         camPos - camera position
#         fov_low  - lower bound in degrees of fov
#         fov_high - upper bound in degrees of fov
# description: removes points outside of the camera's field of view
def keep_in_az(wptsKeep, camPos, fov_low, fov_high):
    wpts_keep_az = convert_polar(wptsKeep, camPos) - 90  # just getting theta

    # convert negative to positive
    wpts_keep_az[wpts_keep_az < 0] += 360

    # monkey business if low is negative
    if fov_low < 0:
        wpts_keep_az[wpts_keep_az >= (360 + fov_low)] -= 360

    # monkey business if high is over 360
    if fov_high > 360:
        wpts_keep_az[wpts_keep_az <= (fov_high - 360)] += 360

    keep_indices = np.argwhere(np.logical_and((wpts_keep_az >= fov_low), (wpts_keep_az <= fov_high)))
    return keep_indices, wpts_keep_az


# name: main
# inputs: wpts - point cloud model
# description: projects 3d point cloud to 2d at different angles
def main(wpts):

    world_points = wpts

    # field of view
    fov = np.degrees(np.arctan2(image_j / 2, focal_length))

    # getting the midpoints from each array so that the model can be centered at the origin
    xMid = np.mean([np.min(x_points), np.max(x_points)])
    yMid = np.mean([np.min(y_points), np.max(y_points)])
    zMid = np.mean([np.min(z_points), np.max(z_points)])

    offsets = [xMid, yMid, zMid]

    # center the model
    world_points -= offsets

    # assuming f_x = f_y and that there is no skew
    intrinsics = np.array([[focal_length, 0, image_j / 2],
                           [0, focal_length, image_i / 2],
                           [0, 0, 1.0]]).astype(np.float)

    # setting initial rotation values so we can rotate around the sides of the model
    rx = 90
    ry = 0
    rz = 0

    # loop through angles that are desired
    for i in range(0, 1, 30):
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
        img_depth = np.array(np.ones((image_i, image_j))*np.inf)
        img_az = np.empty(shape=(image_i, image_j))

        rz = i

        Rx = np.array([[1.0, 0, 0],
                      [0, cos(radians(rx)), -sin(radians(rx))],
                      [0, sin(radians(rx)), cos(radians(rx))]])

        Ry = np.array([[cos(radians(ry)), 0, sin(radians(ry))],
                       [0, 1.0, 0],
                       [-sin(radians(ry)), 0, cos(radians(ry))]])

        Rz = np.array([[cos(radians(rz)), -sin(radians(rz)), 0],
                       [sin(radians(rz)), cos(radians(rz)), 0],
                       [0, 0, 1.0]])

        R = np.matmul(np.matmul(Rz, Ry), Rx)

        # getting the transformation matrix so that the axes are rotated along with the model
        R = np.transpose(R)

        # Get camera position to use in the code below to calculate depth.
        camPos = np.matmul(np.negative(np.transpose(R)), np.transpose(t_vec))
        print(camPos)

        # get rotation vector
        r_vec, _ = cv2.Rodrigues(R)
        # This is the money function.  Project world points to image points.
        projected_pts = cv2.projectPoints(world_points, r_vec, t_vec, intrinsics, distCoeffs=0)[0]

        # reshape to nx2
        projected_pts = projected_pts[0:, 0, 0:]

        # find which projected points are still in the frame
        keep = keep_in_frame(projected_pts)

        # Get keeper XYZ
        wptsKeep = world_points[keep]

        # Get keeper projected i,j points
        projected_pts = projected_pts[keep]

        # Get keeper colors
        colorsKeep = world_colors[keep]

        # find which points are behind the camera to avoid false projections
        keep_indices, wpts_keep_az = keep_in_az(wptsKeep, camPos, i-fov, i+fov)

        wptsKeep = wptsKeep[keep_indices]
        projected_pts = projected_pts[keep_indices]
        colorsKeep = colorsKeep[keep_indices]

        # reshape
        projected_pts = projected_pts[:, 0, :]
        colorsKeep = colorsKeep[:, 0, :]

        # making sure campos is the same shape as a point so the elements don't get
        # broadcasted weirdly in the depth calculation
        camPos = np.transpose(camPos)

        # Fill in data for TIFFs based on depth (we want the closest point if 2+ overlap)
        for n in range(0, (np.size(projected_pts, 0))):
            # Get current pixel i,j
            ii = round(projected_pts[n, 1]) - 1
            jj = round(projected_pts[n, 0]) - 1

            # Compute delta XYZ from the camera to the current point.
            d = np.linalg.norm(wptsKeep[n, :] - camPos)

            # if we already have a closer point in this pixel, skip the current one
            if d > img_depth[ii, jj]:
                continue

            # If we get here we have a good point.
            # Update depth map.
            img_depth[ii, jj] = d

            # Update RGB image.
            img[ii, jj, :] = colorsKeep[n, :]

            # Update XYZ image.
            imgXYZ[ii, jj, :] = wptsKeep[n, :] + offsets

            # Update azimuth image.
            img_az[ii, jj] = wpts_keep_az[n]

        # flipping everything correctly in the tiff file
        img = np.flip(img, 0)
        imgXYZ = np.flip(imgXYZ, 0)
        img_depth = np.flip(img_depth, 0)
        img_az = np.flip(img_az, 0)

        img = np.flip(img, 1)
        imgXYZ = np.flip(imgXYZ, 1)
        img_depth = np.flip(img_depth, 1)
        img_az = np.flip(img_az, 1)

        # Write out images.
        tiff.imwrite(fname_img, img)
        if write_xyz:
            tiff.imwrite(fname_xyz_img, imgXYZ)
        if write_depth:
            tiff.imwrite(fname_depth_img, img_depth)
        if write_az:
            tiff.imwrite(fname_az_img, img_az)


if __name__ == '__main__':
    main(world_points)