import cv2
import numpy as np
import time

show_plot = True

# matchFeatures() parameters
match_threshold = 20.0
max_ratio = 0.9

# estimateGeometricTransform() parameters
transform_type = 'projective' # 'similarity', 'affine', or 'projective'
confidence = 90.0
max_distance = 6.0

min_inliers = 18

# low = 0 and high = 360 iterates over all
a_low = 0  # angle for a frames
a_high = 360

# FLW scans
# a2b_angle = 0
angle_filter = 30

# Apartment
a2b_angle = 315
# a2b_angle = 285
# settings.angleFilter = 30
# settings.a2bAngle = 0
# settings.angleFilter = 45
# settings.aLowAngle = 210
# settings.aHighAngle = 240

# Frames to skip, in this case use every 5th 'A' frame and every 'B' frame
a_skip = 1
b_skip = 1

# 'A' frames are C10 in this instance.
# settings.listASearchStr =
# 'F:\software_development\STE_image_registration\Lynx_C10_Pseudo_Images\C10_TransparentBackground\*.png'
# settings.listASearchStr = 'F:\data\ste_apartment_registration\pseudo_images\apt_*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\apt_*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\hangrtc360*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\scan5*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\scan52*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\apt_*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\output\zebchop_*rgb.tif'
# settings.listASearchStr = 'F:\software_development\STE_image_registration\sample_data\apartments\images\5cm\zebchop_*rgb.tif'
a_fname = "C:\\Users\\rdgrldkb\\PycharmProjects\\STE_Image_Registration_Python\\output\\apartments_local_0.tif"
# listA = dir_b_fullpath(settings.listASearchStr) TODO add back when we implement multiple image matching

# 'B' frames of Lynx in this instance.
# settings.listBSearchStr = 'F:\software_development\STE_image_registration\Lynx_C10_Pseudo_Images\Lynx_TransparentBackground\*.png')
# settings.listBSearchStr = 'F:\data\ste_apartment_registration\pseudo_images\lynx_*rgb.tif'
# settings.listBSearchStr = 'F:\software_development\STE_image_registration\output\lynx_*rgb.tif'
# settings.listBSearchStr = 'F:\software_development\STE_image_registration\output\hangp50*rgb.tif'

b_fname = "C:\\Users\\rdgrldkb\\PycharmProjects\\STE_Image_Registration_Python\\output\\apartments_lynx_0.tif"

# Directory to which we write results
# settings.strOutDir = "f:\temp"
# settings.strOutDir = "F:\data\ste_apartment_registration\image_registration"
output_dir = "C:\\Users\\rdgrldkb\\PycharmProjects\\STE_Image_Registration_Python\\output\\matched"
report_fname = f'{output_dir}_ste_reg_report.xml'

# Count of matches list and description list for summary file.

matchCountList = []
matchDescList = ''


def imAdjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def writeMatchFile(fname, usgs, fmv):
    with open(fname, 'w') as f:
        f.write("Ai\tAj\tBi\tBj\n")
        for i in range(0, len(usgs)):
            f.write(f'{usgs[i, 0]}\t{usgs[i, 1]}\t{fmv[i, 0]}\t{fmv[i, 1]}')


time_start = time.time()
for i in range(0, 1, a_skip):
    # Read frame A and parse filename for description.
    # a_fname = string(listA{i})   # TODO add back when we implement multiple image matching

    # Convert to gray scale
    img_a = cv2.imread(a_fname)
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    str_a = a_fname.split('\\')[-1].split('.')[0]

    # Stretch the image. I also switch to auto-gcp naming
    # as I copied this code from there and didn't feel like changing all
    # the names.So from here on usgs is A frame.
    usgs = img_a_gray
    # usgs = imAdjust(img_a_grey, stretchLim(img_a_grey, [0.0, 1])) #redundant?????

    # Initialize empty placeholders for best matches.
    bestInlierFMV = []
    bestInlierUSGS = []
    bestImgA = ''
    bestImgB = ''
    bestDesc = ''

    for j in range(0, 1, b_skip):

        # Read frame B and parse filename for description.
        # b_fname = str(listB[j])
        img_b = cv2.imread(b_fname)
        img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        str_b = b_fname.split('\\')[-1].split('.')[0]

        # Get rotation angles from filename.
        temp = str_a.split('_')
        print(f'stra {str_a} temp {temp}')
        rA = float(temp[-1])
        temp1 = str_b.split('_')
        rB = float(temp1[-1])

        # Check angles to make it run faster,
        # if rB < rA, rB = rB + 360, end
        # if rB < rA + settings.aLowAngle, continue, end
        # if rB > rA + settings.aHighAngle, continue, end

        angCheck = np.abs(rA + a2b_angle - rB) % 360.0
        # angCheck = np.abs((rA + settings.a2bAngle) % 360.0 - rB)
        print(f"rA = {rA}, rB = {rB}, angCheck =  {angCheck}")

        # if angCheck > angle_filter:
        #     continue

        # Display status message.
        strDesc = f'{str_a} - {str_b}'
        print(strDesc)

        # Convert to gray scale and stretch. See comment above...
        # fmv is just B frame
        fmv = img_b_gray  # put imadjust back eventually
        # fmv = imadjust(imBG, stretchlim(imBG, [0.0, 1])) # redundant?????

        # Detect KAZE features for image A. Other feature detectors are
        # available in Matlab(SURF, ORB, etc.) I've found KAZE works best
        # for this type of work, but we probably should test others. Should we test others???
        detector = cv2.AKAZE.create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
                                    threshold=0.0001)

        # detect and extract image A features
        (keyPtsUSGS, descUSGS) = detector.detectAndCompute(img_a_gray, None)

        # detect and extract features for image B
        (keyPtsFMV, descFMV) = detector.detectAndCompute(img_b_gray, None)

        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Match features between images. There will be a lot of matched features, most of them bad
        matchPairs = bf.knnMatch(descUSGS, descFMV, k=2)
        # indexPairs = (featuresUSGS, featuresFMV, queryDescriptors = keyPtsUSGS, trainingDescriptors = keyPtsFMV)  # Threshold?????

        good = []
        for m, n in matchPairs:
            if m.distance < 0.6*n.distance:
                good.append([m])

        # Get points for matched features.
        # print(descUSGS)
        # print(keyPtsUSGS[0])
        # print(matchPairs)
        # matchedUSGS = keyPtsUSGS(indexPairs[:, 0])  # ask jeff what these are
        # matchedFMV = keyPtsFMV(indexPairs[:, 1])

        # Display matched features.It'll look like a bowl of spaghetti.

        if show_plot:
            im3 = cv2.drawMatchesKnn(img_a, keyPtsUSGS, img_b, keyPtsFMV, good, None, flags=2)
            cv2.imshow('AKAZE matching', im3)
            cv2.waitKey()

        # Display user message
        print(f'Matches:  {len(matchPairs)}')

        src_pts = np.float32([keyPtsUSGS[m[0].queryIdx].pt for m in good])  # .reshape(-1, 1, 2)
        print(f'length of fmv: {len(keyPtsFMV)} length of good: {len(good)}')

        dst_pts = np.float32([keyPtsFMV[m[0].trainIdx].pt for m in good])  # .reshape(-1, 1, 2)
        # Use estimateGeometricTransform to find the best matches from a lot of bad matches
        tform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=6)
        inliers = np.nonzero(mask.ravel().tolist())
        # print(mask)
        print(inliers)

        if show_plot:
            im3 = cv2.drawMatchesKnn(img_a, keyPtsUSGS, img_b, keyPtsFMV, good, None, matchesMask=mask, flags=2)
            cv2.imshow('inlier stoof', im3)
            cv2.waitKey()

        # [tform, inlierFMV, inlierUSGS, status] = estimateGeometricTransform(
        # matchedFMV, matchedUSGS,
        # settings.transformType,
        # 'MaxNumTrials', settings.MaxNumTrials,
        # 'Confidence', settings.Confidence,
        # 'MaxDistance', settings.MaxDistance)
    #
    #     # Start assembling results for summary
        inlierCount = len(inliers)
        #inlierCount = len(inlierFMV)
    #     matchCountList.append(inlierCount)
    #     matchDescList.append(strDesc)
    #
    #     # Display user message
    #     print(f'Inliers:  {inlierCount}')
    #
    #     # If there are more inliers in the registration, make it the new "best" registration.
    #     if inlierCount > len(bestInlierFMV):
    #         bestInlierFMV = inliers
    #         bestInlierUSGS = inliers
    #         # bestInlierFMV = inlierFMV
    #         # bestInlierUSGS = inlierUSGS
    #         bestImgA = str_a
    #         bestImgB = str_b
    #         bestDesc = strDesc
    #
    #     if show_plot:
    #         # Display inliers and write JPG output file
    #         im3 = cv2.drawMatchesKnn(img_a, img_b, inliers, inliers, None, flags=2)
    #         # showMatchedFeatures(usgs, fmv, inlierUSGS, inlierFMV, 'montage')
    #         # title(sprintf("#s, #d inliers, #s", strDesc, inlierCount, transform_type), 'Interpreter', 'none')
    #         fname_matches = f'{output_dir}_match_{strDesc}s.jpg'
    #         cv2.imwrite(fname_matches, im3)
    #
    #     if len(inliers) < min_inliers:
    #         continue
    #
    #     fname_matches = f'{output_dir}_match_{strDesc}s.jpg'
    #     writeMatchFile(fname_matches, inliers.Location, inliers.Location)
    #     # writeMatchFile(fname_matches, inlierUSGS.Location, inlierFMV.Location)
    #
    #
    # # Make sure we have sufficient inliers to write "best_match" file
    # if len(bestInlierFMV) < min_inliers:
    #     continue
    #
    # # Write output file for 'best' match with i, j for A and i, j for B.
    # fnameBestMatch = f'{output_dir}_best_match_{bestDesc}.txt'
    # writeMatchFile(fnameBestMatch, bestInlierUSGS.Location, bestInlierFMV.Location)

# Write summary file.
fnameMatchCounts = f'{output_dir}_match_counts.txt'
with open(fnameMatchCounts, 'w') as f:
    for i in range(0, len(matchCountList)):
        f.write(f'{matchDescList[i]}\t{matchCountList[i]}')

time_end = time.time()
time_total = time_end - time_start
print(f'total time: {time_total}')

