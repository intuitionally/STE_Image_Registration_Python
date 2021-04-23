import cv2
import numpy as np

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
a_fname = "C:\\Users\\rdgrldkb\\PycharmProjects\\STE_Image_Registration_Python\\output\\apartments_lynx_15.tif"
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
for i in range(0, 1, a_skip):
    # Read frame A and parse filename for description.
    # a_fname = string(listA{i})   # TODO add back when we implement multiple image matching

    #Convert to gray scale
    img_a = cv2.imread(a_fname)
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    str_a = a_fname.split('\\')[-1].split('.')[0]

    # Stretch the image. I also switch to auto-gcp naming
    # as I copied this code from there and didn't feel like changing all
    # the names.So from here on usgs is A frame.
    usgs = img_a_gray
    #usgs = imAdjust(img_a_grey, stretchLim(img_a_grey, [0.0, 1])) #redundant?????

    # Initialize empty placeholders for best matches.
    bestInlierFMV =[]
    bestInlierUSGS =[]
    bestImgA = ''
    bestImgB = ''
    bestDesc = ''

    for j in range (1, length(listB), b_skip):

        # Read frame B and parse filename for description.
        fnameB = string(listB[j])
        imB = imread(fnameB)
        [~, strB] = fileparts(fnameB)

        # Get rotation angles from filename.
        temp = split(strA, '_')
        rA = str2double(temp[2][2:end])
        temp = split(strB, '_')
        rB = str2double(temp[2][2: end])

        # Check angles to make it run faster,
        # if rB < rA, rB = rB + 360, end
        # if rB < rA + settings.aLowAngle, continue, end
        # if rB > rA + settings.aHighAngle, continue, end

        angCheck = mod(abs(rA + settings.a2bAngle - rB), 360.0)
        angCheck = abs(mod(rA + settings.a2bAngle, 360.0) - rB)
        disp(sprintf("rA = #d, rB = #d, angCheck =  #d", rA, rB, angCheck))

        if angCheck > settings.angleFilter:
            continue

            # Display status message.
            strDesc = sprintf("#s-#s", strA, strB)
            disp(sprintf("Registering #s", strDesc))

        # Convert to gray scale and stretch. See comment above...
        # fmv is just B frame
        fmv = img_b_gray  # put imadjust back eventually
        # fmv = imadjust(imBG, stretchlim(imBG, [0.0, 1])) # redundant?????

        # Detect KAZE features for image A. Other feature detectors are
        # available in Matlab(SURF, ORB, etc.) I've found KAZE works best
        # for this type of work, but we probably should test others.     Should we test others???
        detector = cv2.AKAZE.create()

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
            if m.distance < 0.9*n.distance:
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
        tform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
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

        # Start assembling results for summary
        inlierCount = length(inlierFMV)
        matchCountList[end + 1] = inlierCount
        matchDescList[end + 1] = strDesc

        # Display user message
        print(f'Inliers:  {inlierCount}')

        # If there are more inliers in the registration, make it the new "best" registration.
        if inlierCount > length(bestInlierFMV):
            bestInlierFMV = inlierFMV
            bestInlierUSGS = inlierUSGS
            bestImgA = strA
            bestImgB = strB
            bestDesc = strDesc


        if settings.showPlot:
            # Display inliers and write JPG output file
            if ~exist('figInliers', 'var') | ~ishandle(figInliers):
                figInliers = figure
            figure(figInliers)
            showMatchedFeatures(usgs, fmv, inlierUSGS, inlierFMV, 'montage')
            title(sprintf("#s, #d inliers, #s", strDesc, inlierCount, settings.transformType), 'Interpreter', 'none')
            fname_matches = fullfile(settings.strOutDir, sprintf("match_#s.jpg", strDesc))
            saveas(gcf, fname_matches, 'jpeg')


        if length(inlierFMV) < settings.minInliers:
            continue

        fname_matches = f'{output_dir}_match_{strDesc}s.jpg'
        writeMatchFile(fname_matches, inliers.Location, inliers.Location)
        # writeMatchFile(fname_matches, inlierUSGS.Location, inlierFMV.Location)


    # Make sure we have sufficient inliers to write "best_match" file
    if length(bestInlierFMV) < settings.minInliers:
        continue

    # Write output file for 'best' match with i, j for A and i, j for B.
    fnameBestMatch = fullfile(settings.strOutDir, sprintf("best_match_#s.txt", bestDesc))
    writeMatchFile(fnameBestMatch, bestInlierUSGS.Location, bestInlierFMV.Location)

# Write summary file.
fnameMatchCounts = fullfile(settings.strOutDir, "_match_counts.txt")
fp = fopen(fnameMatchCounts, 'w')
for i in range (1, length(matchCountList), 1):
    fprintf(fp, "#s\t#.0f\n", matchDescList(i), matchCountList(i))
fclose(fp)

timing.timeTotal = synthImgTocStr(toc(ticTotal))

writestruct(struct('settings', settings, 'timing', timing), settings.fnameReport)

disp(timing)

def writeMatchFile(fname, usgs, fmv):
    fp = fopen(fname, 'w')
    fprintf(fp, "Ai\tAj\tBi\tBj\n")
    for i in range (1, size(usgs), 1):
        fprintf(fp, "#.0f\t#.0f\t#.0f\t#.0f\n", usgs(i,:), fmv(i,:))
    fclose(fp)