import cv2
from timer import Timer
import largestinteriorrectangle


overall_timer = Timer("Overall Time")

# Create a stitcher object
stitcher = cv2.Stitcher_create()
stitcher.setPanoConfidenceThresh(0.1)

# create a list of three images
# left_images = ["./2024-02-27/cam1/2024-02-27-13-36-30.jpg", "./2024-02-27/cam1/2024-02-27-13-38-30.jpg", "./2024-02-27/cam1/2024-02-27-13-38-58.jpg"]
# right_images = ["./2024-02-27/cam0/2024-02-27-13-36-30.jpg", "./2024-02-27/cam0/2024-02-27-13-38-30.jpg", "./2024-02-27/cam0/2024-02-27-13-38-58.jpg"]

left_images = ["left4.jpg", "./2024-02-27/cam1/2024-02-27-13-38-30.jpg", "./2024-02-27/cam1/2024-02-27-13-38-58.jpg"]
right_images = ["right4.jpg", "./2024-02-27/cam0/2024-02-27-13-38-30.jpg", "./2024-02-27/cam0/2024-02-27-13-38-58.jpg"]

for i in range(3):
    # Read images
    #left_img = cv2.imread("./2024-02-27/cam1/2024-02-27-13-36-30.jpg")
    #right_img = cv2.imread("./2024-02-27/cam0/2024-02-27-13-36-30.jpg")
    left_img = cv2.imread(left_images[i])
    right_img = cv2.imread(right_images[i])

    images = [left_img, right_img]

    # Stitch images
    status, result = stitcher.stitch(images)

    # Check if stitching was successful
    if status == cv2.Stitcher_OK:
        print("Stitching successful")
        # Further processing on the stitched result
    else:
        print("Stitching failed, retrying")
        status, result = stitcher.stitch((left_img, right_img))

    # Convert result to grayscale
    mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # resize the mask to accelerate the speed
    resized_mask = cv2.resize(mask, (mask.shape[1] // 3, mask.shape[0] // 3))

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(resized_mask, 0, 255, cv2.THRESH_BINARY)

    # Find contours on the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)


    # Calculate the largest interior rectangle
    x, y, width, height = largestinteriorrectangle.lir(binary_mask > 0, contour.squeeze())


    # Restore coordinates
    x*=3
    y*=3
    width*=3
    height*=3

    # Calculate the coordinates of the rectangle
    x2 = x + width
    y2 = y + height

    # Crop the rectangle region from the original image
    cropped_img = result[y:y2, x:x2]

    # Save the cropped image
    cv2.imwrite('cropped_result{i}.jpg', cropped_img)

overall_timer.stop()
