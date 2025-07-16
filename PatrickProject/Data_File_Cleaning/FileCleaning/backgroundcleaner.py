import cv2
import numpy as np

# 1. Load image (BGR)
img = cv2.imread('/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/Data_File_Cleaning/FileCleaning/BLAH.png', cv2.IMREAD_UNCHANGED)
if img is None:
    raise SystemExit("Could not read input.png")

# 2. Convert to HSV so we can threshold by “lightness”
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Define a range for your background color.
#    Here we pick very light (white/cream) pixels:
lower_bg = np.array([0, 0, 200])    # H: any, S: low, V: high
upper_bg = np.array([180, 30, 255]) # tweak S/V if needed

# 4. Build a mask where background pixels = 255
bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)

# 5. Invert mask to get the foreground
fg_mask = cv2.bitwise_not(bg_mask)

# 6. Split original image channels
b, g, r = cv2.split(img)

# 7. Use the fg_mask as the alpha channel
alpha = fg_mask

# 8. Merge B,G,R and new alpha
rgba = cv2.merge([b, g, r, alpha])

# 9. Save result with transparency
cv2.imwrite('output.png', rgba)

print("Saved output.png with transparent background.")
