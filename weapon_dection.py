"""
Weapon Detection System using OpenCV Haar Cascade
==================================================
This script uses a webcam to detect weapons in real-time using a trained Haar cascade classifier.

What this program does:
1. Opens your webcam
2. Captures video frames continuously
3. Analyzes each frame to detect weapons
4. Draws rectangles around detected weapons
5. Shows "GUN DETECTED!" or "NO GUN" on the video feed
6. Prints a summary when you quit (press 'q')
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np       # Used for numerical operations (array handling)
import cv2              # OpenCV - computer vision library for image/video processing
import imutils          # Helper functions for image operations (resize, rotate, etc.)
import datetime         # For timestamping (not actively used in this version)


# ============================================
# STEP 1: LOAD THE WEAPON DETECTION MODEL
# ============================================

# CascadeClassifier: Loads a pre-trained Haar cascade XML file
# This file contains the "pattern" of what a weapon looks like
# The classifier will use this pattern to find weapons in images
gun_cascade = cv2.CascadeClassifier('cascade_1.2.xml')

# Check if the cascade file loaded successfully
# .empty() returns True if the file wasn't found or is invalid
if gun_cascade.empty():
    raise FileNotFoundError("cascade_1.2.xml not found or failed to load. Check the file path.")


# ============================================
# STEP 2: OPEN THE WEBCAM
# ============================================

# VideoCapture(0): Opens the default webcam (index 0)
# If you have multiple cameras, try VideoCapture(1), VideoCapture(2), etc.
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
# .isOpened() returns True if the camera is accessible
if not camera.isOpened():
    raise RuntimeError("Failed to open camera at index 0. Check permissions and ensure no other app is using it.")


# ============================================
# STEP 3: INITIALIZE VARIABLES
# ============================================

# first_frame: Stores the first frame captured (can be used for motion detection)
first_frame = None

# gun_exists: Flag to track if a weapon was detected at any point during the session
# False = no weapon detected yet, True = weapon was detected at least once
gun_exists = False


# ============================================
# STEP 4: MAIN DETECTION LOOP
# ============================================

# This loop runs continuously until you press 'q' to quit
while True:
    
    # --- 4.1: CAPTURE A FRAME FROM THE CAMERA ---
    
    # .read() returns two values:
    #   ret: Boolean (True if frame captured successfully, False otherwise)
    #   frame: The actual image/frame from the camera (numpy array)
    ret, frame = camera.read()
    
    # Validate that we got a frame
    # If ret is False or frame is None, something went wrong
    if not ret or frame is None:
        print("Warning: Failed to capture frame, retrying...")
        cv2.waitKey(50)  # Wait 50ms before trying again
        continue         # Skip the rest of this loop iteration and try again
    
    
    # --- 4.2: PREPROCESS THE FRAME ---
    
    # Resize the frame to 500 pixels wide (maintains aspect ratio)
    # Why? Smaller images = faster processing, and detection works fine at this size
    frame = imutils.resize(frame, width=500)
    
    # Convert the frame to grayscale (black and white)
    # Why? Haar cascades work on grayscale images (color isn't needed for detection)
    # COLOR_BGR2GRAY: OpenCV uses BGR color order (not RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # --- 4.3: DETECT WEAPONS IN THE FRAME ---
    
    # detectMultiScale: Scans the image at multiple scales to find objects matching the cascade pattern
    # Parameters:
    #   gray: The grayscale image to scan
    #   scaleFactor=1.05: How much to shrink the image at each scale (1.05 = 5% smaller each time)
    #                     Smaller values = more thorough but slower
    #   minNeighbors=25: How many overlapping detections needed to confirm an object
    #                    Higher values = fewer false positives but might miss some real objects
    #   minSize=(120, 120): Minimum object size in pixels (width, height)
    #                       Objects smaller than this will be ignored
    # Returns: A list of rectangles (x, y, width, height) where weapons were found
    gun = gun_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=25, minSize=(120, 120))
    
    
    # --- 4.4: PROCESS DETECTION RESULTS ---
    
    # Check if any weapons were detected
    # len(gun) > 0 means at least one weapon was found
    if len(gun) > 0:
        
        # Set the flag to True (we found a weapon at least once)
        gun_exists = True
        
        # putText: Draws text on the frame
        # Parameters:
        #   frame: The image to draw on
        #   "GUN DETECTED!": The text to display
        #   (10, 50): Position (x, y) in pixels from top-left corner
        #   cv2.FONT_HERSHEY_SIMPLEX: Font style
        #   1: Font scale (size)
        #   (0, 0, 255): Color in BGR format (Blue=0, Green=0, Red=255 = RED)
        #   2: Thickness of the text in pixels
        cv2.putText(frame, "GUN DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Loop through each detected weapon (gun is a list of rectangles)
        for (x, y, w, h) in gun:
            # x, y: Top-left corner of the detected weapon
            # w, h: Width and height of the detected weapon
            
            # rectangle: Draws a rectangle on the frame
            # Parameters:
            #   frame: The image to draw on
            #   (x, y): Top-left corner
            #   (x + w, y + h): Bottom-right corner
            #   (0, 0, 255): Color in BGR (RED)
            #   2: Thickness in pixels
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # ROI (Region of Interest): Extract the detected area from the image
            # This can be used for further analysis (not used in this basic version)
            # roi_gray: The weapon area from the grayscale image
            # roi_color: The weapon area from the color image
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
    
    else:
        # No weapons detected in this frame
        # Display "NO GUN" in green
        # (0, 255, 0) = Green in BGR format
        cv2.putText(frame, "NO GUN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    # --- 4.5: STORE THE FIRST FRAME (OPTIONAL) ---
    
    # If this is the first frame we've captured, save it
    # This can be useful for motion detection or background subtraction (not used here)
    if first_frame is None:
        first_frame = gray.copy()  # .copy() creates a separate copy (not just a reference)
    
    
    # --- 4.6: DISPLAY THE FRAME ---
    
    # imshow: Creates a window and displays the frame
    # Parameters:
    #   "Security Feed": Window title
    #   frame: The image to display (with rectangles and text drawn on it)
    cv2.imshow("Security Feed", frame)
    
    
    # --- 4.7: CHECK FOR USER INPUT ---
    
    # waitKey(1): Wait 1 millisecond for a keyboard press
    # Returns the ASCII code of the pressed key, or -1 if no key was pressed
    # & 0xFF: Masks to get only the last 8 bits (handles platform differences)
    key = cv2.waitKey(1) & 0xFF
    
    # Check if the user pressed 'q' (quit)
    # ord("q"): Converts the character 'q' to its ASCII code (113)
    if key == ord("q"):
        break  # Exit the while loop (stop the program)


# ============================================
# STEP 5: CLEANUP AND FINAL SUMMARY
# ============================================

# Print a summary message after the loop ends (user pressed 'q')
if gun_exists:
    print("Gun Found")  # At least one weapon was detected during the session
else:
    print("Gun Not Found")  # No weapons were detected during the entire session

# Release the camera (stop capturing video)
# This frees up the camera for other applications
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()