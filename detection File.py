"""
Project Title: Artificial Intelligence Based Waste Sorter

This is the main script of our final year project

Main script to run the object detection 
"""
#importing import libraries
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import RPi.GPIO as GPIO
import time

# Set GPIO numbering mode
GPIO.setmode(GPIO.BOARD)

# Set pin 11 as an output, and set servo1 as pin 11 as PWM
GPIO.setup(11,GPIO.OUT)
servo1 = GPIO.PWM(11,50) # Note 11 is pin, 50 = 50Hz pulse
#thres = 0.45 # Threshold to detect object

# Initially servo staring at zero
servo1.start(0)
print ("Waiting for 1 seconds")
time.sleep(2)

# Servo turning to its 90 degrees original position
print ("Original Position (Turning Servo to 90 degrees)")
servo1.ChangeDutyCycle(2+(90/18))
time.sleep(0.5)
servo1.ChangeDutyCycle(0)


time.sleep(1)



def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Arguments:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, score_threshold=0.35)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    
        
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('Waste_Sorting_object_detector', image)
    #Printing the class of the Detected object and taking decision whether the object lies in recyclable category or nonrecyclable category 
    if (detection_result.detections):
        print((detection_result.detections[0].categories[0].category_name[:]))
        
        if detection_result.detections[0].categories[0].category_name[:] == 'R':
            #Turning servo motor to 30 degrees whenever recyclable waste is detected
            print ("--------------------------------------------------------------")
            print ("\n")
            print ("Recyclable waste is detected move the waste into recyclabe Bin")
            print ("\n")
            print ("--------------------------------------------------------------")
            servo1.ChangeDutyCycle(2+(35/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)
            
            time.sleep(1)
            
            # Servo turning to its 90 degrees original position
            print ("Original Position (Turning Servo to 90 degrees)")
            servo1.ChangeDutyCycle(2+(90/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)

            servo1.ChangeDutyCycle(2+(90/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)
            
            time.sleep(1)
            
        elif detection_result.detections[0].categories[0].category_name[:] == 'NR':
            #Turning servo motor to 30 degrees whenever recyclable waste is detected
            print ("--------------------------------------------------------------")
            print ("\n")
            print ("NonRecyclable waste is detected move the waste into nonrecyclabe Bin")
            print ("\n")
            print ("--------------------------------------------------------------")
            servo1.ChangeDutyCycle(2+(140/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)
            
            time.sleep(1)
            
            # Servo turning to its 90 degrees original position
            print ("Original Position (Turning Servo to 90 degrees)")
            servo1.ChangeDutyCycle(2+(95/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)

            servo1.ChangeDutyCycle(2+(95/18))
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)
            
            time.sleep(1)
    
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()



