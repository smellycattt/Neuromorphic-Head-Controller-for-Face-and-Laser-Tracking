#!/usr/bin/env python

import rospy
import cv2
import math
import json
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError


class VisionNodeCenter:
    """ Vision node for robot head """

    def __init__(self):
        rospy.init_node("head_vision_node", anonymous=True)

        # Name of the camera
        self.camera_name = rospy.get_param('~camera_name')
        # Left or Right (0 for left, 1 for right)
        self.left_right = rospy.get_param('~left_right')
        # Red channel threshold
        self.red_threshold = rospy.get_param('~red_threshold')
        # Directory to receptive field file
        self.receptive_file_dir = rospy.get_param('~receptive_file_dir')
        # Publish frequency
        self.pub_freq = rospy.get_param('~pub_freq')
        # CV Bridge
        self.bridge = CvBridge()
        #storing last face position
        self.face_tracker=(360,360,20,20)
        #If in face detection mode
        self.face_detection=True
        # Subscriber to image
        # if self.face_detection:
        #     image_sub = rospy.Subscriber("/" + self.camera_name + "/image_raw", Image, self.image_face_cb)
        # else:
        image_sub = rospy.Subscriber("/" + self.camera_name + "/image_raw", Image, self.image_cb)
        # Output publisher
        self.pub = rospy.Publisher("/" + self.camera_name + "/control_output", Float32MultiArray, queue_size=1)

        
        # image_face_cb(self,msg)
        self.original_img=None
        # Get receptive field mask and center
        self.rf_mask, self.rf_center = self.generate_receptive_field()

        # Init Image
        self.raw_image = None
        rospy.loginfo("Waiting for Image input")
        while self.raw_image is None and not rospy.is_shutdown():
            continue
        rospy.loginfo("Image input init")

    def generate_receptive_field(self):
        """
        Generate receptive field information

        Returns:
            rf_mask: receptive field mask
            rt_center: receptive field center index
            rf_weight: receptive field weight

     """
        with open(self.receptive_file_dir) as f:
            pixel_2_rf = json.load(f)

        # Generate rf mask
        print("generating receptive field")
        rf_mask = np.zeros((720, 720))
        for xx in range(720):
            for yy in range(720):
                rf_idx = pixel_2_rf[str(xx) + ',' + str(yy)][0]
                rf_mask[xx, yy] = rf_idx

        # Generate rf center
        rf_num = int(np.max(rf_mask)) + 1
        rf_center = np.zeros((rf_num, 2))
        # print("rf_center",rf_center,rf_num)
        for rr in range(rf_num):
            pixels = np.argwhere(rf_mask == rr)
            np_pixels = np.array(pixels)
            rf_center[rr] = np.mean(np_pixels, axis=0)
        # print("no2",rf_center.shape,rf_mask.shape)
        return rf_mask, rf_center

    # def image_face_cb(self,msg):
    #     """
    #     Camera image for face callback funtion.
    #     Uses openCV library for face detection and convert it into a point focussing on the center of the face.
    #     """
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg)
    #     except CvBridgeError as e:
    #         print(e)
    #     # print(cv_image)
    #     # self.original_img=cv_image
    #     face_cascade = cv2.CascadeClassifier('/home/vaibhav/Downloads/haarcascade_frontalface_default.xml')
    #     #resize to 720x720
    #     cv_image=cv_image[:, 280:1000, :]
    #     # Convert into grayscale
    #     gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    #     print(gray.shape)
    #     # Detect faces
    #     faces = face_cascade.detectMultiScale(gray, 1.2, 6,minSize=(30,30))
    #     # Draw rectangle around the faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     self.original_img=cv_image 
    #     center_x=0
    #     center_y=0
    #     max_area=0
    #     for (x,y,w,h) in faces:
    #         if max_area<w*h:
    #             center_x=x+(w//2)
    #             center_y=y+(h//2)
    #             max_area=w*h
    #     if len(faces)!=0:
    #         self.face_tracker=[40,40]

    #     # print("faces",faces)    
    #     new_img=np.zeros((720,720), dtype='uint8')
    #     new_img[self.face_tracker[0],self.face_tracker[1]]=255
    #     print("face tracker",self.face_tracker)

    #     self.raw_image=new_img
    #     # print(self.raw_image.shape)
    #     # # Display the output
    #     # cv2.imshow('img', img)
    #     # cv2.waitKey()

    def image_cb(self, msg):
        """
        Camera image callback function

        Args:
            msg (message): Image message

        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)
        
        if self.face_detection:
            face_cascade = cv2.CascadeClassifier('/home/vaibhav/Downloads/haarcascade_frontalface_default.xml')
            #resize to 720x720
            cv_image=cv_image[:, 280:1000, :]
            # Convert into grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            # print(gray.shape)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.2, 6,minSize=(30,30))
            # faces = face_cascade.detectMultiScale(gray,
            #         scaleFactor=1.2,
            #         minNeighbors=5,
            #         minSize=(30, 30),
            #         flags = cv2.CV_HAAR_SCALE_IMAGE
            #     )
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # print("cv_image",cv_image.shape)
            self.original_img=cv_image
            center_x=0
            center_y=0
            width=0
            height=0
            max_area=0
            for (x,y,w,h) in faces:
                if max_area<w*h:
                    center_x=x+(w//2)
                    center_y=y+(h//2)
                    width=w
                    height=h
                    max_area=w*h
            if len(faces)!=0 :
                self.face_tracker=(center_y,center_x,height,width)
            else:
                self.face_tracker=(360,360,10,10)
                

           
            # print("faces",center_x,center_y)    
            new_img=np.zeros((720,720), dtype='uint8')
            new_img[self.face_tracker[0]-20:self.face_tracker[0]+20,self.face_tracker[1]-20:self.face_tracker[1]+20]=255
            # if len(faces)==0:
            #     if len(self.face_tracker)>1:
            #         self.face_tracker.pop(0)
            #     self.face_tracker.append([360,360])
            
            # for face in self.face_tracker:
            #     print("face",face)
            #     new_img[face[0]][face[1]]=255
            #     cv2.circle(new_img,(face[1],face[0]),4,color=(255,255,255),thickness=10)
            # print("face tracker",self.camera_name,self.face_tracker)
            # cv2.circle(new_im,(laser_point_center[0][1],laser_point_center[0][0]),4,color=(255,255,255),thickness=4)
            
            # print("face tracker",self.face_tracker)

            # self.raw_image=new_img
            # print("raw_img_center",np.argwhere(self.raw_image==255))
            self.raw_image=new_img
        else:
            img=cv_image[:,280:1000,:]
            img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # lower mask (0-10)
            # lower_red = np.array([0,50,50])
            # upper_red = np.array([10,255,255])
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper mask (170-180)
            # lower_red = np.array([170,50,50])
            # upper_red = np.array([180,255,255])
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0+mask1

            resize_red_image = cv_image[:, 280:1000, 0]


            # Threshold image
            _, th_image = cv2.threshold(resize_red_image, self.red_threshold, 255, cv2.THRESH_BINARY)
            # Change to numpy array
            

            
            self.raw_image = np.array(th_image, dtype='uint8')
            print("raw_img",self.raw_image.shape)
            print(np.argwhere(self.raw_image==255).mean(axis=0))
            # print(self.raw_image.shape)

    def update_control_output(self, control_output, rf_idx):
        """
        Update control output base on receptive field index

        Args:
            control_output (list): control output
            rf_idx (int): receptive field index

        """
        rf_center_x, rf_center_y = self.rf_center[rf_idx, 0], self.rf_center[rf_idx, 1]
        max_w = math.log(1 + 360.0 / 288.0)

        # Vertical Movement
        if rf_center_x < 360:  # Up
            control_output[0] += math.log(1 + (360.0 - rf_center_x)/288.0) / max_w
        elif rf_center_x > 360:  # Down
            control_output[1] += math.log(1 + (rf_center_x - 360.0)/288.0) / max_w

        # Horizontal Movement
        if rf_center_y < 360:  # Left
            control_output[2] += math.log(1 + (360.0 - rf_center_y)/288.0) / max_w
            # Addition dimension output for left right eye not added here
            # Detail see original code midbrain/new_brain.h/computeColliculusInput
        elif rf_center_y > 360:  # Right
            control_output[3] += math.log(1 + (rf_center_y - 360.0)/288.0) / max_w
            # Addition dimension output for left right eye not added here
            # Detail see original code midbrain/new_brain.h/computeColliculusInput

    def run_node(self):
        """
        Run Camera Node

        """
        ros_rate = rospy.Rate(self.pub_freq)

        while not rospy.is_shutdown():
            # if self.face_detection:
            #     laser_point_image = self.raw_image.copy()
            #     # Get pixel index for center of laser point pixel
            #     laser_point_pixel= np.argwhere(laser_point_image == 255)
            #     if laser_point_pixel.shape[0] > 0:
            #         laser_point_center = laser_point_pixel.mean(axis=0).astype(np.int)
            #         print("laser center",laser_point_center)
            #         rf_image = np.zeros((720, 720), dtype='uint8')
            #         control_output = [0, 0, 0, 0]  # Up, Down, Left, Right
            #         rf_idx = int(self.rf_mask[laser_point_center[0], laser_point_center[1]])
            #         print("rf_idx",rf_idx)
            #         self.update_control_output(control_output, rf_idx)
            #         rf_image[self.rf_mask == rf_idx] = 255
            #     cv2.circle(laser_point_image,(laser_point_center[1], laser_point_center[0]),4,color=(255,255,255),thickness=4)
            #     # print(np.argwhere(laser_point_image == 255))
            #     show_image = cv2.vconcat((laser_point_image, rf_image))
            #     cv2.imshow(self.camera_name, show_image)
            #     cv2.imshow(self.camera_name+"raw",self.original_img)
            #     cv2.waitKey(3)
            # else:
            #     laser_point_image = self.raw_image.copy()source ros_ws/devel/setup.bash
            #     # Get pixel index for center of laser point pixel
            #     laser_point_pixel = np.argwhere(laser_point_image == 255)
                
            #     rf_image = np.zeros((720, 720), dtype='uint8')
            #     control_output = [0, 0, 0, 0]  # Up, Down, Left, Right
            #     # print(laser_point_pixel.shape)
            #     if laser_point_pixel.shape[0] == 0:
            #         laser_point_pixel=np.array([[360,360]])
            #     # print(laser_point_pixel.shape)
            #     if laser_point_pixel.shape[0] > 0:
            #         laser_point_center = laser_point_pixel.mean(axis=0).astype(np.int)
                    
            #         rf_idx = int(self.rf_mask[laser_point_center[0], laser_point_center[1]])
            #         print("laser center",laser_point_center)
            #         self.update_control_output(control_output, rf_idx)
            #         print("rf_idx",rf_idx)
            #         rf_image[self.rf_mask == rf_idx] = 255
            #         # print(rf_image)
            #     # cv2.imshow(self.camera_name+"laser_img",laser_point_image)
            #     #) cv2.imshow(self.camera_name+"rf_img",rf_image)
            #     show_image = cv2.hconcat((laser_point_image, rf_image))
            #     cv2.imshow(self.camera_name, show_image)
            #     cv2.waitKey(3)
            laser_point_image = self.raw_image.copy()
            # Get pixel index for center of laser point pixel
            laser_point_pixel = np.argwhere(laser_point_image == 255)
            
            rf_image = np.zeros((720, 720), dtype='uint8')
            control_output = [0, 0, 0, 0]  # Up, Down, Left, Right
            # print(laser_point_pixel.shape)
            if laser_point_pixel.shape[0] == 0:
                laser_point_pixel=np.array([[360,360]])
            # print(laser_point_pixel.shape)
            if laser_point_pixel.shape[0] > 0:
                laser_point_center = laser_point_pixel.mean(axis=0).astype(np.int)
                
                rf_idx = int(self.rf_mask[laser_point_center[0], laser_point_center[1]])
                print("laser center",self.camera_name,laser_point_center)
                self.update_control_output(control_output, rf_idx)
                print("o/p",self.camera_name,control_output)
                print("rf_idx",self.camera_name,rf_idx)
                rf_image[self.rf_mask == rf_idx] = 255
                # print(rf_image)
            # cv2.imshow(self.camera_name+"laser_img",laser_point_image)
            # cv2.imshow(self.camera_name+"rf_img",rf_image
            # cv2.circle(laser_point_image,(laser_point_center[1], laser_point_center[0]),4,color=(255,255,255),thickness=4)
            if self.face_detection:
                cv2.imshow(self.camera_name+"raw",self.original_img)
            show_image = cv2.vconcat((laser_point_image, rf_image))
            cv2.imshow(self.camera_name, show_image)
            cv2.waitKey(3)
            # print(control_output)
            # Publish control output
            print(self.face_detection)
            pub_msg = Float32MultiArray()
            pub_msg.data = control_output
            self.pub.publish(pub_msg)

            ros_rate.sleep()


if __name__ == '__main__':
    node = VisionNodeCenter()
    node.run_node()
