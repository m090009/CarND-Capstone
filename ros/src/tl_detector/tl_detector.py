#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, Constants
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
from scipy import spatial
# import time


STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.waypoint_tree = None
        self.lights = []
        self.waypoints_2d = None


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # NOTE: image_raw
        # Don't really need the raw image the color image does fine
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # Checking to see if the system is on site or in simulation
        self.is_site = self.config['is_site']


        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        #Init classifier
        self.light_classifier = TLClassifier(self.is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Publish within a controlled loop 
        self.ros_loop()

    def ros_loop(self):
        '''
        Publish upcoming red lights at camera UPDATE_FREQUENCY (50Hz)
        to sync with the DBW system Carla. 
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''                                                                             
        # A publishing structure to publish at a UPDATE_FREQUENCY
        # of 50Hz to sync with the DBW system Carla
        rate = rospy.Rate(Constants.UPDATE_FREQUENCY)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints and self.camera_image:
                light_wp, state = self.process_traffic_lights()
         

                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """Initializes the waypoints and creates a lookup KD-Tree 
           to easily retrive closest waypoints to the car

        Args:
            waypoints : 200 /base_waypoints on the road

        """
        self.waypoints = waypoints
        # Check if waypints_2d needs to be initialized
        if not self.waypoints_2d:
            # Get the Waypoints KD tree to 
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = spatial.KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """ Sets the camera_image to the recieved raw message from 
            camera

        Args:
            msg (Image): image from car-mounted camera

        """
        # 
        self.has_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
            using a KD-Tree
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light
           by using the TLClassifier instance

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # If we don't get an image we return nothing
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        # Load image
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        #Return classified light
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Traffic light 
        closest_light = None
        # Traffic light line waypoint
        tl_line_wp_index = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position_wp = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            # Loop througth traffic lights 
            for i, light in enumerate(self.lights):
                # Get Stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypint index
                d = temp_wp_idx - car_position_wp
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    tl_line_wp_index = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return tl_line_wp_index, state

        # 
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
