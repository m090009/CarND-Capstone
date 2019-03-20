#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import PIL
import yaml
import numpy as np
from scipy import spatial

# Tensorflow imports 
import tensorflow as tflow
from tensorflow.python.platform import gfile


STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.waypoint_tree = None
        self.lights = []


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
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # Checking to see if the system is on site or in simulation
        self.is_site = self.config['is_site']
        # Init model 
        self.init_model()

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def init_model(self):
        print('Loading model ....')
        # Trained model classes
        self.classes = ['Green',
        'Red',
        'Yellow',
        'GreenLeft',
        'RedLeft',
        'RedRight',
        'RedStraight',
        'RedStraightLeft',
        'GreenRight',
        'GreenStraightRight',
        'GreenStraightLeft',
        'GreenStraight',
        'off']
        # Models paths
        models_path = "../../../models/"
        sim_model_name = "sim-FL-2400.pb"
        site_model_name = "site-FL.pb"
        # Model path, by default its going to be the sim model path
        model_path = models_path + sim_model_name 
        self.classes = self.classes[:3]
        if self.is_site:
            # model_graph = 
            # Switch to the model site model path
            model_path = models_path + site_model_name
        # Init tensorflow variables 
        self.tf_graph = self.load_tf_graph(model_path)
        self.tf_sess = tflow.InteractiveSession() 
        # Set session's graph (load model)
        self.tf_sess.graph.as_default()
        tflow.import_graph_def(self.tf_graph, name='')

        self.input_image = self.tf_sess.graph.get_tensor_by_name("0:0")
        self.outputs1 = self.tf_sess.graph.get_tensor_by_name("concat_52:0")
        self.outputs2 = self.tf_sess.graph.get_tensor_by_name("concat_53:0")
        print("Model loaded \\0/")
    
    def load_tf_graph(self, graph_path):
        print('load graph {}'.format(graph_path))
        with gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tflow.GraphDef()
        
        graph_def.ParseFromString(f.read())

        return  graph_def


    def normalize_image(self, image):
        imagenet_stats = (np.array([0.485, 0.456, 0.406]),
                          np.array([0.229, 0.224, 0.225]))
        mean = imagenet_stats[0]
        std = imagenet_stats[1]

        for i in range(image.shape[0]):
            image[i] = (image[i] - mean[i]) / std[i]
        return image 

    def resize_image(self, image, w, h):
        image = cv2.resize(image, (w, h), cv2.INTER_LINEAR).transpose(2, 0, 1)
        return image

    def preprocess_image(self, image):
        # Converting the image into rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image 
        image = self.resize_image(image, 224, 224)
        # Normalize the image
        image = self.normalize(image)
        return image

    # def postprocess_prediction(self):

    def load_model_with_path(self, model_path):
        return 0


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        #if not self.waypoints_2d:
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoint_tree = spatial.KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
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

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        #closest_idx = self.waypoints.query(pose, 1)[1]
        #waypoint_tree = KDTree(self.waypoints, leaf_size = 2)
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # If we don't get an image we return nothing
        if(not self.has_image):
            print("AHAAA")
            self.prev_light_loc = None
            return False


        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        print(cv_image.shape)
        # image = self.preprocess_image(cv_image)

        #Get classification
        # return self.light_classifier.get_classification(cv_image)
        #print(light.state)
        # prediction = self.tf_sess.run([self.outputs1, self.outputs2],
        #                      feed_dict={self.input_image: np.expand_dims(image, axis=0)})
        # print(prediction)
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #Traffic light 
        closest_light = None
        # Traffic light line waypoint
        tl_line_wp_index = None
        print('HERE')

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position_wp = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
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
        self.waypoints = None

        # Better astop if you dont know the traffic statee

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
