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
import yaml
import numpy as np
from scipy import spatial
import PIL 

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
        # Init analyzer 
        self.analyzer = BBoxAnalyzer()

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
        image = self.normalize_image(image)
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
            self.prev_light_loc = None
            return False


        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        pil_image = PIL.Image.fromarray(cv_image)
        pil_image = pil_image.resize((224, 224), resample=PIL.Image.BILINEAR)

        pil_image = np.asarray(pil_image).transpose(2, 0, 1)
        pil_image = np.multiply(pil_image, 1.0 /255.0)
        image = self.normalize_image(pil_image)
        # image = self.preprocess_image(pil_image)

        # cv2.imwrite('messigray.png',cv_image)
        #Get classification
        #print(light.state)

        preds = self.tf_sess.run([self.outputs1, self.outputs2],
                             feed_dict={self.input_image: np.expand_dims(image, axis=0)})
        # print(preds[1])
        # np.savez('tensor', preds_0=preds[0][0], preds_1=preds[1][0])
        
        
        outputs = self.analyzer.analyze_pred((preds[0][0], preds[1][0]), thresh=0.3)
        # print(len(outputs))
        if outputs is None:
            # print('No prediction')
            return TrafficLight.UNKNOWN
        # print("Classes {}".format(outputs[1]))

        # print(np.take(self.classes, outputs[1]))
        predicted_traffic_light = self.classes[outputs[1][0][0] - 1]
        # print(predicted_traffic_light)
        # Default it as red for safety
        light_state = TrafficLight.RED
        
        if predicted_traffic_light == 'Green':
            light_state = TrafficLight.GREEN
        elif predicted_traffic_light == 'Yellow':
            light_state = TrafficLight.YELLOW
        else : 
            light_state = TrafficLight.RED


        return light_state#light.state

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



class BBoxAnalyzer(object):
    def __init__(self,grids=[4, 2, 1], zooms=[0.7, 1., 1.3],ratios=[[1., 1.], [1., 0.5], [0.5, 1.]], bias=-4.):
        super(BBoxAnalyzer, self).__init__()
        self._create_anchors(grids, zooms, ratios)
        
    def _create_anchors(self, anc_grids, anc_zooms, anc_ratios):    
        self.grids = anc_grids
        self.zooms = anc_zooms
        self.ratios =  anc_ratios
        anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        self._anchors_per_cell = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), self._anchors_per_cell, axis=0)

        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])
                       
        self._grid_sizes = np.expand_dims(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids]), 1)
        self._anchors = np.concatenate([anc_ctrs, anc_sizes], axis=1)
        self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])
	
    def _hw2corners(self, ctr, hw): 
        return np.concatenate([ctr-hw/2, ctr+hw/2], axis=1)

    def sigmoid(self, x):
        return  1/(1+np.exp(-x))

    def analyze_pred(self, pred, thresh=0.5, nms_overlap=0.1):
        # print('Heeeey, Im heeerere')
        # def analyze_pred(pred, anchors, grid_sizes, thresh=0.5, nms_overlap=0.1, ssd=None):
        b_clas, b_bb = pred
        a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
        conf_scores = b_clas[:, 1:].max(1)
        conf_scores = self.sigmoid(b_clas.transpose())
        # pdb.set_trace()

        out1, bbox_list, class_list = [], [], []
        # pdb.set_trace()
        for cl in range(1, len(conf_scores)):
            c_mask = conf_scores[cl] > thresh
            if c_mask.sum() == 0: 
                # print("Skipped{}".format(c_mask.sum()))
                continue
            scores = conf_scores[cl][c_mask]
            l_mask = np.expand_dims(c_mask, 1)
            l_mask = np.broadcast_to(l_mask, a_ic.shape)
            boxes = a_ic[l_mask].reshape((-1, 4)) # boxes are now in range[ 0, 1]
            boxes = (boxes-0.5) * 2.0        # putting boxes in range[-1, 1]
            ids, count = self.nms(boxes, scores, nms_overlap, 50) # FIX- NMS overlap hardcoded
            ids = ids[:count]
            out1.append(scores[ids])
            # pdb.set_trace()
            bbox_list.append(boxes[ids])
            class_list.append([cl]*count)
        # pdb.set_trace()
        if len(bbox_list) == 0:
            return None #torch.Tensor(size=(0,4)), torch.Tensor()
        return bbox_list, class_list
    
    def _actn_to_bb(self, actn, anchors, grid_sizes):
        actn_bbs = np.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
        return self._hw2corners(actn_centers, actn_hw)


    def nms(self, boxes, scores, overlap=0.5, top_k=100):
        keep = np.zeros_like(scores, dtype=np.long)
        if np.size(boxes) == 0: return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = np.multiply(x2 - x1, y2 - y1)
        idx = np.argsort(scores, axis=0)  # sort in ascending order
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = np.array([], dtype=boxes.dtype)
        yy1 = np.array([], dtype=boxes.dtype)
        xx2 = np.array([], dtype=boxes.dtype)
        yy2 = np.array([], dtype=boxes.dtype)
        w = np.array([], dtype=boxes.dtype)
        h = np.array([], dtype=boxes.dtype)
        count = 0
        while np.size(idx) > 0:
            i = idx[-1]  # index of current largest val
            keep[count] = i
            count += 1
            if idx.shape[0] == 1: break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            xx1 = np.take(x1, idx)
            yy1 = np.take(y1, idx)
            xx2 = np.take(x2, idx)
            yy2 = np.take(y2, idx)
            # store element-wise max with next highest score
            xx1 = np.clip(xx1, x1[i], None)
            yy1 = np.clip(yy1, y1[i], None)
            xx2 = np.clip(xx2, None, x2[i])
            yy2 = np.clip(yy2, None, y2[i])
            
            w = np.resize(w, xx2.shape)
            h = np.resize(w, yy2.shape)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = np.clip(w, 0.0, None)
            h = np.clip(h, 0.0, None)
            inter = w*h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = np.take(area, idx)  # load remaining areas)
            # pdb.set_trace()
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[np.less_equal(IoU, overlap)]
        return keep, count

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
