from styx_msgs.msg import TrafficLight
from PIL  import Image
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

DEBUG_MODE = True
class TLClassifier(object):
    def __init__(self, is_site):
        '''Initializes the classifier variables and gets it ready for inception

            Args:
            is_site: boolean to specify if the system is used either on site or in simulation
        '''

        self.imagenet_stats = (np.array([0.485, 0.456, 0.406]),
                          np.array([0.229, 0.224, 0.225]))
        # Init Post-processing analyzer 
        self.analyzer = BBoxAnalyzer()
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
        sim_model_name = "OLD-sim-FL-2400.pb"
        site_model_name = "OLD-rl-bosch-FL-600.pb"
        # Model path, by default its going to be the sim model path
        model_path = models_path + sim_model_name 
        

        
        '''Specifies different model and a different threshold for the 
           site and simulation

        '''
        if is_site:
            # On Site
            # Switch to the model site model path
            model_path = models_path + site_model_name
            # Set model Threshold
            self.threshold = 0.4
            # print("\n==========On SITE==========\n")
        else:
            # In Simulattion 
            self.classes = self.classes[:3]
            # Set model Threshold
            self.threshold = 0.3
            # print("\n==========In SIM==========\n")
        # print("\n==========Loading model==========\n")
        # Init tensorflow Graph (Model) 
        self.tf_graph = self.load_tf_graph(model_path)
        # Remove Graph node attrs to accommodate earlier versions of TF
        self.fix_graph()
        # Init Tensorflow's Session
        self.tf_sess = tf.InteractiveSession() 
        # Set session's graph (load model into Session)
        self.tf_sess.graph.as_default()
        tf.import_graph_def(self.tf_graph, name='')
        # Getting model input and output nodes
        self.input_image = self.tf_sess.graph.get_tensor_by_name("0:0")
        self.outputs1 = self.tf_sess.graph.get_tensor_by_name("concat_52:0")
        self.outputs2 = self.tf_sess.graph.get_tensor_by_name("concat_53:0")
        print("\n==========Successfully loaded the model \\0/==========\n")

    def fix_graph(self):
        # fix nodes
        for node in self.tf_graph.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            if "dilations" in node.attr:
                del node.attr["dilations"]
            if "index_type" in node.attr:
                del node.attr["index_type"]
            if "Truncate" in node.attr:
                del node.attr["Truncate"]
            if node.op == 'Where':
                if "T" in node.attr:
                    del node.attr["T"]
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Preprocess image
        preprocessed_image = self.preprocess_image(image)
        # Predict light
        preds = self.tf_sess.run([self.outputs1, self.outputs2],
                             feed_dict={self.input_image: np.expand_dims(preprocessed_image, axis=0)})
        # Post process predictions and get light color 
        light = self.get_light(preds)

        return light
    
    def load_tf_graph(self, graph_path):
        '''Loads the tensorflow grah form the .pb file.

            Args:
            graph_path: path to the .pb file 

            Returns:
            tensorflow graph object
        '''

        # print('load graph {}'.format(graph_path))
        with gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
        
        graph_def.ParseFromString(f.read())

        return  graph_def

    def normalize_image(self, image):
        '''Normaliz image for inception

            Args:
            image: numpy array image

            Returns:
            numpy image: noirmalized image ready to be fed to the model
        '''
        mean = self.imagenet_stats[0]
        std = self.imagenet_stats[1]
        # Normalize the image against the imagenet stats because the model 
        # did the same in training
        for i in range(image.shape[0]):
            image[i] = (image[i] - mean[i]) / std[i]
        return image 


    def preprocess_image(self, cv_image):
        '''Postprocess the predictions and returns 
           format.

            Args:
            perds (RetinaNet Probabilities): predictions to process

            Returns:
            [[bboxes],[class_ids]]: traffic light bboxes and their corresponding class ids
        '''        
        # Convert the image into PIL
        pil_image = Image.fromarray(cv_image)
        # Resize image 
        pil_image = pil_image.resize((224, 224), resample=Image.BILINEAR)
        # From PIL to numpy array
        # Convert image to channel, w, h 
        pil_image = np.asarray(pil_image).transpose(2, 0, 1)
        # Limit image range from [0,1]
        pil_image = np.multiply(pil_image, 1.0 /255.0)
        # Normalize image with imagenet stats
        image = self.normalize_image(pil_image)
        return image


    def postprocess_prediction(self, preds):
        '''Postprocess the predictions and returns 
           format.

            Args:
            perds (RetinaNet Probabilities): predictions to process

            Returns:
            [[bboxes],[class_ids]]: traffic light bboxes and their corresponding class ids
        '''
        return self.analyzer.analyze_pred((preds[0][0], preds[1][0]), thresh=self.threshold)
    

    def get_light(self, preds):
        '''Processes the predictions and returns the light_state in styx_msgs/TrafficLight
           format.

            Args:
            perds (all classes probabilities, all bboxes probabilities): predictions to process

            Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        '''
        # Postprocess predictions
        postprocessed_prediction = self.postprocess_prediction(preds)
        if postprocessed_prediction is None:
            # print("Nothing")
            return TrafficLight.UNKNOWN
        # Get light class
        predicted_traffic_light = self.classes[postprocessed_prediction[1][0][0] - 1]
        
        # Default it as red for safety
        light_state = TrafficLight.RED
        
        if predicted_traffic_light == 'Green':
            light_state = TrafficLight.GREEN
        elif predicted_traffic_light == 'Yellow':
            light_state = TrafficLight.YELLOW
        else : 
            light_state = TrafficLight.RED
        # print(predicted_traffic_light)
        return light_state

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

classifier = TLClassifier(False)
# image = cv2.imread('left0017v.jpg')
image = cv2.imread('test_image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i in range(10):
    print(classifier.get_classification(image))