#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import sys
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        """ Init subsriber/publishers and member variables """
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Other member variables 
        self.max_velocity = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))
        self.pos = None
        self.waypoints = None  

        rospy.spin()

    def pose_cb(self, msg):
        """ Callback for a position update """
        self.pos = msg.pose 
        if self.waypoints is not None:                
            self.publish()

    def waypoints_cb(self, msg):
        """ 
        Callback for an update to the list of waypoints 
        (should only happen on first read) 
        """ 
        if self.waypoints is None:
            self.waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass       
    
    def get_waypoint_velocity(self, waypoint):
        """ Get the linear velocity at a given waypoint """ 
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        """ Set the linear velocity at a given waypoint """ 
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        """ Find distance between 2 waypoints """
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # copied from waypoint_loader.py
    def distance_euclid(self, p1, p2):
        """ Find distance between 2 positions """
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)  
    
   # copied from waypoint_loader.py
    def kmph2mps(self, velocity_kmph):
        """ Convert kilometers per hour to meters per second """
        return (velocity_kmph * 1000.) / (60. * 60.)

    def get_next_waypoint(self, pos, waypoints):
        """ Find the next waypoint ahead of the given position """
        closest_distance = sys.float_info.max
        next_waypoint = 0
        for i, wp in enumerate(self.waypoints):
            distance = self.distance_euclid(pos.position, wp.pose.pose.position)
            if (distance < closest_distance):
                closest_distance = distance
                next_waypoint = i   
                
        # Make sure we aren't looking at a wp behind us
        angle_of_wp = math.atan2((waypoints[next_waypoint].pose.pose.position.y - pos.position.y), 
                             (waypoints[next_waypoint].pose.pose.position.x - pos.position.x))
        _, _, yaw = tf.transformations.euler_from_quaternion((pos.orientation.x, 
                                                              pos.orientation.y, 
                                                              pos.orientation.z, 
                                                              pos.orientation.w))

        
        # If the closest wp we found was behind us, chose the next one beyond that
        if abs(yaw - angle_of_wp) > (math.pi / 4):
            next_waypoint += 1
        
        return next_waypoint
    
    def publish(self): 
        """ Publish the updated list of lookahead waypoints """
        if self.pos is not None:
            next_waypoint = self.get_next_waypoint(self.pos, self.waypoints)
            updated_waypoints = self.waypoints[next_waypoint:next_waypoint+LOOKAHEAD_WPS]
            
            for i in range(len(updated_waypoints) - 1):
                 self.set_waypoint_velocity(updated_waypoints, i, self.max_velocity)

            # copy Lane() setup from waypoint_loader.py 
            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time(0)
            
            lane.waypoints = updated_waypoints
            
            """ 
            # Debug 
            rospy.logerr("Next waypoint=%d, x=%d, y=%d", 
                         next_waypoint,
                         self.waypoints[next_waypoint].pose.pose.position.x,
                         self.waypoints[next_waypoint].pose.pose.position.y)
            rospy.logerr("Curr pos= x=%d, y=%d", 
                         self.pos.position.x, 
                         self.pos.position.y)
            """
            
            self.final_waypoints_pub.publish(lane)
            
if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
