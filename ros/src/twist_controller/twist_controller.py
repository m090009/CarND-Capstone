
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController
class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, decel_limit, accel_limit, wheel_radius,wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.decel_limit_Nm = decel_limit * vehicle_mass * wheel_radius
        self.accel_limit_Nm = accel_limit * vehicle_mass * wheel_radius
        
        kp_trq = 100
        ki_trq = 1.0
        kd_trq = 0.0
        self.throt = 0
        self.brake = 0
        self.steering = 0 

        #Low Pass Filter object
        #controller rate is 50Hz -> 0.02 Ts
        #cut off the driver at 10Hz -> 0.1 Tau
        self.low_pass_filter = LowPassFilter(0.1, 0.02)
        self.trq_pid   = PID(kp_trq,   ki_trq,   kd_trq,   self.decel_limit_Nm, self.accel_limit_Nm)
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

    def control(self, dbw_enabled, veh_spd_cmd, veh_spd_act, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        if dbw_enabled:
            veh_spd_act_filt = self.low_pass_filter.filt(veh_spd_act)
            veh_spd_err = veh_spd_cmd - veh_spd_act_filt
            #print('veh_spd_err = ', veh_spd_err, 'veh_spd_cmd = ', veh_spd_cmd)
            veh_trq_req = self.trq_pid.step(veh_spd_err, 0.02)
            if veh_trq_req <= 0:
                
                if(veh_spd_cmd < 0.1):
                    self.brake = 1000 #per udacity suggestion
                    self.throt = 0.0
                    self.trq_pid.reset()
                else:
                    self.throt = 0.0
                    self.brake = -veh_trq_req
            else:
                self.brake = 0 
                self.throt = veh_trq_req / self.accel_limit_Nm

            self.steering = self.yaw_controller.get_steering(veh_spd_cmd, angular_vel, veh_spd_act_filt) 
            return self.throt, self.brake, self.steering
        
        else:
            self.trq_pid.reset()
            return 0.0, 0.0, 0.0
        
