
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from lowpass import LowPassFilter
from pid import PID

class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, decel_limit, accel_limit, wheel_radius):
        # TODO: Implement
        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.decel_limit_Nm = decel_limit * vehicle_mass * wheel_radius
        self.accel_limit_Nm = accel_limit * vehicle_mass * wheel_radius
        
        kp_trq = 10
        ki_trq = 1.0
        kd_trq = 0.0
        self.throt = 0
        self.brake = 0

        #Low Pass Filter object
        #controller rate is 50Hz -> 0.02 Ts
        #cut off the driver at 10Hz -> 0.1 Tau
        self.low_pass_filter = LowPassFilter(0.1, 0.02)
        self.trq_pid   = PID(kp_trq,   ki_trq,   kd_trq,   self.decel_limit_Nm, self.accel_limit_Nm)

    def linear_control(self, dbw_enabled, veh_spd_cmd, veh_spd_act):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        if dbw_enabled:
            veh_spd_act_filt = self.low_pass_filter.filt(veh_spd_act)
            veh_spd_err = veh_spd_cmd - veh_spd_act_filt
            
            veh_trq_req = self.trq_pid.step(veh_spd_err, 0.02)
            if veh_trq_req <= 0:
                self.brake = -veh_trq_req
                self.throt = 0
            else:
                self.brake = 0 
                self.throt = veh_trq_req / self.accel_limit_Nm
        else:
            self.trq_pid.reset()
        return self.throt, self.brake
