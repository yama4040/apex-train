from actions import Actions
from track import Track


class Train(object):
    def __init__(self, target_station, position=0, speed=0,weight_correction=1.0):
        self.time_step_base=0.01
        self.DECELERATE = -2.4/10*360
        self.TRAIN_WEIGHT = 28  # 列車重量
        #self.FACTOR_OF_INERTIA = 28.34391294  # 慣性係数]
        self.FACTOR_OF_INERTIA = 28.34467  # 慣性係数]
        self.TARGET_STATION = target_station
        self.WEIGTH_CORRECTION=weight_correction
        self.__speed = speed
        self.__position = position
        self.__pre_acceleration=0
        self.track = Track()

    def set_states(self, speed, position):
        self.__speed = speed
        self.__position = position

    def step(self, action, time_step):
        for i in range(int(time_step/self.time_step_base)):
            if self.position < 0:
                return
            force=0
            if (action==Actions.acceleration): force = self.tractive_force
            accel = ((((force - self.travel_resistance)*self.WEIGTH_CORRECTION)-(self.grade_resistance+self.curve_resistance))/self.FACTOR_OF_INERTIA)
            res = self.travel_resistance+self.grade_resistance+self.curve_resistance
            base_accel = ((force - res)/self.FACTOR_OF_INERTIA)
            if (action==Actions.deceleration): 
                accel+=self.DECELERATE*self.time_step_base*self.WEIGTH_CORRECTION
                base_accel+=self.DECELERATE*self.time_step_base
            if self.speed + accel * self.time_step_base >= 0:
                self.__position += (self.__speed / 3600) * self.time_step_base + 1 * (accel /3600) * (self.time_step_base**2)
                self.__speed += accel * self.time_step_base
            else:
                self.__speed = 0
        self.__pre_acceleration=accel-base_accel
    
    # 引張力[kg/t]
    @property
    def tractive_force(self):
        if 0<=self.__speed<42:
            return -1.489*self.__speed+92.408
        elif 42<=self.__speed<68:
            return -0.4*self.__speed+46.68
        else:
            return -0.0963*self.__speed+26.0284
    
    # 走行抵抗[kg/t]
    @property
    def travel_resistance(self):
        return 2.39+0.0224*self.__speed+0.00062*(self.__speed**2)
    
    # 勾配抵抗[kg/t]
    @property
    def grade_resistance(self):
        return self.track.get_grade_resistance(self.position)
    
    # 曲線抵抗[kg/t]
    @property
    def curve_resistance(self):
        return self.track.get_curve_resistance(self.position)
    
    @property
    def speed(self):
        return self.__speed

    @property
    def motor_acceleration(self):
        a = self.tractive_force / self.FACTOR_OF_INERTIA
        return a

    @property
    def pre_acceleration(self):
        return self.__pre_acceleration

    @property
    def position(self):
        return self.__position

    @property
    def current_speed_limit(self):
        if len(self.front_sections) == 0:
            return 0
        return self.front_sections[0]["speed_limit"]

    @property
    def section_remaining_distance(self):
        if len(self.front_sections) == 0:
            return 0
        return self.front_sections[0]["distance"]

    @property
    def front_sections(self):
        return self.track.get_front_sections(self.position, self.TARGET_STATION)

    @property
    def front_curves(self):
        return self.track.get_front_curves(self.position, self.TARGET_STATION)
    
    @property
    def front_grades(self):
        return self.track.get_front_grades(self.position, self.TARGET_STATION)
    
    @property
    def resistance(self):
        return self.__get_resistance() + self.__get_run_resistance()
