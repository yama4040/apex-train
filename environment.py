from train import Train
from actions import Actions
import numpy as np
import random
import pandas as pd
import codecs
import math

class Environment:
    def __init__(self, time_step=1.0):
        self.__time_step = time_step
        self.MAX_SECTIONS = 6
        self.MAX_CURVES=4
        self.MAX_GRADES=8
        stations_csv = self.read_csv("./input/Station.csv")
        self.stations = []
        for i in range(len(stations_csv)):
            self.stations.append({"position": stations_csv["position"][i], "running_time": stations_csv["rt"][i]})

    #環境のリセット関数。出発駅のインデックス、遅延時間、重量補正値、前方列車の位置オフセット、出発位置のオフセット、前方列車の制御データ（CSVファイル）を引数に取る。これらの引数はエピソードの初期化方法をランダムに選択するために使用される。
    def reset(self, departure_index=None, delay=0.0, weight_correction=1.0, fowerd_train_position_offset=None, start_position_offset=0.0, fowerd_train_controls=None):
        #出発駅のインデックスが指定されていない場合は、ランダムに選択する
        if departure_index is None:
            departure_index = random.randrange(1)
        self.t = 0.0
        self.departure_station = self.stations[departure_index]
        #self.departure_station["running_time"]+=delay
        self.arrival_station = self.stations[departure_index + 1]
        start_position=self.departure_station["position"]+start_position_offset
        self.train = Train(self.arrival_station["position"], start_position, 0.0,weight_correction)
        self.fowerd_train=None
        
        #前方列車の制御データが指定されている場合は、CSVファイルから制御データを読み込み、前方列車の状態を初期化する
        if (fowerd_train_controls): 
            ftc=self.read_csv(fowerd_train_controls)
            self.fowerd_train_controls=[]
            for i in range(len(ftc)):
                action=Actions.deceleration
                if (ftc["action"][i]=="Actions.coasting"): action=Actions.coasting
                elif (ftc["action"][i]=="Actions.acceleration"): action=Actions.acceleration
                elif (ftc["action"][i]=="Actions.deceleration"): action=Actions.deceleration
                self.fowerd_train_controls.append({"time":i,"position":ftc["position"][i],"speed":ftc["speed"][i], "action": action})
            self.fowerd_train=Train(self.arrival_station["position"], self.fowerd_train_controls[0]["position"],self.fowerd_train_controls[0]["speed"],1.0)
        self.fowerd_train_position_offset=fowerd_train_position_offset
        
        #前方列車の位置オフセットが指定されている場合、基準運転曲線をCSVファイルから読み込み、前方列車の位置に応じた基準運転時間を設定する。
        #出発位置のオフセットが指定されている場合、基準運転時間に遅延時間を加算する。
        if (self.fowerd_train_position is not None or start_position_offset !=0.0):
            sr_csv = self.read_csv(f"./input/sr_{departure_index}.csv")
            self.standerd_running = []
            for i in range(len(sr_csv)):
                self.standerd_running.append({"position": sr_csv["position"][i], "time": sr_csv["time"][i]})
        if (start_position_offset!=0.0):
            for i in range(len(self.standerd_running)):
                self.t=self.standerd_running[i]["time"]+delay
                break
        self.pre_action = Actions.deceleration  #前の行動を減速に初期化
        self.holding_time = 30  #前の行動を保持している時間を30秒に初期化（これにより、最初の行動が減速であるため、加速がすぐに禁止されないようにする）
        return self.normalized_state    #初期状態を正規化して返す

    #環境のステップ関数。エージェントの行動を引数に取り、次の状態、報酬、エピソードの終了フラグを返す。
    def step(self, action):
        if (self.fowerd_train is not None and self.fowerd_train.speed<=0.0 and self.fowerd_train.position>self.arrival_station["position"]): self.fowerd_train=None
        if (self.fowerd_train is not None and self.t>self.fowerd_train_controls[-1]["time"]): self.fowerd_train.step(Actions.deceleration,1)
        elif (self.fowerd_train is not None and round(self.t%1,1)==0): self.fowerd_train.step(self.fowerd_train_controls[int(self.t)]["action"],1)
        action = Actions(action)
        done = False
        tve = 0.0
        tce=0.0
        de=0.0
        ic = 0.0
        ef = 0.0
        sl = 0.0
        
        #前の行動と現在の行動が異なり、かつ前の行動を7秒未満保持している場合、インスタントペナルティを計算する。加速と減速の切り替えの場合は、ペナルティが2倍になる。
        if self.pre_action != action and self.holding_time < 7.0:
            ic = max((10.0 - self.holding_time) / 7.0, 0.0) #保持時間が短いほどペナルティが大きくなる．
            #加速→減速，減速→加速の場合はペナルティ2倍
            if (self.pre_action == Actions.acceleration and action == Actions.deceleration) or (
                self.pre_action == Actions.deceleration and action == Actions.acceleration
            ):
                ic *= 2.0
        time_step=self.time_step
        self.train.step(action,time_step)
        self.t += time_step
        if self.pre_action == action:
            self.holding_time += time_step
        else:
            self.holding_time = time_step
        self.pre_action = action
        
        #制限速度超過のペナルティ
        if self.speed > self.current_speed_limit and self.position < self.arrival_station["position"]:
            sl = 1.0
        #到着駅を通過している場合のペナルティ
        if self.speed>0.0 and self.position>self.arrival_station["position"]+0.005:
            sl=1.0
        #前方列車との衝突リスクのペナルティ
        if self.fowerd_train_position is not None and self.position > self.fowerd_train_position:
            sl=1.0
        #先行列車がいない場合
        if self.fowerd_train_position is None:
            #現在時間が出発駅の基準運転時間を超えている場合、到着駅までの距離に応じたペナルティを計算する。距離が近いほどペナルティが小さくなる．
            if self.t > self.departure_station["running_time"]:
                tve = abs(self.arrival_station["position"] - self.position)**(1.0/3.0)
                tce=1
                tve*=time_step
                tce*=time_step
        else:
            #先行列車に追従するための規定時間を過ぎてしまった場合ペナルティ
            if self.t > self.fixed_running_time:
                tve = abs(self.fowerd_train_position - self.position)**(1.0/3.0)
                tce=1
                tve*=time_step
                tce*=time_step
        
        #到着駅に近づいている場合の報酬
        if self.position>=self.arrival_station["position"]-0.01 and self.speed<=0.0:
            de=(max(min((abs(self.arrival_station["position"] - self.position))*1000,10),0.1)**(-0.5))-0.31623
        #電力の計算
        if action == Actions.acceleration:
            ef = self.motor_acceleration * self.speed*time_step
            
        #報酬の計算。速度超過、到着駅通過、前方列車との衝突リスクに対するペナルティ、遅延時間に対するペナルティ、インスタントペナルティ、電力消費に対するペナルティ、到着駅への接近に対する報酬を組み合わせて計算する。
        reward = -30.0 * sl - 0.5 * tve-15.0*tce+20*de - 100.0 * ic - 0.02 * ef
        
        #60秒以上遅延した場合，10m以内に停車，先行列車の手前で停止できた（100メートル手前）場合，エピソード終了
        if self.t >= self.departure_station["running_time"] + 60.0:
            done = True
        if self.speed<=0.0 and self.position>=self.arrival_station["position"]-0.01:
            done=True
        if self.fowerd_train_position is not None and self.speed<=0 and self.position>=self.fowerd_train_position-0.1:
            done=True

        return self.normalized_state, reward/30, done

    @property
    def normalized_state(self):
        pre_action_c = 1.0 if self.pre_action == Actions.coasting else 0
        pre_action_a = 1.0 if self.pre_action == Actions.acceleration else 0
        pre_action_d = 1.0 if self.pre_action == Actions.deceleration else 0
        return [
            self.speed / 80.0,
            #*front_sections,
            #*front_curves,
            #*front_grades,
            (max(self.station_remaining_distance,-0.5)+0.5)/2,
            (max(min(self.station_remaining_distance,0.2),-0.05)+0.05)*4,
            self.remaining_time / 360.0,
            min(self.holding_time,30) / 30.0,
            #(self.train.pre_acceleration+0.7)/1.4,
            pre_action_c,
            pre_action_a,
            pre_action_d,
            (max(self.fowerd_train_remaining_distance,-0.5)+0.5)/2,
        ]

    @property
    def raw_state(self):

        return [
            self.speed,
            self.station_remaining_distance,
            self.remaining_time,
            self.holding_time,
            #self.train.pre_acceleration,
            self.pre_action,
            self.station_remaining_distance,
            self.fowerd_train_remaining_distance
        ]

    @property
    #禁止行動を表すベクトルを返すプロパティ。加速、惰行、減速の各行動が禁止されているかどうかを判定し、
    # それぞれの行動に対してTrue（禁止）またはFalse（許可）を返す。
    def forbidden_action(self):
        acceleration = False
        coasting = False
        deceleration = False
        """
        #速度超過の場合、加速と惰行が禁止される
        if self.speed > self.current_speed_limit:
            acceleration = True
            coasting = True
        #前方列車が存在し、かつ自分の位置が前方列車の位置を超えている場合、加速と惰行が禁止される
        if self.speed + self.train.motor_acceleration > self.current_speed_limit:
            acceleration = True
        #次の区間の速度制限が現在の速度制限よりも大幅に低い場合、加速と惰行が禁止される
        if self.speed**2.0 > (self.next_speed_limit**2.0) * 0.9 - 2.0 * self.train.DECELERATE * 3600 * self.section_remaining_distance:
            acceleration = True
            coasting = True
        #減速が禁止される条件は、前の行動が減速であり、かつその行動を30秒以上保持している場合
        if self.holding_time == self.time_step and self.pre_action == Actions.deceleration:
            acceleration = True
        """
        #速度超過の場合、加速と惰行が禁止される
        if self.speed > self.current_speed_limit:
            acceleration = True
            coasting = True
        #前方列車が存在し、かつ自分の位置が前方列車の位置を超えている場合、加速と惰行が禁止される
        if self.fowerd_train_position is not None and self.position > self.fowerd_train_position:
            acceleration = True
            coasting = True
        return np.array([coasting, acceleration, deceleration])

    @property
    def station_remaining_distance(self):
        return self.arrival_station["position"] - self.position

    @property
    def fowerd_train_position(self):
        if (self.fowerd_train_position_offset is None and self.fowerd_train is None): return None
        if (self.fowerd_train is None): return self.departure_station["position"]+self.fowerd_train_position_offset
        else : return self.fowerd_train.position

    @property
    def fowerd_train_remaining_distance(self):
        if (self.fowerd_train_position is None): return self.station_remaining_distance
        return self.fowerd_train_position-self.position
    
    @property
    def remaining_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"] - self.t
        return self.fixed_running_time - self.t
    
    @property
    def fixed_running_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"]
        for i in range(len(self.standerd_running)):
            if (self.fowerd_train_position<self.standerd_running[i]["position"]):
                return self.standerd_running[i]["time"]
        return self.departure_station["running_time"]

    @property
    def current_speed_limit(self):
        return self.train.current_speed_limit

    @property
    def speed(self):
        return self.train.speed

    @property
    def position(self):
        return self.train.position

    @property
    def motor_acceleration(self):
        return self.train.motor_acceleration
    
    @property
    def time_step(self):
        if self.position<self.arrival_station["position"]-0.1:
            return self.__time_step
        else:
            return self.__time_step*0.1

    def read_csv(self, path):
        with codecs.open(path, "r", "utf-8", "ignore") as f:
            return pd.read_csv(f)
