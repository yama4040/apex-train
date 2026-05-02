from os import curdir
import pandas as pd
import codecs


class Track:
    def __init__(self):
        # =============================================================================
        #         停車時間
        #         停車時間個別指定           kotei_station
        #         検証時間帯停車時間個別指定 station_study
        # 　　　　　　　指定しない場合              Station
        # =============================================================================
        csv = self.__read_csv("./input/speed_limit.csv")
        self.sections = []
        for i in range(len(csv["start"])):
            #self.sections.append({"start": csv["start"][i], "speed_limit": csv["speed_limit"][i]})
            # ▼【修正】numpy.float64を純粋なfloatにキャスト
            self.sections.append({
                "start": float(csv["start"][i]), 
                "speed_limit": float(csv["speed_limit"][i])
            })

            

        # =============================================================================
        #         出発時間
        #         #遅れのみ                   departure
        # 　　　　　　　#遅れなしの場合              default
        #         #ダイヤ通りの走り              departure_Diagram
        #         #運転間隔調整除いたもの       departure_delayOnly
        # =============================================================================
        self.departure = self.__read_csv("./input/simulationcase1/departure_Diagram.csv")
        # =============================================================================
        #         到着時間
        #         #遅れのみ(時隔120)　　　　　　　arrive
        #         #理想的なダイヤ             arrive_ideal
        #         #シミュレータでの理想ダイヤ     arrive_sim_Ideal
        #         #二段減速　理想ダイヤ        arrive_sim_Ideal_2dan
        # =============================================================================
        # self.arrive     = self.__read_csv('arrive')
        csv=self.__read_csv("./input/curve.csv")
        self.curve=[]
        
        for i in range(len(csv["start"])):
            """
            if (i>0 and round(csv["end"][i-1],4)!=round(csv["start"][i],4)):
                self.curve.append({"start":csv["end"][i-1],"curve":0.0})
            self.curve.append({"start":csv["start"][i],"curve":800.0/csv["curve"][i]}
            """
            # ▼【修正】すべての数値を取り出す際にfloat()でキャスト
            if (i>0 and round(float(csv["end"][i-1]), 4) != round(float(csv["start"][i]), 4)):
                self.curve.append({"start": float(csv["end"][i-1]), "curve": 0.0})
            self.curve.append({
                "start": float(csv["start"][i]), 
                "curve": float(800.0 / csv["curve"][i])
            })
        
        
            
        csv=self.__read_csv("./input/grade.csv")
        self.grade=[]
        
        for i in range(len(csv["start"])):
            #if -40 < csv["grade"][i] <= 30:
            grade_val = csv["grade"][i]
            """
            if -40 < grade_val and grade_val <= 30:
                self.grade.append({"start":csv["start"][i],"grade":csv["grade"][i]})
            else:
                self.grade.append({"start":csv["start"][i],"grade":0.0})
            """
            # ▼【修正】キャスト＆連続比較(Chained Comparison)の分割
            grade_val = float(csv["grade"][i])
            start_val = float(csv["start"][i])
            
            if -40 < grade_val and grade_val <= 30:
                self.grade.append({"start": start_val, "grade": grade_val})
            else:
                self.grade.append({"start": start_val, "grade": 0.0})
        
        

    def __read_csv(self, path):  # inputディレクトリにあるcsvname.csvのファイルを開く
        with codecs.open(path, "r", "utf-8", "ignore") as f:
            return pd.read_csv(f)

    def get_grade_resistance(self, position):
        for i in range(len(self.grade)-1):
            #if self.grade[i]["start"] <= position <= self.grade[i+1]["start"]:
            if self.grade[i]["start"] <= position and position <= self.grade[i+1]["start"]:
                return self.grade[i]["grade"]
        return 0

    def get_curve_resistance(self, position):
        for i in range(len(self.curve)-1):
            #if self.curve[i]["start"] <= position <= self.curve[i+1]["start"]:
            if self.curve[i]["start"] <= position and position <= self.curve[i+1]["start"]:
                return self.curve[i]["curve"]
        return 0

    def get_section_id(self, position):
        for i in range(len(self.sections) - 1):
            if position < self.sections[i + 1]["start"]:
                return i
        return len(self.sections) - 1
    
    def get_curve_id(self, position):
        for i in range(len(self.curve) - 1):
            if position < self.curve[i + 1]["start"]:
                return i
        return len(self.curve) - 1
    
    def get_grade_id(self, position):
        for i in range(len(self.grade) - 1):
            if position < self.grade[i + 1]["start"]:
                return i
        return len(self.grade) - 1
    
    def get_front_sections(self, start, end):
        section_id = self.get_section_id(start)
        if end <= start:
            return []
        sections = []
        while section_id + 1 < len(self.sections) and self.sections[section_id + 1]["start"] < end:
            sections.append({"distance": self.sections[section_id + 1]["start"] - start, "speed_limit": self.sections[section_id]["speed_limit"]})
            section_id += 1
            start = self.sections[section_id]["start"]
        if end > start:
            sections.append({"distance": end - start, "speed_limit": self.sections[section_id]["speed_limit"]})
        return sections

    def get_front_curves(self, start, end):
        curve_id = self.get_curve_id(start)
        if end <= start:
            return []
        curves = []
        while curve_id + 1 < len(self.curve) and self.curve[curve_id + 1]["start"] < end:
            curves.append({"distance": self.curve[curve_id + 1]["start"] - start, "curve": self.curve[curve_id]["curve"]})
            curve_id += 1
            start = self.curve[curve_id]["start"]
        if end > start:
            curves.append({"distance": end - start, "curve": self.curve[curve_id]["curve"]})
        return curves

    def get_front_grades(self, start, end):
        grade_id = self.get_grade_id(start)
        if end <= start:
            return []
        grades = []
        while grade_id + 1 < len(self.grade) and self.grade[grade_id + 1]["start"] < end:
            grades.append({"distance": self.grade[grade_id + 1]["start"] - start, "grade": self.grade[grade_id]["grade"]})
            grade_id += 1
            start = self.grade[grade_id]["start"]
        if end > start:
            grades.append({"distance": end - start, "grade": self.grade[grade_id]["grade"]})
        return grades
