import tkinter as tk
from tkinter import ttk
from collections import Counter
from webcam import *
from emotionRecognizer import EmotionRecognizer
from PIL import Image, ImageTk

import matplotlib.pyplot as plt
from collections import Counter

import torch
import clothes_recommend as er

class UI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.cam =Webcam()
        self.ER = EmotionRecognizer(r"data\best_model2.pth")
        #self.configure()
        
        self.img = None
        self.width = 1280
        self.height = 720
        
        self.geometry(f"{self.width}x{self.height}")
        
        self.title("Emotion Detection UI")
        self.resizable(False, False)
        #rect(60, 68, 739, 642) 
        self.insertUI()

        self.is_recognizing = False
        self.image_on_canvas  = None
        self.emotion = "준비중"


        self.elapsed_time = 0.0

        self.update()
    def update(self):

        self.cam.update()
        self.img = self.cam.getImg_tk()
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        
        self.UIupdate()
        
        self.after(1, self.update)

    def insertUI(self):
        # 카메라 프레임 캔버스
        self.canvas  = tk.Canvas(self, width = 720, height = 580, bg = "white")
        self.cam.setSize(720, 580)
        self.canvas.place(x = 60, y = 60)

        btn = tk.Button(self, text = "감정 인식",
                        command=self.start_recognize,
                        font=("맑은 고딕", 20))
        btn.place(x = 1080, y = 570, width = 150, height = 60)
        self.btn = btn


        #실시간 감정 인식 데이터 라벨
        self.emotion_labels: list[tk.Label] = []
        y_positions = [100, 180, 260, 340]
        for y in y_positions:
            label = tk.Label(self, text="감정 인식 준비중", font=("맑은 고딕", 20), anchor="nw")
            label.place(x=820, y=y, width=400, height=60)
            self.emotion_labels.append(label)

        #감정인식 결과 라벨
        label = tk.Label(self, text="감정 인식 준비중", font=("맑은 고딕", 22), anchor="nw")
        label.place(x = 820, y = 450, width = 400, height = 60)
        self.config_label = label    

        #실시간 감정인식 데이터 바
        self.emotion_bars: list[ttk.Progressbar] = []
        for y in y_positions:
            bar = ttk.Progressbar(self, length=200, maximum=100)
            bar.place(x=1000, y=y+50, height=30)  # 라벨과 수직 정렬 보정
            self.emotion_bars.append(bar)
        
        #감정인식 결과 보기 버튼
        btn = tk.Button(self, text = "결과 분석",
                        command=self.show_pie_chart,
                        font=("맑은 고딕", 18))
        btn.place(x = 820, y = 500, width = 120, height = 40, anchor="nw")
        btn.place_forget()
        self.rst_btn = btn
    def UIupdate(self):
        #print(self.is_recognizing)
        
        if self.is_recognizing:

            if self.timer():
                #인식 완료
                self.is_recognizing = False
                self.btn.config(state="normal")
                #최빈값
                if self.detected_emotions:
                    self.emotion = Counter(self.detected_emotions).most_common(1)[0][0]
                    

                    print(f"\n 감정 인식 결과: '{self.emotion}'")
                    self.rst_btn.place(x = 820, y = 500)
                    
                else:
                    #인식된 감정이 없을 떄(인식률이 확률이 65 %넘는 감정이 한개도 없을 때)
                    print("\n 감정 데이터를 수집하지 못했습니다.")
                    self.emotion = "인식 불가"
    
            else:
                class_names = self.ER.getClassNames()
                emotion, probs = self.detect_emotion()


                #최빈값 추출을 위한 코드
                pred_idx = torch.argmax(probs).item()
                pred_label = class_names[pred_idx]

                if probs[pred_idx] >= 0.65:
                    self.detected_emotions.append(pred_label)

                #self.detected_emotions.append(pred_label)

                self.config_label.config(text=f"현재 감정: {emotion}")

                for i, prob in enumerate(probs):
                    percent = prob.item() * 100
                    self.emotion_labels[i].config(text=f"{class_names[i]}: {percent:5.1f}%")
                    self.emotion_bars[i]['value'] = percent
                
                self.btn.config(text = f"인식중{self.elapsed_time:.1f}")
        else:
            self.btn.config(text = f"감정 인식")
            self.config_label.config(text=f"인식 결과: {self.emotion}")


    def show_pie_chart(self):
        counts = Counter(self.detected_emotions)
        labels = list(counts.keys())
        sizes = list(counts.values())
        er.recommend_clothes(self.emotion)
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("recognized emotions ratio")
        plt.axis('equal')  # 원형 유지
        plt.show()
        
    def timer(self):
        dt = self.cam.getDT()
        self.elapsed_time += dt
        if self.elapsed_time > 5.0:
            return True
        
        return False

    def start_recognize(self):
        self.btn.config(state="disabled")
        self.is_recognizing = True
        self.elapsed_time = 0.0
        self.detected_emotions = []
        self.rst_btn.place_forget()
    def detect_emotion(self):
        
        return self.ER.recognize_emotion(self.cam.getFrame())

if __name__ == '__main__':
    ui = UI()
    ui.mainloop()