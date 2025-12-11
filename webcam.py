import cv2
import time
import numpy as np
import os
from PIL import Image, ImageTk
class Webcam:
	def __init__(self):
		cap = cv2.VideoCapture(0)

		self.cam = cap
		time.sleep(2)
		self.pt = time.time()
		self.ct = 0
		self.dt = 0
		self.size = (640, 480)
	def __str__(self):
		return f"webcam"
	def update(self):

		self.ct = time.time()
		ret, self.frame = self.cam.read()
		
		if not ret:
			print("E: No Cam")
			return
		#self.preprocess()
		self.frame = cv2.resize(self.frame, self.size)
		self.frame = cv2.flip(self.frame, 1)
		self.dt =  self.ct - self.pt
		self.pt = self.ct

	def saveFrame(self, path):
		ret, buf = cv2.imencode('.jpg', self.frame)
		if not ret:
			raise RuntimeError("JPEG incoding fail")
		
		buf.tofile(path)
		print("save sucessfully")
		#success = cv2.imwrite(path, self.frame)
		#print(f"save img: {path} → 성공: {success}")


		
		
	def show(self):
		cv2.imshow("Webcam",self.frame)
 
	def release(self):
		self.cam.release()
		cv2.destroyAllWindows()
	
	def getFrame(self):
		return self.frame
	def getImg_tk(self):
		frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(frame)
		return ImageTk.PhotoImage(img_pil)
	
	def setSize(self, x, y):
		self.size = (x, y)
		
	def getDT(self):
		return self.dt
		
	def imgShow(self, title, img):
		cv2.imshow(title, img)
	
	def __del__(self):
		pass
	

if __name__ == '__main__':
	cam = Webcam()
	
	while True:
		cam.update()
		cam.show()
		#print(cam.getDT())
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
	cam.release()
	
	
	
