import cv2
from tkinter import *
import pandas
from datetime import datetime

class Cascade:
	
	global cascade_obj
	
	def __init__(self, window):
		self.window = window
		self.window.wm_title("Face detection")
		
		l1 = Label(window, text = "Enter path of the image")
		l1.grid(row = 0, column = 0)
		
		self.path_text = StringVar()
		self.e1 = Entry(window, textvariable = self.path_text)
		self.e1.grid(row = 0, column = 2, rowspan = 2)
		
		b1 = Button(window, text = "Process face", width = 12, command = self.face_cascade_fun)
		b1.grid(row = 4, column = 4)

		b2 = Button(window, text = "Process eye", width = 12, command = self.eye_cascade_fun)
		b2.grid(row = 5, column = 4)

		b3 = Button(window, text = "Detect Smile", width = 12, command = self.smile_cascade_fun)
		b3.grid(row = 6, column = 4)

		b4 = Button(window, text = "Full body detect", width = 12, command = self.full_body_cascade_fun)
		b4.grid(row = 7, column = 4)
		
		b5 = Button(window, text = "Close", width = 12, command = self.window.destroy)
		b5.grid(row = 8, column = 4)
		
		b6 = Button(window, text = "Creator", width = 12, command = self.creator)
		b6.grid(row = 8, column = 1)
		
		b7 = Button(window, text = "Capture picture", width = 12, command = self.face_webcam_pic)
		b7.grid(row = 8, column = 2)
		
		b8 = Button(window, text = "Capture Video", width = 12, command = self.video_capture)
		b8.grid(row = 8, column = 3)
		
		b8 = Button(window, text = "Motion detect", width = 12, command = self.motion_detect)
		b8.grid(row = 6, column = 0)
		
		self.list1 = Listbox(window, height = 3, width = 25)
		self.list1.grid(row = 6, column = 1, columnspan = 3)
		
		sb1 = Scrollbar(window)
		sb1.grid(row = 6, column = 1)
		
		self.list1.configure(yscrollcommand = sb1.set)
		sb1.configure(command = self.list1.yview)

		
	def face_cascade_fun(self): 
		try:
			img = cv2.imread(self.path_text.get())
			img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cascade_obj = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
			faces = cascade_obj.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
			for x,y,w,h in faces:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			cv2.imshow("Gray",img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			self.list1.delete(0,END)
			self.list1.insert(END, "Success!")
		except AttributeError:
			self.list1.delete(0,END)
			self.list1.insert(END, "Please give a valid path!")


	def eye_cascade_fun(self):
		try:
			img = cv2.imread(self.path_text.get())
			img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cascade_obj = cv2.CascadeClassifier("haarcascade_eye.xml")
			eyes = cascade_obj.detectMultiScale(gray_img, scaleFactor = 1.25, minNeighbors = 5)
			for x,y,w,h in eyes:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			cv2.imshow("Gray",img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			self.list1.delete(0,END)
			self.list1.insert(END, "Success!")
		except AttributeError:
			self.list1.delete(0,END)
			self.list1.insert(END, "Please give a valid path!")
	
	def smile_cascade_fun(self):
		try:
			img = cv2.imread(self.path_text.get())
			img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cascade_obj = cv2.CascadeClassifier("haarcascade_smile.xml")
			smile = cascade_obj.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
			for x,y,w,h in smile:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			cv2.imshow("Gray",img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			self.list1.delete(0,END)
			self.list1.insert(END, "Success!")
		except AttributeError:
			self.list1.delete(0,END)
			self.list1.insert(END, "Please give a valid path!")
			
	def full_body_cascade_fun(self):
		try:
			img = cv2.imread(self.path_text.get())
			img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cascade_obj = cv2.CascadeClassifier("haarcascade_fullbody.xml")
			eyes = cascade_obj.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
			for x,y,w,h in eyes:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			cv2.imshow("Gray",img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			self.list1.delete(0,END)
			self.list1.insert(END, "Success!")
		except AttributeError:
			self.list1.delete(0,END)
			self.list1.insert(END, "Please give a valid path!")
	
	def face_webcam_pic(self):
			video = cv2.VideoCapture(0)

			check, frame = video.read()
			
			img = frame
			img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cascade_obj = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
			faces = cascade_obj.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
			for x,y,w,h in faces:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			cv2.imshow("Captured",img)
			cv2.waitKey(0)
		
			self.list1.delete(0,END)
			self.list1.insert(END, "Success!")
			video.release()
			cv2.destroyAllWindows()
	
	def video_capture(self):
		video = cv2.VideoCapture(0)

		frame_no = 0
		while True:
			frame_no = frame_no + 1
			check, frame = video.read()
			img = frame
			img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
			smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
			smile = smile_cascade.detectMultiScale(gray_img, scaleFactor = 1.5, minNeighbors = 5)
			for x,y,w,h in smile:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
			faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
			for x,y,w,h in faces:
				img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)

			cv2.imshow("Capturing",img)
			key = cv2.waitKey(80)
			
			if key == ord('q'):
				break

		
		self.list1.insert(END,"Number of frames:"+str(frame_no))
		video.release()
		cv2.destroyAllWindows()
	
	def motion_detect(self):
		self.list1.delete(0,END)
		video = cv2.VideoCapture(0)

		first_frame = None

		times = []
		status_list = [None, None]
		df = pandas.DataFrame(columns = ["Start","End"])

		while True:
			check, frame = video.read()
			status = 0

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#for removing noises and edge, param: img, gaussian kernel dim, standard deviation
			gray = cv2.GaussianBlur(gray, (21,21), 0)
			#below operations is not to be done to the first frame
			if first_frame is None:
				first_frame = gray
				continue
			
			delta_frame = cv2.absdiff(first_frame, gray)
			
			key = cv2.waitKey(1)

			#beyond the intensity of 40 would be classified as object
			thresh_frame = cv2.threshold(delta_frame, 40, 255, cv2.THRESH_BINARY)[1]
			
			#to remove the black holes within the detection frame and smoothen it 
			thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

			#outlining the contours of all the objects
			(cnts,_) = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			#not considering pixel area of less than 1000
			for contour in cnts:
				if cv2.contourArea(contour) < 1000:
					continue
				status = 1
				(x,y,w,h) = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
			
			cv2.imshow("Gray Frame",gray)
			cv2.imshow("Delta Frame", delta_frame)
			cv2.imshow("Thresh hold", thresh_frame)
			cv2.imshow("Color frame",frame)
			
			if key == ord('q'):
				if status == 1:
					self.list1.insert(END,datetime.now())
					self.list1.insert(END,"Exited screen")
				break
				
			status_list.append(status)
			
			if status_list[-1] == 1 and status_list[-2] == 0:
				self.list1.insert(END,"Entered screen")
				self.list1.insert(END,datetime.now())
			if status_list[-1] == 0 and status_list[-2] == 1:
				self.list1.insert(END,"Exited screen")
				self.list1.insert(END,datetime.now())

		'''for i in range(0,len(times),2):
			df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

		df.to_csv("Times.csv")'''
				
		video.release()
		cv2.destroyAllWindows()
	
	
	def creator(self):
		self.list1.delete(0,END)
		self.list1.insert(END, "Created by Shubham Banerjee!")
	
	#def load_image():
	#	self.img = cv2.imread(self.path_text.get())
	#	self.img = cv2.resize(self.img,(int(self.img.shape[1]/3),int(self.img.shape[0]/3)))
	
	
window = Tk()
Cascade(window)
window.mainloop()

