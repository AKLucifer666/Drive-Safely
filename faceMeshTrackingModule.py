import cv2
import mediapipe as mp
import time

class faceMeshDetector():
	def __init__(self,staticMode=False,maxFaces=1,refineLandmarks=True,minDetectionCon=0.5):
		self.staticMode = staticMode
		self.maxFaces = maxFaces
		self.minDetectionCon = minDetectionCon
		self.refineLandmarks = refineLandmarks

		self.mpFaceMesh = mp.solutions.face_mesh
		self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refineLandmarks,self.minDetectionCon)
		self.mpDraw = mp.solutions.drawing_utils
		self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

	def findFaceMesh(self, img, draw=True):
		imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		self.results = self.faceMesh.process(imgRGB)
		faces = []
		if self.results.multi_face_landmarks:
			for faceLms in self.results.multi_face_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
				face = []
				for idd,lm in enumerate(faceLms.landmark):
					ih, iw, ic = img.shape
					x,y = int(lm.x*iw),int(lm.y*ih)
					if idd>=0:
						if draw:
							cv2.putText(img,str(idd),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
					face.append([x,y])
				faces.append([face])
		return img, faces

def main():
	cap = cv2.VideoCapture(0)
	pTime = 0
	detector = faceMeshDetector()
	while True:
		success,img = cap.read()
		img, faces = detector.findFaceMesh(img)
		cTime = time.time()
		fps = int(1/(cTime-pTime))
		pTime = cTime
		cv2.putText(img,"FPS: "+str(fps),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
		cv2.imshow("Image",img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

if __name__ == "__main__":
	main()