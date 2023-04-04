import cv2
import mediapipe as mp
import time
import faceMeshTrackingModule as fmt
from playsound import playsound
import cvzone

#EYES [133,33,362,263,159,145,386,374]
def dist(l1,l2):
	return (int(((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**0.5))

def test(faces):
	try:
		eye1w,eye2w = dist(faces[133],faces[33]),dist(faces[362],faces[263])
		eye1h, eye2h = dist(faces[159],faces[145]),dist(faces[386],faces[374])
		r1,r2 = eye1w/eye1h,eye2w/eye2h
		if min(r1,r2)>5:
			return "EyesClosed"
		else:
			return "none"
	except:
		"return none"

def run():
	cap = cv2.VideoCapture(0)
	pTime = 0
	detector = fmt.faceMeshDetector()
	res = "none"
	fps = 20
	count=0
	alertduration = fps//2
	alert_type="none"
	while True:
		success,img = cap.read()
		img = cv2.flip(img,1)
		img, faces = detector.findFaceMesh(img,draw=False)
		imgGraphics = cv2.imread("steer.png", cv2.IMREAD_UNCHANGED)
		ix,iy = imgGraphics.shape[0],imgGraphics.shape[1]
		imgGraphics = cv2.resize(imgGraphics, (int(iy*2.5), int(ix*2.5)))
		img = cvzone.overlayPNG(img,imgGraphics,(300,300))
		cv2.putText(img,"DRIVING MODE",(550,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
		if alert_type=="WakeUp" and alertduration:
			cv2.putText(img,"WAKE UP!",(300,400),cv2.FONT_HERSHEY_PLAIN,10,(0,0,255),10)
			alertduration-=1
		try:
			faces = faces[0][0]
			chosen = test(faces)
			if chosen=="EyesClosed" and count>=fps:
				alert_type="WakeUp"
				playsound("WakeUp.mp3",block=False)
				alertduration=fps//2
				count=0
			elif chosen=="none":
				count=0
			elif chosen!=res:
				count=0
				res = chosen
			else:
				count+=1
		except:
			pass
		cTime = time.time()
		fps = int(1/(cTime-pTime))
		pTime = cTime
		cv2.putText(img,"FPS: "+str(fps),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
		cv2.imshow("Image",img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


run()