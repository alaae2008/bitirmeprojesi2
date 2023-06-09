# python vicc.py --input traffic.mp4

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import json
import cv2
import os
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required = True, type=str,
	help="path to input video file")
args = vars(ap.parse_args())



def count_objects(objects, object_class, total, temp):
	global total_cars
	global temp_cars
	global total_persons
	global temp_persons
	global total_trucks
	global temp_trucks
	global total_buses
	global temp_buses
	global total_bikes
	global temp_bikes
	global total_bicycles
	global temp_bicycles


	if object_class == "car":
		total, temp = total_cars, temp_cars
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_cars = total
		temp_cars = temp	

	
	
	elif object_class == "person":
		total, temp = total_persons, temp_persons
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_persons = total
		temp_persons = temp	

	
	
	
	elif object_class == "truck":
		total, temp = total_trucks, temp_trucks
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_trucks = total
		temp_trucks = temp	
	
	
	elif object_class == "bus":

		total, temp = total_buses, temp_buses
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_buses = total
		temp_buses = temp	
	
	
	elif object_class == "bike":
		total, temp = total_bikes, temp_bikes	
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_bikes = total
		temp_bikes = temp	
	
	
	elif object_class == "bicycle":
		total, temp = total_bicycles, temp_bicycles		
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		total_bicycles = total
		temp_bicycles = temp	
	
	
	return total	





def draw_centroids(frame, objects, trackableObjects):
	for (objectID, centroid) in objects.items():

		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		trackableObjects[objectID] = to


		text = "ID {}".format(objectID + 1)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

net = cv2.dnn.readNet("yolo/yolov3_608.weights","yolo/yolov3_608.cfg")
with open("yolo/yolov3_608.names", 'r') as f:
	CLASSES = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
vs = cv2.VideoCapture("videos/"+ args["input"])
heig = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
wid = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
fps= vs.get(cv2.CAP_PROP_FPS)
print (heig,wid,fps)
inpWidth = (round(wid/32))*32
inpHeight = (round(heig/32))*32
print(inpHeight,inpWidth)
writer = None
output_count = 1
while True:
	if output_count > 100:
		for file in os.listdir("output"):
			os.remove(os.getcwd() + "/output/" + file)
			output_count = 1
	if "{}_proccesed.avi".format(output_count) not in os.listdir("output"):
		writer_path = "output/" + "{}_proccesed.avi".format(output_count)
		break
	else:
		output_count += 1


width = None
height = None


car_ct = CentroidTracker()
car_ct.maxDisappeared = 10
person_ct = CentroidTracker()
person_ct.maxDisappeared = 10
truck_ct = CentroidTracker()
truck_ct.maxDisappeared = 10
bike_ct = CentroidTracker()
bike_ct.maxDisappeared = 10
bicycle_ct = CentroidTracker()
bicycle_ct.maxDisappeared = 10
bus_ct = CentroidTracker()
bus_ct.maxDisappeared = 10

trackers = []
car_trackableObjects = {}
person_trackableObjects = {}
truck_trackableObjects = {}
bus_trackableObjects = {}
bike_trackableObjects = {}
bicycle_trackableObjects = {}

total_cars, temp_cars = 0, None
total_persons, temp_persons = 0, None
total_trucks, temp_trucks = 0, None
total_buses, temp_buses = 0, None
total_bikes, temp_bikes = 0, None
total_bicycles, temp_bicycles = 0, None


totalFrames = 0

total = 0

status = None

frame_number = 0

count_cars, count_persons, count_trucks, count_buses, count_bikes, count_bicycles = 0, 0, 0, 0, 0, 0

while True:
	frame_number += 1
	frame = vs.read()
	frame = frame[1]




	if frame is None:
		print("=============================================")
		print("The end of the video reached")
		print("Total number of freams on the video is ", totalFrames)
		print("=============================================")
		break

	frame = imutils.resize(frame, width=inpWidth, height=inpHeight)

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if width is None or height is None:
		height, width, channels = frame.shape

	car_rects = []
	person_rects = []
	truck_rects = []
	bus_rects = []
	bike_rects = []
	bicycle_rects = []

	if  writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(writer_path,fourcc, round(int(vs.get(cv2.CAP_PROP_FPS))),(width, height), True)


	if totalFrames % round(int(vs.get(cv2.CAP_PROP_FPS)*2)) == 0:
		trackers = []
		class_ids = []
		count = 0

		status = "Detecting..."

		blob = cv2.dnn.blobFromImage(rgb, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)


		for out in outs:
			for detection in out:
				
				scores = detection[5:]
				class_id = np.argmax(scores)
				

				if class_id == 0: 
					pass


				confidence = scores[class_id]
				if confidence > 0.90:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)


					x1 = int(center_x - w / 2)
					y1 = int(center_y - h / 2)
					x2 = x1 + w
					y2 = y1 + h

					person_ct.maxDistance = w
					bike_ct.maxDistance = w
					bicycle_ct.maxDistance = w
					bus_ct.maxDistance = w
					truck_ct.maxDistance = w
					car_ct.maxDistance = w


					count += 1

					cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255 , 0), 1)
					cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(x1, y1, x2, y2)
					tracker.start_track(rgb, rect)
					trackers.append(tracker)
					class_ids.append(class_id)

						

	else:
		for tracker, class_id in zip(trackers, class_ids):
			status = "Tracking..."

		
			tracker.update(rgb)
			pos = tracker.get_position()

			x1 = int(pos.left())
			y1 = int(pos.top())
			x2 = int(pos.right())
			y2 = int(pos.bottom())


			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255 , 0), 1)
			cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

			obj_class = CLASSES[class_id]
			
			if obj_class == "car":
				car_rects.append((x1, y1, x2, y2))
			elif obj_class == "person":
				person_rects.append((x1, y1, x2, y2))	
			elif obj_class == "truck":
				truck_rects.append((x1, y1, x2, y2))	
			elif obj_class == "bus":
				bus_rects.append((x1, y1, x2, y2))			
			elif obj_class == "motorcycle":
				bike_rects.append((x1, y1, x2, y2))		
			elif obj_class == "bicycle":
				bicycle_rects.append((x1, y1, x2, y2))	


	cars = car_ct.update(car_rects)
	persons = person_ct.update(person_rects)
	trucks = truck_ct.update(truck_rects)
	buses = bus_ct.update(bus_rects)
	bikes = bike_ct.update(bike_rects)
	bicycles = bicycle_ct.update(bicycle_rects)



	if cars != {}:
		count_cars = count_objects(cars, "car", total_cars, temp_cars)
	if persons != {}:	
		count_persons = count_objects(persons, "person", total_persons, temp_persons)
	if trucks != {}:
		count_trucks = count_objects(trucks, "truck", total_trucks, temp_trucks)
	if buses != {}:	
		count_buses = count_objects(buses, "bus", total_buses, temp_buses)
	if bikes != {}:
		count_bikes = count_objects(bikes, "bike", total_bikes, temp_bikes)
	if bicycles != {}:
		count_bicycles = count_objects(bicycles, "bicycle", total_bicycles, temp_bicycles)


	draw_centroids(frame, cars, car_trackableObjects)
	draw_centroids(frame, persons, person_trackableObjects)
	draw_centroids(frame, trucks, truck_trackableObjects)
	draw_centroids(frame, buses, bus_trackableObjects)
	draw_centroids(frame, bikes, bike_trackableObjects)
	draw_centroids(frame, bicycles, bicycle_trackableObjects)


	info = [
		("cars: ", count_cars),
		("people: ", count_persons),
		("trucks: ", count_trucks),
		("buses: ", count_buses),
		("bikes: ", count_bikes),
		("bicycles", count_bicycles),
	]

	data = [{
		"cars" : str(count_cars),
		"people" : str(count_persons),
		"trucks" : str(count_trucks),
		"buses:" : str(count_buses),
		"motorcycles:": str(count_bikes),
		"bycicles" : str(count_bicycles),
	}]

	for (i, (object_class, total)) in enumerate(info):
		text = "{}: {}".format(object_class, total)
		cv2.putText(frame, text, (10, height - ((i * 20) + 20)),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

	cv2.putText(frame, "Now: " + str(count), (width - 130, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)	

	if writer is not None:
		writer.write(frame)



	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		print("[INFO] process finished by user")

		break

	totalFrames += 1

plt.show()


with open("output/"  + "analysis_results_{}.json".format(output_count), 'w') as f:
	json.dump(data, f)

print("\nThe results are:")
with open("output/"  + "analysis_results_{}.json".format(output_count), 'r') as f:
	data = json.load(f)
	for el in data:
		for key, value in el.items():
			print(key + " " + value)

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
