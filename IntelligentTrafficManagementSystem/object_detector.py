import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO
from flask import request, Response, Flask, render_template, jsonify
from PIL import Image
import json

app = Flask(__name__)

@app.route('/')
def test():
    return render_template('test.html')

@app.route('/', methods=["POST"])
def index():
    return render_template('index.html')

@app.route('/test2', methods=["POST"])
def test2():
    return render_template('test2.html')

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes, box_count = detect_objects_on_image(Image.open(buf.stream))
    response_data = {
        "boxes": boxes,
        "box_count": box_count
    }
    return Response(
        json.dumps(response_data),
        mimetype='application/json'
    )

def detect_objects_on_image(buf, confidence_threshold=0.01):
    model = YOLO("models/best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    totalbox = []

    box_count = 0  # Initialize box count
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        if prob >= confidence_threshold:  # Check if confidence meets threshold
            output.append([
              x1, y1, x2, y2, result.names[class_id], prob
            ])
            box_count += 1  # Increment box count for each box detected
    totalbox.append(box_count)
    return output, totalbox


@app.route("/predict_img", methods=["POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'mp4':
                # Load the best fine-tuned YOLOv8 model
                model = YOLO('models/best.onnx')

                # Define the threshold for considering traffic as heavy
                heavy_traffic_threshold = 10

                # Define the vertices for the quadrilaterals
                vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
                vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

                # Define the vertical range for the slice and lane threshold
                x1, x2 = 325, 635
                lane_threshold = 609

                # Define the positions for the text annotations on the image
                text_position_left_lane = (10, 50)
                text_position_right_lane = (820, 50)
                intensity_position_left_lane = (10, 100)
                intensity_position_right_lane = (820, 100)

                # Define font, scale, and colors for the annotations
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 255)  # White color for text
                background_color = (0, 0, 255)  # Red background for text

                # Add video processing logic here
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('processed_sample_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

                # Read until video is completed
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret:
                        # Create a copy of the original frame to modify
                        detection_frame = frame.copy()

                        # Black out the regions outside the specified vertical range
                        detection_frame[:x1, :] = 0  # Black out from top to x1
                        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame

                        # Perform inference on the modified frame
                        results = model.predict(detection_frame, imgsz=640, conf=0.4)
                        processed_frame = results[0].plot(line_width=1)

                        # Restore the original top and bottom parts of the frame
                        processed_frame[:x1, :] = frame[:x1, :].copy()
                        processed_frame[x2:, :] = frame[x2:, :].copy()

                        # Draw the quadrilaterals on the processed frame
                        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
                        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

                        # Retrieve the bounding boxes from the results
                        bounding_boxes = results[0].boxes

                        # Initialize counters for vehicles in each lane
                        vehicles_in_left_lane = 0
                        vehicles_in_right_lane = 0

                        # Loop through each bounding box to count vehicles in each lane
                        for box in bounding_boxes.xyxy:
                            # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
                            if box[0] < lane_threshold:
                                vehicles_in_left_lane += 1
                            else:
                                vehicles_in_right_lane += 1

                        # Determine the traffic intensity for the left lane
                        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
                        # Determine the traffic intensity for the right lane
                        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

                        # Add a background rectangle for the left lane vehicle count
                        cv2.rectangle(processed_frame, (text_position_left_lane[0] - 10, text_position_left_lane[1] - 25), (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)

                        # Add the vehicle count text on top of the rectangle for the left lane
                        cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane, font, font_scale, font_color, 2, cv2.LINE_AA)

                        # Add a background rectangle for the left lane traffic intensity
                        cv2.rectangle(processed_frame, (intensity_position_left_lane[0] - 10, intensity_position_left_lane[1] - 25), (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)

                        # Add the traffic intensity text on top of the rectangle for the left lane
                        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, font, font_scale, font_color, 2, cv2.LINE_AA)

                        # Add a background rectangle for the right lane vehicle count
                        cv2.rectangle(processed_frame, (text_position_right_lane[0] - 10, text_position_right_lane[1] - 25), (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)

                        # Add the vehicle count text on top of the rectangle for the right lane
                        cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane, font, font_scale, font_color, 2, cv2.LINE_AA)

                        # Add a background rectangle for the right lane traffic intensity
                        cv2.rectangle(processed_frame, (intensity_position_right_lane[0] - 10, intensity_position_right_lane[1] - 25), (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)

                        # Add the traffic intensity text on top of the rectangle for the right lane
                        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, font, font_scale, font_color, 2, cv2.LINE_AA)

                        # Display the processed frame
                        cv2.imshow('Real-time Traffic Analysis', processed_frame)

                        # Press Q on keyboard to exit the loop
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break

                # Release the video capture and video write objects
                cap.release()
                out.release()

                # Close all the frames
                cv2.destroyAllWindows()
                return jsonify({"message": "Image/Video processed successfully."}), 200



if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
    # Set logging level to ERROR
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)