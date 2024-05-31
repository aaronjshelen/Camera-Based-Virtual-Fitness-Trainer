import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl counter variables
left_curl_counter = 0 
left_curl_stage = None
right_curl_counter = 0 
right_curl_stage = None

# Lateral raise counter variables
left_lat_counter = 0
left_lat_stage = None
right_lat_counter = 0
right_lat_stage = None

# Tricep extension counter variables
left_tri_counter = 0
left_tri_stage = None
right_tri_counter = 0
right_tri_stage = None

# Squat counter variables
squat_counter = 0
squat_stage = None

# Leg extension variables
leg_ext_counter = 0
leg_ext_stage = None

# Jumping jack variables
jumping_jack_counter = 0
jumping_jack_stage = None

# High knee variables
left_knee_counter = 0
right_knee_counter = 0
left_knee_stage = None
right_knee_stage = None



def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return round(angle, 2) 

def show_loading_label():
    global loading_label
    loading_label = tk.Label(root, text="Loading...", font=("Arial", 12))
    loading_label.pack(pady=20)

def hide_loading_label():
    loading_label.destroy()

def dumbell_curl(count_reps):
    global left_curl_counter, left_curl_stage, right_curl_counter, right_curl_stage
    show_loading_label()  # Show the loading text
    root.update()  # Update the GUI to display the loading text
    # root.after(1000, hide_loading_label)  # Hide the loading text after 1 second


    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if count_reps:
                    num_reps = 5  # Change this to 5 for 5 reps
                    if left_curl_counter >= num_reps and right_curl_counter >= num_reps:
                        print("Reached reps for both arms. Quitting...")
                        left_curl_counter = 0
                        right_curl_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps


                landmarks = results.pose_landmarks.landmark
                
                # LEFT CURL: Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Visualize angle
                cv2.putText(image, str(left_angle), 
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if left_angle > 160:
                    left_curl_stage = "down"
                if left_angle < 30 and left_curl_stage == 'down':
                    left_curl_stage = "up"
                    left_curl_counter += 1
                    # print("Dumbell - LEFT: " + left_curl_counter)

                # RIGHT CURL: Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Visualize angle
                cv2.putText(image, str(right_angle), 
                            tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if right_angle > 160:
                    right_curl_stage = "down"
                if right_angle < 30 and right_curl_stage == 'down':
                    right_curl_stage = "up"
                    right_curl_counter += 1
                    # print("Dumbell - RIGHT: " + right_curl_counter)


                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_curl_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (582,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_curl_counter), 
                        (582,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                left_curl_counter = 0
                right_curl_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()

        
        # main()  # Restart the main menu

def lateral_raise(count_reps):
    global left_lat_counter, left_lat_stage, right_lat_counter, right_lat_stage
    show_loading_label()  # Show the loading text
    root.update()  # Update the GUI to display the loading text
    # root.after(1000, hide_loading_label)  # Hide the loading text after 1 second

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                if count_reps:
                    num_reps = 5  # Change this to 5 for 5 reps
                    if left_lat_counter >= num_reps and right_lat_counter >= num_reps:
                        print("Reached reps for both arms. Quitting...")
                        left_lat_counter = 0
                        right_lat_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps                
                landmarks = results.pose_landmarks.landmark
                
                # LEFT CURL: Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Calculate angle
                left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                
                # Visualize angle
                cv2.putText(image, str(left_angle), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if left_angle > 80:
                    left_lat_stage = "down"
                if left_angle < 45 and left_lat_stage =='down':
                    left_lat_stage="up"
                    left_lat_counter +=1
                    # print("Lateral Raise - LEFT: " + left_lat_counter)

                # RIGHT CURL: Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angle
                right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                
                # Visualize angle
                cv2.putText(image, str(right_angle), 
                            tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if right_angle > 80:
                    right_lat_stage = "down"
                if right_angle < 45 and right_lat_stage =='down':
                    right_lat_stage="up"
                    right_lat_counter +=1
                    # print("Lateral Raise - RIGHT: " + right_lat_counter)
                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_lat_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (582,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_lat_counter), 
                        (582,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                left_lat_counter = 0
                right_lat_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()

        
        # main()  # Restart the main menu

def tricep_ext(count_reps):
    global left_tri_counter, left_tri_stage, right_tri_counter, right_tri_stage
    show_loading_label()  # Show the loading text
    root.update()  # Update the GUI to display the loading text
    # root.after(1000, hide_loading_label)  # Hide the loading text after 1 second

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                if count_reps:
                    num_reps = 5  # Change this to 5 for 5 reps
                    if left_tri_counter >= num_reps and right_tri_counter >= num_reps:
                        print("Reached reps for both arms. Quitting...")
                        left_tri_counter = 0
                        right_tri_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps
                landmarks = results.pose_landmarks.landmark
                

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)

                cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y:
                    if left_angle < 60 :
                        left_tri_stage = "compressed"
                        # print(left_tri_stage)
                    if left_angle > 165 and left_tri_stage == "compressed":
                        left_tri_stage = "extended"
                        left_tri_counter += 1
                        print(left_tri_counter)

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

                cv2.putText(image, str(right_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                if landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y:
                    if right_angle < 60 :
                        right_tri_stage = "compressed"
                        # print(left_tri_stage)
                    if right_angle > 165 and right_tri_stage == "compressed":
                        right_tri_stage = "extended"
                        right_tri_counter += 1
                        # print(right_tri_counter)

                cv2.putText(image, str(right_angle), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
 
                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_tri_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (582,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_tri_counter), 
                        (582,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                left_tri_counter = 0
                right_tri_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()

        
        # main()  # Restart the main menu


def squats(count_reps):
    global squat_counter, squat_stage
    show_loading_label()
    root.update()

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if count_reps:
                    num_reps = 5  # Change this to 5 for 5 reps
                    if squat_counter >= num_reps:
                        print("Reached reps for squats. Quitting...")
                        squat_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps

                landmarks = results.pose_landmarks.landmark
                

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_angle = calculate_angle(left_ankle, left_knee, left_hip)

                cv2.putText(image, str(left_angle), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                right_angle = calculate_angle(right_ankle, right_knee, right_hip)

                cv2.putText(image, str(right_angle), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)



                if left_angle < 90 and right_angle < 90:
                    squat_stage = "compressed"
                    print("compressed")

                if left_angle > 165 and right_angle > 165 and squat_stage == "compressed":
                    squat_stage = "extended"
                    squat_counter += 1
                    print(squat_counter)


                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            # cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(squat_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                squat_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()


def leg_ext(count_reps):
    global leg_ext_counter, leg_ext_stage
    show_loading_label()
    root.update()

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if count_reps:
                    num_reps = 5  # Change this to 5 for 5 reps
                    if leg_ext_counter >= num_reps:
                        print("Reached reps for squats. Quitting...")
                        leg_ext_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps

                landmarks = results.pose_landmarks.landmark
                

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_angle = calculate_angle(left_ankle, left_knee, left_hip)

                cv2.putText(image, str(left_angle), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                right_angle = calculate_angle(right_ankle, right_knee, right_hip)

                cv2.putText(image, str(right_angle), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)



                if left_angle < 90 and right_angle < 90:
                    leg_ext_stage = "compressed"
                    # print("compressed")

                if left_angle > 165 and right_angle > 165 and leg_ext_stage == "compressed":
                    leg_ext_stage = "extended"
                    leg_ext_counter += 1
                    # print(squat_counter)


                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            # cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(leg_ext_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                squat_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()




def jumping_jacks(count_reps):
    global jumping_jack_counter, jumping_jack_stage
    show_loading_label()  # Show the loading text
    root.update()  # Update the GUI to display the loading text
    # root.after(1000, hide_loading_label)  # Hide the loading text after 1 second

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                if count_reps:
                    num_reps = 10  # Change this to 5 for 5 reps
                    if jumping_jack_counter == num_reps:
                        print("Reached reps for jumping jacks. Quitting...")
                        jumping_jack_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps                
                landmarks = results.pose_landmarks.landmark
                
                # LEFT ARM: Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Calculate angle
                left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                
                # Visualize angle
                cv2.putText(image, str(left_angle), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                

                # RIGHT ARM: Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angle
                right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                
                # Visualize angle
                cv2.putText(image, str(right_angle), 
                            tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # LEFT and RIGHT LEG: Get coordinates
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculate angle
                left_leg_angle = calculate_angle(left_knee, left_hip, right_hip)
                
                # Visualize angle
                cv2.putText(image, str(left_leg_angle), 
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                
                # Calculate angle
                right_leg_angle = calculate_angle(right_knee, right_hip, left_hip)
                
                # Visualize angle
                cv2.putText(image, str(right_leg_angle), 
                            tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                
                if left_angle < 20 and right_angle < 20 and left_leg_angle < 90 and right_leg_angle < 90:
                    jumping_jack_stage = "down"
                    # print("compressed")

                if  left_angle > 150 and right_angle > 150 and left_leg_angle > 100 and right_leg_angle > 100 and jumping_jack_stage == "down":
                    jumping_jack_stage = "up"
                    jumping_jack_counter += 1
                    # print(squat_counter)
                
                # if left_angle < 20 and right_angle < 20:
                #     jumping_jack_stage = "down"
                #     # print("compressed")

                # if left_angle > 160 and right_angle > 160 and jumping_jack_stage == "down":
                #     jumping_jack_stage = "up"
                #     jumping_jack_counter += 1
                #     # print(squat_counter)
                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(jumping_jack_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # cv2.putText(image, 'REPS', (582,12), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(left_lat_counter), 
            #             (582,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                left_lat_counter = 0
                right_lat_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()

        
        # main()  # Restart the main menu


def high_knees(count_reps):
    global left_knee_counter, right_knee_counter, left_knee_stage, right_knee_stage
    show_loading_label()
    root.update()

    cap = cv2.VideoCapture(0)
    hide_loading_label()
    root.update()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if count_reps:
                    num_reps = 10  # Change this to 5 for 5 reps
                    if left_knee_counter >= num_reps and right_knee_counter >= num_reps:
                        print("Reached reps for squats. Quitting...")
                        left_knee_counter = 0
                        right_knee_counter = 0
                        break  # Exit the loop if both arms have completed 5 reps

                landmarks = results.pose_landmarks.landmark
                


                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]




                if left_knee[1] > left_hip[1]:
                    left_knee_stage = "up"
                    # print("compressed")
                if left_knee[1] < left_hip[1] and left_knee_stage == "up":
                    left_knee_stage = "down"
                    left_knee_counter += 1
                    # print("left")

                

                if right_knee[1] > right_hip[1]:
                    right_knee_stage = "up"
                    # print("compressed")
                if right_knee[1] < right_hip[1] and right_knee_stage == "up":
                    right_knee_stage = "down"
                    right_knee_counter += 1
                    # print("right")



                        
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (73,73), (245,117,16), -1)
            cv2.rectangle(image, (567, 0), (640, 73), (245, 117, 16), -1)  # Rectangle on the other side

            # Rep data 
            # GOTTA FLIP RIGHT AND LEFT COUNTERS BECAUSE OF CAMERA FLIP
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_knee_counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (582,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_knee_counter), 
                        (582,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                squat_counter = 0
                break

        cap.release()
        cv2.destroyAllWindows()




def full_arm_workout():
    dumbell_curl(True)
    # wanting to implement a rest timer here
    lateral_raise(True)
    # wanting to implement a rest timer here
    tricep_ext(True)


def full_leg_workout():
    squats(True)
    # rest timer
    leg_ext(True)

def full_cardio_workout():
    jumping_jacks(True)
    # rest timer
    high_knees(True)








def main():
    global root
    # Create the main window
    root = tk.Tk()
    root.title("Workout Menu")

    root.geometry("400x750")

    arms_label = tk.Label(root, text="Arms", font=("Helvetica", 15, "bold"))
    arms_label.pack(pady=10)

    button = tk.Button(root, text="Dumbell Curl", command=lambda: dumbell_curl(False))
    button.pack(pady=10)

    button2 = tk.Button(root, text="Lateral Raise", command=lambda: lateral_raise(False))
    button2.pack(pady=10)

    button3 = tk.Button(root, text="Overhead Tricep Extension", command=lambda: tricep_ext(True))
    button3.pack(pady=10)

    button4 = tk.Button(root, text="Full Arm Workout", command=full_arm_workout, bg="#964B00", fg="white")
    button4.pack(pady=10)

    # Legs Section
    legs_label = tk.Label(root, text="Legs", font=("Helvetica", 15, "bold"))
    legs_label.pack(pady=10)

    button5 = tk.Button(root, text="Squats", command=lambda: squats(False))
    button5.pack(pady=10)

    button6 = tk.Button(root, text="Leg Extensions", command=lambda: leg_ext(False))
    button6.pack(pady=10)

    button7 = tk.Button(root, text="Full Leg Workout", command=full_leg_workout, bg="#964B00", fg="white")
    button7.pack(pady=10)

    
    # Cardio Section
    cardio_label = tk.Label(root, text="Cardio", font=("Helvetica", 15, "bold"))
    cardio_label.pack(pady=10)

    button8 = tk.Button(root, text="Jumping Jacks", command=lambda: jumping_jacks(False))
    button8.pack(pady=10)

    button9 = tk.Button(root, text="High Knees", command=lambda: high_knees(False))
    button9.pack(pady=10)

    button10 = tk.Button(root, text="Full Cardio", command=full_cardio_workout, bg="#964B00", fg="white")
    button10.pack(pady=10)


    root.mainloop()



if __name__ == "__main__":
    main()
