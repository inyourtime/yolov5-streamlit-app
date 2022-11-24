import torch
import numpy as np
import cv2
import pafy
import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self, img_path=None, video_path=None, video_out=None):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.img_path = img_path
        self.video_path = video_path
        self.video_out = video_out
        self.model = self.load_model()
        self.classes = self.model.names
        self.track = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)


    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        print(f"[INFO] Loading model... ")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5l.pt')
        model.conf = 0.5
        model.iou = 0.4
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        # for param in model.parameters():
        #     print(param)
        # print(model.parameters()[0])
        # for name, param in model.named_parameters():
        #     print(name)
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        print(f"[INFO] Detecting. . . ")
        results = self.model(frame, size=640)
     
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cordinates


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        # print(x_shape, y_shape)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                print(f"[INFO] Extracting BBox coordinates. . . ")
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                # if len(self.track) >= 60:
                #     self.track.pop(0)
                # self.track.append((int((x1+x2)/2), int((y1+y2)/2)))
                # print(self.track[0][0]/x_shape, self.track[0][1]/y_shape)                   
                bgr = (42, 88, 220)

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                # for t in self.track:
                #     cv2.circle(frame, t, radius=2, color=(0,0,255), thickness=-1)
                text = self.class_to_label(labels[i]) + " " + str(round(row[4].item(), 2))
                self.draw_text(frame, text, pos=(x1,y1), text_color_bg=bgr)
                # cv2.putText(frame, self.class_to_label(labels[i]) + " " + str(round(row[4].item(), 2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def draw_text(self, img, text,
                font=cv2.FONT_HERSHEY_PLAIN,
                pos=(0,0),
                font_scale=3,
                font_thickness=2,
                text_color=(255, 255, 255),
                text_color_bg=(0, 0, 0)
                ):
        x, y = pos
        text_w, text_h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(img, (x, y - text_h), (x + text_w, y), text_color_bg, -1)
        cv2.putText(img, text, (x, y + font_scale - 1), font, font_scale, text_color, font_thickness)


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        
        if self.img_path != None:
            print(f"[INFO] Working with image: {self.img_path}")
            frame = cv2.imread(self.img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            results = self.score_frame(frame)

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = self.plot_boxes(results, frame)

            cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

            while True:
                cv2.imshow("img_only", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"[INFO] Exiting. . . ")
                    # cv2.imwrite("cat_output_02.jpg",frame) ## if you want to save he output result.
                    break
        
        elif self.video_path != None:
            print(f"[INFO] Working with video: {self.video_path}")

            cap = cv2.VideoCapture(self.video_path)
            windows = []
            p = "video_out"
            # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(w,h)

            if self.video_out: ### creating the video writer if video output path is given

                # by default VideoCapture returns float instead of int
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
                out = cv2.VideoWriter(self.video_out, codec, fps, (width, height))

            while cap.isOpened():
                
                # start_time = time.perf_counter()
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = self.score_frame(frame)
                # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = self.plot_boxes(results, frame)
                # end_time = time.perf_counter()
                end_time = time.time()
                fps = 1 / np.round(end_time - start_time, 3)
                cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                # frame = cv2.resize(frame, (w,h))
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), frame.shape[1], frame.shape[0])
                cv2.imshow(str(p), frame)

                if self.video_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # out.release()
            cv2.destroyAllWindows()


# Create a new object and execute.
# detection = ObjectDetection(img_path="test_cat_02.jpeg")
# detection = ObjectDetection(video_path="new_test.mp4", video_out="runs/exp5.mp4")
# detection = ObjectDetection(video_path="video/test2.mp4")
detection = ObjectDetection(video_path="http://192.168.111.26:80/cam")
# detection = ObjectDetection(video_path=0)


detection()