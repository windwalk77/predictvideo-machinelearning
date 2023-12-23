# import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import shutil
from StorageUpload import StorageUpload

class PredictVideo:
    
    def __init__(self):    
        self.MODEL_PATH = 'model_tf/model.h5'
        self.TEMP_VIDEO_PATH = 'temp_video'
        self.TEMP_IMAGES_PATH = 'temp_images'

    def predict_and_upload(self,video_file):
        try:
            VIDEO_PATH, IMAGES_PATH = self.save_and_extract_video(video_file)
            result_predict = self.predict_images(IMAGES_PATH)
            gs_path = None

            if result_predict > 0.5:
                StorageUpload_instance = StorageUpload()
                video_name = VIDEO_PATH.split('/')[1]
                gs_path = StorageUpload_instance.upload_file(VIDEO_PATH,video_name)

            os.remove(VIDEO_PATH)
            shutil.rmtree(IMAGES_PATH)

            return result_predict,gs_path
        except Exception as e:
            print(f"Error: {e}")
    
    def predict_images(self,imagesPath):
        IMAGES_PATH = imagesPath
        MODEL_TF = self.load_model_tf()
        
        predictions = []
        labels = []
        
        for frame in os.listdir(IMAGES_PATH):
            frame_path = os.path.join(IMAGES_PATH, frame)
        
            image = cv2.imread(frame_path)
            image = cv2.resize(image, (224, 224))

            img = image / 255
            img = np.expand_dims(img, axis=0)

            predicted_image = MODEL_TF.predict(img)
            predicted_class = str(np.argmax(predicted_image))

            true_label = frame.split('_')[0]

            labels.append(true_label)
            predictions.append(predicted_class)
        
        return self.calculate_acc_score(labels,predictions)

            
    def calculate_acc_score(self,labelsArr,predictionsArr):
        correct_predictions = sum(label == prediction for label, prediction in zip(labelsArr, predictionsArr))
        total_predictions = len(predictionsArr)
        # accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def load_model_tf(self):
        return load_model(self.MODEL_PATH)
    
    def save_and_extract_video(self, video):
        video_path = self.TEMP_VIDEO_PATH
        if not os.path.exists(video_path):
            os.mkdir(video_path)
        
        saved_vid_path = video_path + '/'+ f'{video.filename}'
        video.save(saved_vid_path)
        return saved_vid_path, self.extract_frame(saved_vid_path)

    def extract_frame(self, video_path):
        images_save_path =  self.TEMP_IMAGES_PATH
        filename = os.path.basename(video_path).split('.')[0]
        filename = filename.split('_')
        
        images_folder_path = os.path.join(images_save_path,f'{filename[0]}_{filename[2]}')
        extract_frame_name = f'{filename[0]}_{filename[2]}'

        try:
            if not os.path.exists(images_folder_path):
                os.makedirs(images_folder_path)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise Exception("Failed opened video")

            frame_count = 1

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_filename = f"{extract_frame_name}_{frame_count}.jpg"
                cv2.imwrite(os.path.join(images_folder_path, frame_filename),frame)
                frame_count += 1

            cap.release()

            return images_folder_path
        except Exception as e:
            print(f"Error : {e}")

