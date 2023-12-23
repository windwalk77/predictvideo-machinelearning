import os
from google.cloud import storage

class StorageUpload:
    def __init__(self):
        self.project_id = 'testing-407302'
        self.bucket_name = 'user_vid_bisyarat'
        self.client = storage.Client(project=self.project_id)

    def upload_file(self, file_path, destination_blob_name):
        with open(file_path, "rb") as file:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob_name)

            # Upload to Google Storage
            blob.upload_from_file(file)

        return f"https://storage.googleapis.com/{self.bucket_name}/{destination_blob_name}"