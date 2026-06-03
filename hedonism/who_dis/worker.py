from celery import Celery
from PIL import Image
from pillow_heif import register_heif_opener
from deepface import DeepFace
import tempfile
import urllib.request
import psycopg2
from sklearn.cluster import DBSCAN
import numpy as np
import json
from itertools import groupby

# Register the plugin to enable HEIF support in Pillow
register_heif_opener()

app = Celery('hedonism_who_dis', broker='redis://default@127.0.0.1:6379/0', result_backend='redis://default@127.0.0.1:6379/0')

@app.task
def cluster_faces(date_grouping):
    # 1. Connect to PostgreSQL and fetch embeddings
    with psycopg2.connect("dbname=hedonism_bot_development host=localhost port=5432") as conn:
        with conn.cursor() as cur:
            if date_grouping is None:
                cur.execute("SELECT photo_people.id, arc_face_embedding FROM photo_people WHERE photo_people.arc_face_embedding IS NOT NULL")
            else:
                cur.execute("SELECT photo_people.id, arc_face_embedding FROM photo_people INNER JOIN photos ON photo_people.photo_id = photos.id WHERE photos.folder_date = %s", (date_grouping,))
            data = cur.fetchall()

            # 2. Separate IDs and Vectors
            ids = [row[0] for row in data]
            embeddings = np.array([json.loads(row[1]) for row in data])

            # 3. Apply DBSCAN
            # eps: maximum distance between two samples for them to be considered as in the same neighborhood
            # min_samples: number of samples in a neighborhood for a point to be considered as a core point
            clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine').fit(embeddings)
            labels = clustering.labels_

            # 4. Update the database with cluster assignments
            update_data = [ [ int(label), id_] for label, id_ in zip(labels, ids)   ]

    return list(update_data)

@app.task
def download_convert_and_extract_facial_data(heif_url):
    print(f"heif_url -> {heif_url}")
    with tempfile.NamedTemporaryFile(suffix=".hif") as temp_file:
        urllib.request.urlretrieve(heif_url, temp_file.name)
        temp_file.seek(0)
        convert_and_extract_facial_data(temp_file)
        return convert_and_extract_facial_data(temp_file)

@app.task
def convert_and_extract_facial_data(heif_data):
    image = Image.open(heif_data)

    # Convert image mode to RGB (required for saving as JPEG)
    rgb_image = image.convert("RGB")

    with tempfile.TemporaryFile() as temp_file:
        # Save as a JPEG file
        rgb_image.save(temp_file, "JPEG", quality=95)
        return extract_facial_data(temp_file)

@app.task
def download_extract_facial_data(jpeg_url):
    pass

@app.task
def extract_facial_data(jpeg_path):
    # Extract embeddings for all faces found in the image
    try:
        embedding_objects = DeepFace.represent(
            img_path=jpeg_path,
            model_name="ArcFace",         # Alternatives: "VGG-Face", "ArcFace", "OpenFace"
            detector_backend="retinaface",
            enforce_detection=True
        )

        return embedding_objects

    except Exception as e:
        print(f"An error occurred: {e}")

___all__ = ['app']

if __name__ == '__main__':
    # Equivalent to calling the celery worker command line tool
    app.worker_main(argv=['worker', '--loglevel=INFO', '--concurrency=4'])