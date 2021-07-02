from google.cloud import storage
import os
import glob
storage_client = storage.Client()

def gcs_blob_and_bucket(gcs_url):
    gcs_loc = gcs_url.split('gs://')[1]
    bucket_name = gcs_loc.split('/')[0]
    blob_loc = '/'.join(gcs_loc.split('/')[1:])
    bucket = storage_client.bucket(bucket_name)

    return bucket, blob_loc

def download_from_gcs(gcs_url, fname):
    """Downloads from gcs url (eg. gs://example-bucket/blob) and saves to fname"""
    bucket, blob_loc = gcs_blob_and_bucket(gcs_url)
    blob = bucket.blob(blob_loc)
    blob.download_to_filename(fname)


def upload_blob(gcs_url, fname):
    """Uploads a file to the bucket."""
    bucket, blob_loc = gcs_blob_and_bucket(gcs_url)
    blob = bucket.blob(blob_loc)
    blob.upload_from_filename(fname)

def upload_local_directory_to_gcs(local_path, gcs_url):
    bucket, blob_loc = gcs_blob_and_bucket(gcs_url)

    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
           upload_local_directory_to_gcs(local_file, bucket, blob_loc + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(blob_loc, local_file[1 + len(local_path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)

def download_directory_from_gcs(gcs_url, local_path):
    bucket, blob_loc = gcs_blob_and_bucket(gcs_url)
    if not os.path.isdir(local_path):
        os.makedirs(local_path)

    blobs = bucket.list_blobs(prefix=blob_loc)  # get list of files in dir
    for blob in blobs:
        filename = os.path.join(local_path, blob.name[1+len('susubert/susubert-model'):])
        blob.download_to_filename(filename)  # download
