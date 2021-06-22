from google.cloud import storage
storage_client = storage.Client()


def download_from_gcs(gcs_url, fname):
    """Downloads from gcs url (eg. gs://example-bucket/blob) and saves to fname"""
    gcs_loc = gcs_url.split('gs://')[1]
    bucket_name = gcs_loc.split('/')[0]
    blob_loc = '/'.join(gcs_loc.split('/')[1:])

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_loc)
    blob.download_to_filename(fname)
