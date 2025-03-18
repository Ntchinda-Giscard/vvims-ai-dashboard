import boto3
import os


def download_s3_bucket(bucket_name, local_folder, prefix=""):
    """
    Download all objects from an S3 bucket into a local folder.

    :param bucket_name: Name of the S3 bucket.
    :param local_folder: Path to the local directory to store the files.
    :param prefix: (Optional) S3 prefix to filter specific folder.
    """

    s3 = boto3.client(
        's3',
        aws_access_key_id='AKIAR2BMOVON3NQAL2UV',
        aws_secret_access_key='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',
        region_name='eu-north-1'
    )
    paginator = s3.get_paginator('list_objects_v2')

    # Ensure the local folder exists
    os.makedirs(local_folder, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            print(f"No files found in bucket {bucket_name} with prefix '{prefix}'.")
            return

        for obj in page['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_folder, s3_key.lstrip(prefix).lstrip('/'))

            # Ensure parent directories exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Downloading {s3_key} to {local_file_path}...")
            s3.download_file(bucket_name, s3_key, local_file_path)

    print("Download completed!")


# Example Usage
download_s3_bucket("vvims", "./data")