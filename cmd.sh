zenml secret create s3_secret \
    --aws_access_key_id='AKIAR2BMOVON3NQAL2UV' \
    --aws_secret_access_key='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'

zenml artifact-store register s3_store -f s3 \
    --path='s3://vvims-ai' \
    --authentication_secret=s3_secret

zenml integration install evidently -y