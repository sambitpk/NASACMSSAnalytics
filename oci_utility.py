import oci

def fetch_from_oci(bucket_name, object_name, download_path):
    # Load OCI configuration
    config = oci.config.from_file("C:/Users/sambi/.oci/config.txt")  # Path to config file

    # Initialize Object Storage client
    object_storage = oci.object_storage.ObjectStorageClient(config)

    # Get namespace
    namespace = object_storage.get_namespace().data

    # Fetch the object
    try:
        print(f"Fetching {object_name} from bucket {bucket_name}...")
        obj = object_storage.get_object(namespace, bucket_name, object_name)
        
        # Save file locally
        with open(download_path, 'wb') as f:
            f.write(obj.data.content)
        print(f"File downloaded successfully to {download_path}.")
    except oci.exceptions.ServiceError as e:
        print(f"Error fetching file: {e}")

# Example Usage
bucket_name = "bucket-20241222-2100_DataanalysisTEST"
object_name = "train_FD001.txt"
download_path = "D:/LLM Learing/NASA TURBOFAN CASE STUDY/OCI/train"

fetch_from_oci(bucket_name, object_name, download_path)


