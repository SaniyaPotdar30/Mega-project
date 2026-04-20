import os
import requests

def download_model():
    url = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
    output_filename = "frozen_east_text_detection.pb"
    
    if os.path.exists(output_filename):
        print(f"[{output_filename}] already exists! No need to download.")
        return

    print("Downloading EAST model (~90MB)... This may take a moment.")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Download completed successfully!")
    except Exception as e:
        print(f"Failed to download the model. Error: {e}")
        print("\nPlease manually download the 'frozen_east_text_detection.pb' model from a reliable source and save it in this directory.")

if __name__ == "__main__":
    download_model()
