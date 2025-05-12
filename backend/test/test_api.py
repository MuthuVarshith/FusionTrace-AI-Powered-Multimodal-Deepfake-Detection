
import requests
import os

def test_audio_detect_endpoint(audio_file_path):
    """
    Test the /audio/detect endpoint by uploading an audio file.
    """
    url = "http://localhost:8000/audio/detect"
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file {audio_file_path} does not exist")
        return
    
    with open(audio_file_path, "rb") as f:
        files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json()
                print("Success!")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Saved WAV Path: {result['saved_wav_path']}")
            else:
                print(f"Error: {response.status_code} - {response.json()['detail']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Replace with the path to a sample .wav or .mp3 file
    sample_audio_path = "path/to/sample/audio.wav"
    test_audio_detect_endpoint(sample_audio_path)