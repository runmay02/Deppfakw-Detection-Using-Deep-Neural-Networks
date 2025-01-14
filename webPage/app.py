from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import EfficientNetGRU, load_model, predict_from_cropped_images

# Initialize Flask app
app = Flask(__name__)

# App configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_FRAMES_FOLDER'] = 'extracted_frames'
app.config['CROPPED_FACES_FOLDER'] = 'face_cropped'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FRAMES_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPPED_FACES_FOLDER'], exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "celebDF": load_model(EfficientNetGRU(sequence_length=10), "./models/best_model_epoch_celebDF3.pth", device),
    "dfdc": load_model(EfficientNetGRU(sequence_length=10), "./models/best_model_epoch_dfd3.pth", device)
}



def predict_from_video(cropped_faces_dir, model_choice, model_celebdf, model_dfdc, device):
    # Set the model to evaluation mode
    model = model_celebdf if model_choice == "celebDF" else model_dfdc
    model.eval()

    # Prepare image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the cropped face images
    face_files = sorted(os.listdir(cropped_faces_dir))[:10]  # Only use the first 10 faces
    face_images = []

    for face_file in face_files:
        face_path = os.path.join(cropped_faces_dir, face_file)
        image = Image.open(face_path).convert("RGB")
        image = transform(image)
        face_images.append(image)

    # Stack images into a batch
    face_images = torch.stack(face_images).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(face_images)
        _, predicted_class = torch.max(outputs, 1)

    return "Fake" if predicted_class.item() == 1 else "Real"


def process_video(video_path):
    # Create directories for frames and cropped faces if they don't exist
    extracted_frames_dir = "static/extracted_frames"
    extracted_faces_dir = "static/extracted_faces"
    os.makedirs(extracted_frames_dir, exist_ok=True)
    os.makedirs(extracted_faces_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // 10, 1)  # Extract 10 frames

    frame_files = []
    face_files = []

    # Extract exactly 10 frames from the video
    for i in range(10):  # Extract 10 frames
        frame_position = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        if ret:
            # Save frame
            frame_filename = f"frame_{i}.jpg"
            frame_path = os.path.join(extracted_frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_filename)

            # Detect and crop faces from the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Crop faces (limit to 1 face per frame to ensure exactly 10 faces)
            for (x, y, w, h) in faces[:1]:  # Only process the first face in each frame
                cropped_face = frame[y:y + h, x:x + w]
                cropped_face_resized = cv2.resize(cropped_face, (224, 224))  # Resize for model input
                face_filename = f"face_{i}.jpg"
                face_path = os.path.join(extracted_faces_dir, face_filename)
                cv2.imwrite(face_path, cropped_face_resized)
                face_files.append(face_filename)

        # Break if we've already saved 10 faces
        if len(face_files) >= 10:
            break

    cap.release()

    return extracted_frames_dir, extracted_faces_dir, frame_files, face_files


def preprocess_and_display_faces(video_path, frame_dir="extracted_frames", cropped_face_dir="face_cropped", num_frames=10):
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(cropped_face_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)
    frames_to_process = []

    for i in range(num_frames):
        frame_position = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        if ret:
            frames_to_process.append(frame)
            frame_filename = os.path.join(frame_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
        else:
            break

    cap.release()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cropped_faces = []

    for idx, frame in enumerate(frames_to_process):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (224, 224))
            cropped_faces.append(cropped_face)

            cropped_face_path = os.path.join(cropped_face_dir, f"face_{idx}.jpg")
            cv2.imwrite(cropped_face_path, cropped_face)

            if len(cropped_faces) >= num_frames:
                break

        if len(cropped_faces) >= num_frames:
            break

    return len(cropped_faces)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded video and model choice
    video = request.files['video']
    model_choice = request.form['model_choice']
    
    # Save the video to a temporary location
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    # Load the models (you need to have the models loaded when starting the app)
    model_celebdf = load_model(EfficientNetGRU(), 'models/best_model_epoch_celebDF3.pth', device)
    model_dfdc = load_model(EfficientNetGRU(), 'models/best_model_epoch_dfd3.pth', device)

    # Process the video (extract frames and faces)
    extracted_frames_dir, extracted_faces_dir, extracted_frame_files, cropped_face_files = process_video(video_path)

    # Get the prediction
    prediction = predict_from_video(extracted_faces_dir, model_choice, model_celebdf, model_dfdc, device)

    # Render the result template
    return render_template('result.html', prediction=prediction, 
                           model_choice=model_choice,
                           extracted_frames=extracted_frame_files,
                           cropped_faces=cropped_face_files)

if __name__ == '__main__':
    app.run(debug=True)
