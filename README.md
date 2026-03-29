🎯 Real-Time Emotion Coach
Multimodal Emotion Recognition for Interview Coaching
📌 Overview
This project presents a lightweight multimodal emotion recognition system integrated with a real-time interview coaching application.

The system combines physiological signal analysis (WESAD dataset) with real-time behavioral analysis (facial expressions and speech) to provide instant feedback during interviews.

🚀 Features
🎥 Real-time video capture using webcam

🎤 Audio input processing using microphone

🙂 Facial emotion detection (DeepFace / FER)

🧠 Behavioral analysis (face, posture, speech cues)

🤖 Machine Learning models (Random Forest, SVM, Logistic Regression)

⚡ Low-latency real-time inference (~50–150 ms)

📊 Confidence score for predictions

💡 Instant feedback generation

🧠 System Architecture
The system follows a Hybrid Monolithic Architecture:

🔹 Offline Module
WESAD dataset processing

Feature extraction (EDA, BVP, Temperature, ACC)

Model training & optimization

🔹 Real-Time Module
Live video & audio capture

Feature extraction

Emotion prediction

Feedback generation

⚙️ Technologies Used
Python – Core implementation

OpenCV – Video capture & display

MediaPipe – Face detection & landmarks

DeepFace / FER – Emotion recognition

Scikit-learn – ML models

NumPy, Pandas – Data processing

📊 Evaluation Metrics
✔ Traditional Metrics
Accuracy: ~87.5%

Precision: ~86.2%

Recall: ~85.8%

F1 Score: ~86.0%

LOSO Cross-Validation

✔ Real-Time Metrics
Latency: ~50–150 ms

Confidence Score

🧪 Dataset
WESAD (Wearable Stress and Affect Detection Dataset)

Signals used:

Electrodermal Activity (EDA)

Blood Volume Pulse (BVP)

Skin Temperature

Accelerometer (ACC)

🎬 How to Run
# Clone the repository
git clone https://github.com/your-username/real-time-emotion-coach.git

# Navigate to folder
cd real-time-emotion-coach

# Install dependencies
pip install -r requirements.txt

# Run the application
python coach.py
🎯 Use Cases
💼 Interview preparation & coaching

🧠 Behavioral analysis

📈 Soft skills improvement

🏥 Mental state monitoring (future scope)

🔥 Key Highlights
Lightweight ML model (no GPU required)

Real-time performance

Multimodal analysis (physiological + behavioral)

Practical real-world application

🚀 Future Work
Integration with wearable devices

Lightweight deep learning models

Real-world dataset expansion

Personalized emotion detection
