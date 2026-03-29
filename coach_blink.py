"""
Real-Time Interview Emotion Coach  v4 FINAL  (Demo Ready)
===========================================================
Controls:
  I = toggle interview question mode
  N = next question
  R = start/stop recording
  Q = quit + show report card

Blink Detection (5-second rolling window):
  >= 3 blinks in 5s  →  "Blinking too fast — calm down!"
  0 blinks in 10s    →  "Why are you staring? Blink naturally!"
  1-2 blinks in 5s   →  "Blink rate normal"

Speech WPM (fixed):
  Measures actual audio duration instead of assuming full chunk length.
  Fast threshold raised to 180 WPM so normal speakers are not wrongly flagged.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import random
import os
from collections import deque

try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_OK = True
    except ImportError:
        PDF_OK = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_OK = True
except ImportError:
    PLT_OK = False

try:
    import speech_recognition as sr
    SPEECH_OK = True
except ImportError:
    SPEECH_OK = False

try:
    from fer import FER
    EMOTION_BACKEND = "fer"
except ImportError:
    try:
        from deepface import DeepFace
        EMOTION_BACKEND = "deepface"
    except ImportError:
        EMOTION_BACKEND = None

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CAM_INDEX        = 0
WINDOW_NAME      = "Interview Coach  |  Press Q to quit"
EMOTION_INTERVAL = 1.0
SMOOTHING_FRAMES = 6
CHECKLIST_SECS   = 12
PANEL_W          = 340
QUESTION_TIME    = 90

# ── Blink thresholds ──────────────────────────────────────────────────────────
BLINK_WINDOW_SECS = 5    # rolling window for fast-blink check
BLINK_FAST_COUNT  = 3    # >= 3 blinks in 5s  → nervous alert
STARE_WINDOW_SECS = 10   # zero blinks in 10s → staring alert
EAR_THRESH        = 0.18
BLINK_CONSEC      = 2

# ── Speech WPM thresholds ─────────────────────────────────────────────────────
WPM_IDEAL_LOW  = 100   # below this = too slow
WPM_IDEAL_HIGH = 180   # raised from 150 — normal conversational speed is ~140-160
WPM_CHUNK_SECS = 5     # max audio chunk length

INTERVIEW_QUESTIONS = [
    "Tell me about yourself and your background.",
    "What is your greatest strength?",
    "What is your biggest weakness?",
    "Why do you want this job?",
    "Where do you see yourself in 5 years?",
    "Describe a challenge you faced and how you overcame it.",
    "Why should we hire you over other candidates?",
    "Tell me about a time you worked in a team.",
    "How do you handle stress and pressure?",
    "What motivates you to do your best work?",
    "Describe your ideal work environment.",
    "Tell me about a project you are proud of.",
    "How do you prioritize tasks when you have multiple deadlines?",
    "What are your salary expectations?",
    "Do you have any questions for us?",
]

C_GREEN  = (80,  210,  90)
C_YELLOW = (40,  210, 210)
C_RED    = (70,   70, 220)
C_WHITE  = (240, 240, 240)
C_ACCENT = (60,  180, 240)
C_DIM    = (120, 120, 130)
C_BLACK  = (15,   15,  20)
C_PANEL  = (22,   22,  32)
C_ORANGE = (40,  160, 255)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

POSITIVE_EM = {"happy", "surprise"}
NEGATIVE_EM = {"angry", "disgust", "fear", "sad"}

CHECKLIST = [
    ("LIGHTING",    "Light should be in FRONT of you, not behind"),
    ("CAMERA",      "Position camera at eye level, not up or down"),
    ("BACKGROUND",  "Use a clean, plain background if possible"),
    ("POSTURE",     "Sit upright, shoulders back, spine straight"),
    ("EYE CONTACT", "Look into the camera lens, not at your face"),
    ("EXPRESSION",  "Relax -- aim for a natural, warm expression"),
    ("DISTANCE",    "Sit 50-70 cm away, face fills the frame"),
    ("NOISE",       "Find a quiet place, mute phone notifications"),
]

# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE
# ══════════════════════════════════════════════════════════════════════════════
mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose           = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose_detector = mp_pose.Pose(
    static_image_mode=False, model_complexity=0,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

if EMOTION_BACKEND == "fer":
    emotion_detector = FER(mtcnn=False)

LEFT_IRIS_CENTER = 468;  RIGHT_IRIS_CENTER = 473
LEFT_EYE_OUTER   = 33;   LEFT_EYE_INNER   = 133
RIGHT_EYE_INNER  = 362;  RIGHT_EYE_OUTER  = 263
MOUTH_LEFT = 61;  MOUTH_RIGHT = 291
L_EYE_TOP  = 159; L_EYE_BOT  = 145
R_EYE_TOP  = 386; R_EYE_BOT  = 374

MODEL_3D = np.array([
    [0.0,      0.0,    0.0  ],
    [-225.0,  170.0, -135.0 ],
    [-150.0, -150.0, -125.0 ],
    [0.0,    -330.0,  -65.0 ],
    [225.0,   170.0, -135.0 ],
    [150.0,  -150.0, -125.0 ],
], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class SessionTracker:
    def __init__(self):
        self.confidence_history = []
        self.smile_history      = []
        self.gaze_history       = []
        self.posture_history    = []
        self.wpm_history        = []
        self.emotion_counts     = {}
        self.start_time         = None
        self.frame_count        = 0
        self.question_log       = []
        self._cur_q             = None

    def start(self):
        self.start_time = time.time()

    def start_question(self, text):
        self._save_cur()
        self._cur_q = {"text": text, "confs": [], "smiles": [], "gazes": []}

    def _save_cur(self):
        if self._cur_q and self._cur_q["confs"]:
            d = self._cur_q
            self.question_log.append({
                "text":      d["text"],
                "avg_conf":  int(np.mean(d["confs"])),
                "avg_smile": int(np.mean([1 if s > 0.15 else 0 for s in d["smiles"]]) * 100),
                "avg_gaze":  int(np.mean(d["gazes"]) * 100),
            })
        self._cur_q = None

    def finish(self):
        self._save_cur()

    def record(self, confidence, smile, gaze_l, gaze_r, posture_ok, wpm, emotion="neutral"):
        if self.start_time is None:
            return
        t = time.time() - self.start_time
        self.confidence_history.append((t, confidence))
        self.smile_history.append(smile)
        gaze_ok = 0.28 < (gaze_l + gaze_r) / 2 < 0.72
        self.gaze_history.append(1 if gaze_ok else 0)
        self.posture_history.append(1 if posture_ok else 0)
        if wpm > 0:
            self.wpm_history.append(wpm)
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
        self.frame_count += 1
        if self._cur_q is not None:
            self._cur_q["confs"].append(confidence)
            self._cur_q["smiles"].append(smile)
            self._cur_q["gazes"].append(1 if gaze_ok else 0)

    def dominant_emotion(self):
        if not self.emotion_counts: return "neutral"
        return max(self.emotion_counts, key=self.emotion_counts.get)

    def emotion_pct(self, emotion):
        total = sum(self.emotion_counts.values())
        return int(self.emotion_counts.get(emotion, 0) / total * 100) if total else 0

    def stress_pct(self):
        neg   = sum(self.emotion_counts.get(e, 0) for e in ["angry", "disgust", "fear", "sad"])
        total = sum(self.emotion_counts.values())
        return int(neg / total * 100) if total else 0

    def best_worst_questions(self):
        if not self.question_log: return None, None
        srt = sorted(self.question_log, key=lambda x: x["avg_conf"])
        return srt[-1], srt[0]

    def duration_secs(self):
        return int(time.time() - self.start_time) if self.start_time else 0

    def avg_confidence(self):
        return int(np.mean([v for _, v in self.confidence_history])) if self.confidence_history else 0

    def pct_smiling(self):
        return int(np.mean([1 if s > 0.15 else 0 for s in self.smile_history]) * 100) if self.smile_history else 0

    def pct_eye_contact(self):
        return int(np.mean(self.gaze_history) * 100) if self.gaze_history else 0

    def pct_good_posture(self):
        return int(np.mean(self.posture_history) * 100) if self.posture_history else 100

    def avg_wpm(self):
        return int(np.mean(self.wpm_history)) if self.wpm_history else 0

    def grade(self):
        avg = self.avg_confidence()
        if avg >= 78: return "A", C_GREEN
        if avg >= 60: return "B", C_GREEN
        if avg >= 45: return "C", C_YELLOW
        return "D", C_RED


session_tracker = SessionTracker()

# ══════════════════════════════════════════════════════════════════════════════
# RESUME + QUESTION GENERATION
# ══════════════════════════════════════════════════════════════════════════════
RESUME_TEMPLATES = {
    "python":          "Walk me through a Python project you are proud of.",
    "java":            "Tell me about your Java development experience.",
    "machine learning":"How have you applied machine learning in a real project?",
    "deep learning":   "Explain a deep learning project you have worked on.",
    "data":            "Tell me about a data analysis project you completed.",
    "sql":             "Give an example of a complex SQL query you have written.",
    "react":           "Walk me through a React component you built.",
    "node":            "How have you handled asynchronous operations in Node.js?",
    "aws":             "Which AWS services have you used and for what purpose?",
    "cloud":           "Tell me about a cloud deployment you managed.",
    "project":         "Tell me about the most challenging project you have led.",
    "team":            "How do you handle disagreements with teammates?",
    "intern":          "What did you learn during your internship?",
    "research":        "How do you approach a new research problem?",
    "leadership":      "Give an example of when you demonstrated leadership.",
    "srm":             "How has your time at SRM prepared you for industry?",
}

DEFAULT_QUESTIONS = [
    "Walk me through your resume from start to finish.",
    "What is the most technically challenging project on your resume?",
    "Why did you choose your current field of study?",
    "What skills do you bring that are not obvious from your resume?",
    "How have your past experiences prepared you for this role?",
    "Describe the biggest impact you have had in a previous role.",
    "Where do you see yourself in 5 years?",
]

def extract_resume_text(pdf_path):
    if not PDF_OK or not pdf_path: return ""
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.lower()
    except Exception as e:
        print(f"[WARN] PDF read error: {e}"); return ""

def generate_resume_questions(resume_text):
    if not resume_text: return DEFAULT_QUESTIONS.copy()
    qs = []
    for kw, q in RESUME_TEMPLATES.items():
        if kw in resume_text: qs.append(q)
    qs += random.sample(DEFAULT_QUESTIONS, min(3, len(DEFAULT_QUESTIONS)))
    random.shuffle(qs)
    return qs if qs else DEFAULT_QUESTIONS.copy()

def find_resume_pdf():
    for f in os.listdir("."):
        if f.lower().endswith(".pdf"): return f
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        print("[INFO] Opening file picker — select resume PDF (Cancel to skip)...")
        path = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        root.destroy()
        return path if path else None
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# BLINK TRACKER  ── 5-second window, demo-ready
# ══════════════════════════════════════════════════════════════════════════════
class BlinkTracker:
    """
    Two independent alert rules:
      FAST  : >= BLINK_FAST_COUNT (3) blinks inside BLINK_WINDOW_SECS (5 s)
               → "Blinking too fast (X/5s) — calm down!"
      STARE : 0 blinks at all inside STARE_WINDOW_SECS (10 s)
               → "Why are you staring? Blink naturally!"
    """
    def __init__(self):
        self.counter       = 0           # consecutive closed frames
        self.total         = 0           # lifetime blink count
        self.blink_times   = deque()     # timestamps of every confirmed blink
        self.session_start = time.time()

    def update(self, ear):
        if ear < EAR_THRESH:
            self.counter += 1
        else:
            if self.counter >= BLINK_CONSEC:
                self.total += 1
                self.blink_times.append(time.time())
            self.counter = 0
        # Prune entries older than our longest window + 1 s buffer
        cutoff = time.time() - max(BLINK_WINDOW_SECS, STARE_WINDOW_SECS) - 1
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    def blinks_in(self, secs):
        now = time.time()
        return sum(1 for t in self.blink_times if now - t <= secs)

    def status(self):
        """
        Returns (count_5s, label, colour).
        Priority order:
          1. < 2s elapsed  → calibrating, silent
          2. >= 3 blinks/5s → fast / nervous
          3. 0 blinks/10s AND session >= 10s → staring
          4. else           → normal
        """
        elapsed   = time.time() - self.session_start
        if elapsed < 2.0:
            return 0, "Calibrating...", C_DIM

        count_5s  = self.blinks_in(BLINK_WINDOW_SECS)
        count_10s = self.blinks_in(STARE_WINDOW_SECS)

        if count_5s >= BLINK_FAST_COUNT:
            return count_5s, f"Blinking too fast ({count_5s}/5s) -- calm down!", C_RED

        if count_10s == 0 and elapsed >= STARE_WINDOW_SECS:
            return 0, "Why are you staring? Blink naturally!", C_YELLOW

        return count_5s, "Blink rate normal", C_GREEN


blink_tracker = BlinkTracker()

# ══════════════════════════════════════════════════════════════════════════════
# SPEECH MONITOR  ── fixed WPM (actual duration, not assumed chunk length)
# ══════════════════════════════════════════════════════════════════════════════
class SpeechMonitor:
    """
    FIX 1: WPM is now calculated from the ACTUAL wall-clock duration of each
            audio chunk, not the assumed WPM_CHUNK_SECS constant.
            Previously: 10 words assumed over 5s = 120 WPM (correct)
                        10 words spoken in 2s but assumed 5s = 120 WPM (WRONG, should be 300)
            Now:        10 words spoken in 2s = 10/2*60 = 300 WPM (correct)

    FIX 2: Fast threshold raised to WPM_IDEAL_HIGH (180) so normal
            conversational speakers (~140-160 WPM) are not wrongly flagged.
    """
    def __init__(self):
        self.wpm      = 0
        self.label    = "Listening..."
        self.col      = C_DIM
        self.lock     = threading.Lock()
        self.running  = False
        self.wpm_hist = deque(maxlen=4)   # smooth over last 4 chunks

    def start(self):
        if not SPEECH_OK: return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[SPEECH] Microphone monitoring started.")

    def stop(self):
        self.running = False

    def _loop(self):
        rec = sr.Recognizer()
        rec.energy_threshold         = 300
        rec.dynamic_energy_threshold = True
        rec.pause_threshold          = 0.8

        with sr.Microphone() as src:
            rec.adjust_for_ambient_noise(src, duration=1)
            while self.running:
                try:
                    t_start = time.time()                          # wall clock start
                    audio   = rec.listen(src,
                                         timeout=2,
                                         phrase_time_limit=WPM_CHUNK_SECS)
                    t_end   = time.time()                          # wall clock end

                    text         = rec.recognize_google(audio)
                    word_count   = len(text.split())
                    actual_secs  = max(0.5, t_end - t_start)      # real duration
                    wpm_chunk    = int(word_count / actual_secs * 60)

                    with self.lock:
                        self.wpm_hist.append(wpm_chunk)
                        self.wpm = int(sum(self.wpm_hist) / len(self.wpm_hist))
                        self._update_label()

                except sr.WaitTimeoutError:
                    with self.lock:
                        self.wpm   = 0
                        self.label = "Not speaking"
                        self.col   = C_DIM
                except sr.UnknownValueError:
                    pass
                except Exception:
                    pass

    def _update_label(self):
        if self.wpm == 0:
            self.label = "Not speaking";                         self.col = C_DIM
        elif self.wpm < WPM_IDEAL_LOW:
            self.label = f"Too slow ({self.wpm} WPM)";          self.col = C_YELLOW
        elif self.wpm > WPM_IDEAL_HIGH:
            self.label = f"Too fast ({self.wpm} WPM)";          self.col = C_RED
        else:
            self.label = f"Good pace ({self.wpm} WPM)";         self.col = C_GREEN

    def get(self):
        with self.lock:
            return self.wpm, self.label, self.col

    def draw(self, frame, fh, cam_w):
        wpm, lbl, col = self.get()
        bx = 8; bh = 120
        by1 = fh // 2 - bh // 2; by2 = by1 + bh
        cv2.rectangle(frame, (bx, by1), (bx+6, by2), (40,40,50), -1)
        if wpm > 0:
            fh2 = int(min(1, wpm / 250.0) * bh)
            cv2.rectangle(frame, (bx, by2-fh2), (bx+6, by2), col, -1)
        lo_y = by2 - int((WPM_IDEAL_LOW  / 250.0) * bh)
        hi_y = by2 - int((WPM_IDEAL_HIGH / 250.0) * bh)
        cv2.line(frame, (bx, lo_y), (bx+10, lo_y), C_GREEN, 1)
        cv2.line(frame, (bx, hi_y), (bx+10, hi_y), C_GREEN, 1)
        cv2.putText(frame, "WPM", (bx-2, by1-6), FONT, 0.28, C_DIM, 1, cv2.LINE_AA)
        if wpm > 0:
            cv2.putText(frame, str(wpm), (bx-4, by2+14), FONT, 0.38, col, 1, cv2.LINE_AA)
        if lbl and lbl not in ("Not speaking", "Listening..."):
            lw = cv2.getTextSize(lbl, FONT, 0.40, 1)[0][0]
            lx = max(20, min(cam_w - lw - 10, (cam_w - lw) // 2))
            cv2.putText(frame, lbl, (lx, fh-38), FONT, 0.40, col, 1, cv2.LINE_AA)


speech_monitor = SpeechMonitor()

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class QuestionManager:
    def __init__(self):
        self.active      = False
        self.questions   = INTERVIEW_QUESTIONS.copy()
        self.idx         = 0
        self.q_start     = 0.0
        self.current_q   = ""
        self.resume_mode = False

    def set_questions(self, qs):
        if qs:
            self.questions   = qs
            self.resume_mode = True
            print(f"[INFO] {len(qs)} resume questions loaded.")

    def start(self):
        self.active    = True
        self.idx       = 0
        random.shuffle(self.questions)
        self.current_q = self.questions[0]
        self.q_start   = time.time()
        session_tracker.start_question(self.current_q)

    def stop(self):
        self.active = False

    def next_question(self):
        self.idx       = (self.idx + 1) % len(self.questions)
        self.current_q = self.questions[self.idx]
        self.q_start   = time.time()
        session_tracker.start_question(self.current_q)

    def time_remaining(self):
        return max(0, QUESTION_TIME - (time.time() - self.q_start))

    def update(self):
        if self.active and self.time_remaining() <= 0:
            self.next_question(); return True
        return False

    def draw(self, frame, fw, fh):
        if not self.active: return
        rem   = self.time_remaining()
        q_num = self.idx + 1
        total = len(self.questions)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (fw-PANEL_W, 110), (10,12,20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, f"Q{q_num}/{total}", (14,22), FONTB, 0.55, C_ACCENT, 1, cv2.LINE_AA)
        bx1, bx2 = 70, fw-PANEL_W-14
        prog  = rem / QUESTION_TIME
        t_col = C_GREEN if prog > 0.5 else (C_YELLOW if prog > 0.25 else C_RED)
        cv2.rectangle(frame, (bx1,10), (bx2,20), (40,40,50), -1)
        cv2.rectangle(frame, (bx1,10), (bx1+int(prog*(bx2-bx1)),20), t_col, -1)
        cv2.putText(frame, f"{int(rem)}s", (bx2+6,20), FONT, 0.40, t_col, 1, cv2.LINE_AA)
        words = self.current_q.split(); lines = []; line = ""
        for w in words:
            test = (line+" "+w).strip()
            if len(test) <= 60: line = test
            else:
                if line: lines.append(line); line = w
        if line: lines.append(line)
        y_text = 44
        for ln in lines[:3]:
            cv2.putText(frame, ln, (14,y_text), FONTB, 0.52, C_WHITE, 1, cv2.LINE_AA)
            y_text += 22
        cv2.putText(frame, "N=next   I=exit", (14,100), FONT, 0.30, C_DIM, 1, cv2.LINE_AA)


question_mgr = QuestionManager()

# ══════════════════════════════════════════════════════════════════════════════
# RECORDING MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class RecordingManager:
    def __init__(self):
        self.recording = False; self.writer = None
        self.filename = ""; self.start_t = 0.0; self.frame_count = 0

    def start(self, fw, fh):
        ts = time.strftime("%Y%m%d_%H%M%S"); self.filename = f"interview_{ts}.mp4"
        self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw,fh))
        self.recording = True; self.start_t = time.time(); self.frame_count = 0
        print(f"[REC] Started: {self.filename}")

    def stop(self):
        if self.writer: self.writer.release(); self.writer = None
        self.recording = False
        print(f"[REC] Saved: {self.filename} ({self.frame_count} frames)")

    def write(self, frame):
        if self.recording and self.writer:
            self.writer.write(frame); self.frame_count += 1

    def draw_indicator(self, frame, fw, fh):
        if not self.recording: return
        m, s = divmod(int(time.time()-self.start_t), 60)
        cv2.putText(frame, f"{m:02d}:{s:02d}", (fw-PANEL_W-110,23), FONT, 0.38, C_RED, 1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (fw-PANEL_W-55,23), FONTB, 0.45, C_RED, 1, cv2.LINE_AA)
        if int(time.time()*2)%2==0:
            cv2.circle(frame, (fw-PANEL_W-22,18), 7, C_RED, -1)


recorder = RecordingManager()

# ══════════════════════════════════════════════════════════════════════════════
# TOAST MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class ToastManager:
    DURATION = 2.5; FADE_OUT = 0.5; MAX = 3

    def __init__(self):
        self.toasts = []; self.lock = threading.Lock(); self._shown = {}

    def push(self, text, colour, key=None, cooldown=5.0):
        now = time.time()
        if key:
            if now - self._shown.get(key, 0) < cooldown: return
            self._shown[key] = now
        with self.lock:
            for t in self.toasts:
                if t["text"] == text: return
            if len(self.toasts) >= self.MAX: self.toasts.pop(0)
            self.toasts.append({"text": text, "colour": colour, "born": now})

    def draw(self, frame, fw, fh):
        now = time.time()
        with self.lock:
            self.toasts = [t for t in self.toasts if now-t["born"] < self.DURATION]
            y_base = fh-60
            for t in reversed(self.toasts):
                age   = now - t["born"]
                alpha = max(0.0, min(1.0, 1-(age-(self.DURATION-self.FADE_OUT))/self.FADE_OUT))
                text = t["text"]; col = t["colour"]
                (tw,th),_ = cv2.getTextSize(text, FONT, 0.52, 1)
                pad = 10
                bx1 = fw-PANEL_W-tw-pad*2-24; bx2 = fw-PANEL_W-24
                by1 = y_base-th-pad;           by2 = y_base+pad//2
                ov  = frame.copy()
                cv2.rectangle(ov,(bx1,by1),(bx2,by2),C_PANEL,-1)
                cv2.rectangle(ov,(bx1,by1),(bx2,by2),col,1)
                cv2.addWeighted(ov,alpha*0.85,frame,1-alpha*0.85,0,frame)
                tx = frame.copy()
                cv2.putText(tx,text,(bx1+pad,y_base),FONT,0.52,col,1,cv2.LINE_AA)
                cv2.addWeighted(tx,alpha,frame,1-alpha,0,frame)
                y_base -= (th+pad*2+6)


toast_mgr = ToastManager()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def lm_px(landmarks, idx, w, h):
    lm = landmarks[idx]; return int(lm.x*w), int(lm.y*h)

def compute_head_pose(landmarks, w, h):
    pts = np.array([lm_px(landmarks,i,w,h) for i in [1,33,61,199,263,291]], dtype=np.float64)
    cam_m = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    ok,rvec,_ = cv2.solvePnP(MODEL_3D,pts,cam_m,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    rmat,_ = cv2.Rodrigues(rvec); fwd = rmat[:,2]
    yaw   = float(np.clip(np.degrees(np.arctan2(fwd[0],abs(fwd[2]))),-90,90))
    pitch = float(np.clip(np.degrees(np.arctan2(fwd[1],abs(fwd[2]))),-90,90))
    roll  = float(np.clip(np.degrees(np.arctan2(-rmat[0,1],rmat[0,0])),-90,90))
    return yaw, pitch, roll

def compute_gaze(landmarks, w, h):
    def ratio(outer, inner, iris):
        ox,_ = lm_px(landmarks,outer,w,h); ix,_ = lm_px(landmarks,inner,w,h)
        px,_ = lm_px(landmarks,iris,w,h);  span = abs(ix-ox)
        return (px-min(ox,ix))/span if span else 0.5
    return (ratio(LEFT_EYE_OUTER,LEFT_EYE_INNER,LEFT_IRIS_CENTER),
            ratio(RIGHT_EYE_INNER,RIGHT_EYE_OUTER,RIGHT_IRIS_CENTER))

def compute_smile(landmarks):
    nose=landmarks[1]; chin=landmarks[152]; ml=landmarks[61]; mr=landmarks[291]
    nc=abs(chin.y-nose.y)+1e-5; cy=(ml.y+mr.y)/2.0; mid=(nose.y+chin.y)/2.0
    return float(np.clip((mid-cy)/nc/0.12,0.0,1.0))

def eye_aspect_ratio(landmarks, top, bot, inner, outer):
    vert  = abs(landmarks[top].y - landmarks[bot].y)
    horiz = abs(landmarks[inner].x - landmarks[outer].x) + 1e-5
    return vert / horiz

def compute_confidence(yaw, pitch, gaze_l, gaze_r, smile, emotion, blink_count_5s):
    pose_s  = max(0,1-abs(yaw)/30) * max(0,1-abs(pitch)/30)
    gaze_s  = 1 - min(1,abs((gaze_l+gaze_r)/2-0.5)/0.25)
    smile_s = min(1,smile/0.5)
    em_map  = {"happy":1.0,"surprise":0.7,"neutral":0.5,"sad":0.2,"fear":0.1,"angry":0.1,"disgust":0.1}
    em_s    = em_map.get(emotion,0.5)
    if 1 <= blink_count_5s <= 2:   blink_s = 1.0
    elif blink_count_5s == 0:      blink_s = 0.4
    else:                          blink_s = max(0, 1.0-(blink_count_5s-2)*0.3)
    return round(min(100, pose_s*28 + gaze_s*22 + smile_s*20 + em_s*15 + blink_s*10 + 5))

def run_emotion(frame):
    try:
        if EMOTION_BACKEND == "fer":
            res = emotion_detector.detect_emotions(frame)
            if res: return max(res[0]["emotions"], key=res[0]["emotions"].get)
        elif EMOTION_BACKEND == "deepface":
            res = DeepFace.analyze(frame,actions=["emotion"],enforce_detection=False,silent=True)
            if isinstance(res,list): res=res[0]
            return res.get("dominant_emotion","neutral")
    except Exception: pass
    return "neutral"

def compute_posture(pose_lms, fw, fh):
    if not pose_lms: return True,"",C_DIM
    lms=pose_lms.landmark
    ls=lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs=lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    ns=lms[mp_pose.PoseLandmark.NOSE.value]
    if ls.visibility<0.5 or rs.visibility<0.5: return True,"",C_DIM
    if abs(ls.y-rs.y) > 0.06:      return False,"Uneven shoulders -- sit straight",C_YELLOW
    if (ls.y+rs.y)/2-ns.y < 0.15:  return False,"Sit up -- you are hunching",C_RED
    return True,"Good posture",C_GREEN

def draw_posture_overlay(frame, pose_lms, fw, fh, posture_ok, msg, col):
    if not pose_lms: return
    lms=pose_lms.landmark
    ls=lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs=lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if ls.visibility<0.5 or rs.visibility<0.5: return
    lx,ly=int(ls.x*fw),int(ls.y*fh); rx,ry=int(rs.x*fw),int(rs.y*fh)
    cv2.line(frame,(lx,ly),(rx,ry),col,2,cv2.LINE_AA)
    cv2.circle(frame,(lx,ly),5,col,-1); cv2.circle(frame,(rx,ry),5,col,-1)
    if msg:
        cv2.putText(frame,msg,((lx+rx)//2-120, min(max(ly,ry)+20,fh-40)),FONT,0.5,col,1,cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
# COACHING TOASTS + TIPS
# ══════════════════════════════════════════════════════════════════════════════
def push_toasts(yaw, pitch, roll, gaze_l, gaze_r, smile, emotion,
                posture_ok, blink_count_5s, blink_label, blink_col):
    avg = (gaze_l+gaze_r)/2
    if abs(yaw)>18:
        toast_mgr.push(f"Face fwd -- turning {'right' if yaw>0 else 'left'}!",C_RED,"yaw",4)
    if pitch>14:   toast_mgr.push("Chin up -- you're looking down",C_YELLOW,"pitch_d",5)
    elif pitch<-12:toast_mgr.push("Lower your chin slightly",C_YELLOW,"pitch_u",5)
    if abs(roll)>14: toast_mgr.push("Head tilting -- straighten up",C_YELLOW,"roll",5)
    if avg<0.25 or avg>0.75: toast_mgr.push("Look at the camera!",C_RED,"gaze",4)
    if smile<0.12: toast_mgr.push("Try to smile -- you look tense",C_YELLOW,"smile",6)
    elif smile>0.55: toast_mgr.push("Great smile -- keep it natural!",C_GREEN,"smile_g",8)
    if emotion in NEGATIVE_EM: toast_mgr.push(f"Looks {emotion} -- take a breath",C_RED,"neg_em",6)
    elif emotion in POSITIVE_EM: toast_mgr.push(f"Great energy -- looking {emotion}!",C_GREEN,"pos_em",8)
    if not posture_ok: toast_mgr.push("Sit up straight -- slouching detected",C_YELLOW,"posture",8)
    # Blink toasts — push exactly when rule fires
    if blink_col == C_RED:
        toast_mgr.push(blink_label, C_RED, "blink_fast", 4)
    elif blink_col == C_YELLOW and "staring" in blink_label.lower():
        toast_mgr.push(blink_label, C_YELLOW, "blink_stare", 8)
    # Speech toasts
    wpm,_,_ = speech_monitor.get()
    if wpm > WPM_IDEAL_HIGH:
        toast_mgr.push(f"Slow down! Speaking at {wpm} WPM",C_RED,"speech_fast",10)
    elif 0 < wpm < WPM_IDEAL_LOW:
        toast_mgr.push(f"Speak up! Too slow at {wpm} WPM",C_YELLOW,"speech_slow",10)

def build_tips(yaw, pitch, roll, gaze_l, gaze_r, smile, emotion,
               blink_count_5s, blink_label, blink_col, posture_ok, posture_msg):
    tips = []
    avg  = (gaze_l+gaze_r)/2
    if not posture_ok and posture_msg:
        tips.append(("!",posture_msg[:22],C_YELLOW))
    if abs(yaw)>18: tips.append(("!",f"Face fwd({yaw:+.0f})",C_RED))
    else:           tips.append(("OK","Head position good",C_GREEN))
    if pitch>14:    tips.append(("!","Chin up",C_YELLOW))
    elif pitch<-12: tips.append(("!","Lower chin",C_YELLOW))
    if avg<0.25 or avg>0.75: tips.append(("!","Look at camera",C_RED))
    else:                    tips.append(("OK","Good eye contact",C_GREEN))
    if smile<0.12:  tips.append(("!","Try to smile",C_YELLOW))
    elif smile>0.55:tips.append(("OK","Great smile!",C_GREEN))
    else:           tips.append(("OK","Expression OK",C_GREEN))
    if emotion in NEGATIVE_EM: tips.append(("!",f"Looks {emotion}--relax",C_RED))
    elif emotion in POSITIVE_EM:tips.append(("OK",f"Energy: {emotion}!",C_GREEN))
    else:                       tips.append(("-",f"Mood: {emotion}",C_DIM))
    tips.append(("!" if blink_col!=C_GREEN else "OK", blink_label[:28], blink_col))
    if posture_ok and posture_msg=="Good posture":
        tips.append(("OK","Good posture",C_GREEN))
    return tips

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def filled_rect(img, x1, y1, x2, y2, colour, alpha=1.0):
    if alpha>=1.0: cv2.rectangle(img,(x1,y1),(x2,y2),colour,-1)
    else:
        ov=img.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),colour,-1)
        cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def h_bar(frame, x1, x2, y, ht, value, fg, bg=(50,50,60)):
    cv2.rectangle(frame,(x1,y),(x2,y+ht),bg,-1)
    f=int(max(0,min(1,value))*(x2-x1))
    if f>2: cv2.rectangle(frame,(x1,y),(x1+f,y+ht),fg,-1)

def draw_panel(frame, fw, fh, emotion, smile, yaw, pitch, tips,
               confidence, blink_count_5s, blink_lbl, blink_col):
    px = fw-PANEL_W
    filled_rect(frame,px,0,fw,fh,C_PANEL,alpha=0.85)
    cv2.line(frame,(px,0),(px,fh),C_ACCENT,1)
    y=22
    cv2.putText(frame,"INTERVIEW COACH",(px+10,y),FONTB,0.48,C_ACCENT,1,cv2.LINE_AA)
    if recorder.recording and int(time.time()*2)%2==0:
        cv2.circle(frame,(fw-20,14),5,C_RED,-1)
    y+=5; cv2.line(frame,(px+10,y),(fw-10,y),C_ACCENT,1); y+=14

    conf_col=C_GREEN if confidence>=70 else (C_YELLOW if confidence>=45 else C_RED)
    cv2.putText(frame,"CONFIDENCE",(px+10,y),FONT,0.30,C_DIM,1,cv2.LINE_AA); y+=13
    sw=cv2.getTextSize(f"{confidence}%",FONTB,1.1,2)[0][0]
    cv2.putText(frame,f"{confidence}%",(px+10,y+22),FONTB,1.1,conf_col,2,cv2.LINE_AA)
    desc="Excellent!" if confidence>=80 else("Good" if confidence>=60 else("Fair" if confidence>=40 else"Needs work"))
    cv2.putText(frame,desc,(px+sw+18,y+22),FONT,0.38,conf_col,1,cv2.LINE_AA)
    y+=28; h_bar(frame,px+8,fw-8,y,9,confidence/100.0,conf_col); y+=18
    cv2.line(frame,(px+8,y),(fw-8,y),(50,50,65),1); y+=10

    def badge(lbl,val,good,bx):
        col=C_GREEN if good else C_YELLOW
        cv2.putText(frame,lbl,(bx,y),FONT,0.24,C_DIM,1,cv2.LINE_AA)
        cv2.putText(frame,val,(bx,y+13),FONT,0.34,col,1,cv2.LINE_AA)
    badge("YAW",  f"{yaw:+.0f}d",       abs(yaw)<15,        px+8)
    badge("PITCH",f"{pitch:+.0f}d",     abs(pitch)<12,      px+68)
    badge("SMILE",f"{smile*100:.0f}%",  smile>0.15,         px+132)
    badge("MOOD", emotion.upper()[:7],  emotion not in NEGATIVE_EM, px+195)
    badge("BLINK",f"{blink_count_5s}/5s", 1<=blink_count_5s<=2, px+262)
    y+=28

    wpm,sp_lbl,sp_col=speech_monitor.get()
    cv2.putText(frame,"SPEECH",(px+8,y),FONT,0.24,C_DIM,1,cv2.LINE_AA)
    wpm_str=f"{wpm} WPM" if wpm>0 else "---"
    cv2.putText(frame,wpm_str,(px+8,y+13),FONT,0.34,sp_col if wpm>0 else C_DIM,1,cv2.LINE_AA)
    pace="Good" if WPM_IDEAL_LOW<=wpm<=WPM_IDEAL_HIGH else("Fast!" if wpm>WPM_IDEAL_HIGH else("Slow" if wpm>0 else"Listening"))
    cv2.putText(frame,pace,(px+72,y+13),FONT,0.28,sp_col,1,cv2.LINE_AA); y+=22

    for lbl,val,good in [
        ("Smile",   smile,                           smile>0.15),
        ("Head",    max(0,1-abs(yaw)/45),            abs(yaw)<15),
        ("Blink/5s",min(1,blink_count_5s/2) if blink_count_5s<=2 else max(0,1-(blink_count_5s-2)*0.3),
                    1<=blink_count_5s<=2),
    ]:
        cv2.putText(frame,lbl,(px+8,y),FONT,0.26,C_DIM,1,cv2.LINE_AA); y+=4
        h_bar(frame,px+8,fw-8,y,6,max(0,min(1,val)),C_GREEN if good else C_YELLOW); y+=12

    cv2.line(frame,(px+8,y),(fw-8,y),(50,50,65),1); y+=8
    cv2.putText(frame,blink_lbl[:30],(px+8,y),FONT,0.27,blink_col,1,cv2.LINE_AA); y+=14
    cv2.line(frame,(px+8,y),(fw-8,y),(50,50,65),1); y+=8
    cv2.putText(frame,"LIVE TIPS",(px+8,y),FONT,0.28,C_DIM,1,cv2.LINE_AA); y+=13

    for _,text,col in tips[:6]:
        cv2.circle(frame,(px+14,y-3),3,col,-1)
        words=[]; line=""
        for w in text.split():
            test=(line+" "+w).strip()
            if len(test)<=25: line=test
            else:
                if line: words.append(line); line=w
        if line: words.append(line)
        for li,ln in enumerate(words[:2]):
            cv2.putText(frame,ln,(px+21,y+li*12),FONT,0.29,col,1,cv2.LINE_AA)
        y+=len(words)*12+4
        if y>fh-24: break

    cv2.putText(frame,"Q=quit  I=questions  R=record",(px+4,fh-10),FONT,0.24,C_DIM,1,cv2.LINE_AA)

def draw_face_overlay(frame, lms, fw, fh):
    mp_drawing.draw_landmarks(frame,lms,mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    for idx in [LEFT_IRIS_CENTER,RIGHT_IRIS_CENTER]:
        px,py=lm_px(lms.landmark,idx,fw,fh); cv2.circle(frame,(px,py),3,C_ACCENT,-1)

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
def generate_graphs(tracker):
    if not PLT_OK or tracker.frame_count<5: return None
    has_q=len(tracker.question_log)>0; ncols=3 if has_q else 2
    fig,axes=plt.subplots(1,ncols,figsize=(14 if has_q else 10,4))
    fig.patch.set_facecolor('#0b0f16')
    ax1=axes[0]; ax2=axes[1]; ax3=axes[2] if has_q else None

    times=[t for t,_ in tracker.confidence_history]; confs=[v for _,v in tracker.confidence_history]
    ax1.set_facecolor('#0f1520')
    if times:
        ax1.plot(times,confs,color='#00c8f0',linewidth=1.5)
        ax1.fill_between(times,confs,alpha=0.12,color='#00c8f0')
        ax1.axhline(y=70,color='#00e87a',linestyle='--',linewidth=1,alpha=0.5,label='Good 70%')
        ax1.axhline(y=45,color='#f5c842',linestyle='--',linewidth=1,alpha=0.5,label='Fair 45%')
    ax1.set_xlim(0,max(times) if times else 1); ax1.set_ylim(0,100)
    ax1.set_xlabel('Time (s)',color='#8aaabb',fontsize=8); ax1.set_ylabel('Confidence (%)',color='#8aaabb',fontsize=8)
    ax1.set_title('Confidence Over Session',color='#00c8f0',fontsize=10,fontweight='bold')
    ax1.tick_params(colors='#8aaabb',labelsize=7); ax1.spines[:].set_color('#1a2535')
    ax1.legend(fontsize=7,facecolor='#0f1520',labelcolor='#8aaabb',loc='lower right')

    mlabels=['Confidence','Smiling','Eye Contact','Posture','Speech WPM']
    mvals=[tracker.avg_confidence(),tracker.pct_smiling(),tracker.pct_eye_contact(),
           tracker.pct_good_posture(),min(100,int(tracker.avg_wpm()/1.5))]
    mcolors=[]
    for i,v in enumerate(mvals):
        if i==4: mcolors.append('#00e87a' if WPM_IDEAL_LOW<=tracker.avg_wpm()<=WPM_IDEAL_HIGH else('#f5c842' if tracker.avg_wpm()>0 else'#3d5268'))
        else: mcolors.append('#00e87a' if v>=70 else('#f5c842' if v>=45 else'#ff3d5a'))
    ax2.set_facecolor('#0f1520')
    bars2=ax2.bar(mlabels,mvals,color=mcolors,alpha=0.85,width=0.5)
    ax2.set_ylim(0,110); ax2.set_title('Session Metrics',color='#00c8f0',fontsize=10,fontweight='bold')
    ax2.tick_params(colors='#8aaabb',labelsize=7); ax2.spines[:].set_color('#1a2535')
    ax2.set_ylabel('Score (%)',color='#8aaabb',fontsize=8)
    for bar,val,i in zip(bars2,mvals,range(len(mvals))):
        lbl=f"{tracker.avg_wpm()} WPM" if i==4 and tracker.avg_wpm()>0 else f"{val}%"
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,lbl,ha='center',va='bottom',color='white',fontsize=7)

    if ax3 is not None:
        ql=tracker.question_log; q_labels=[f"Q{i+1}" for i in range(len(ql))]
        q_confs=[q["avg_conf"] for q in ql]
        q_colors=['#00e87a' if v>=70 else('#f5c842' if v>=45 else'#ff3d5a') for v in q_confs]
        ax3.set_facecolor('#0f1520')
        bars3=ax3.bar(q_labels,q_confs,color=q_colors,alpha=0.85,width=0.6)
        ax3.axhline(y=70,color='#00e87a',linestyle='--',linewidth=1,alpha=0.4)
        ax3.set_ylim(0,110); ax3.set_title('Confidence Per Question',color='#00c8f0',fontsize=10,fontweight='bold')
        ax3.tick_params(colors='#8aaabb',labelsize=7); ax3.spines[:].set_color('#1a2535')
        ax3.set_ylabel('Avg Confidence (%)',color='#8aaabb',fontsize=8)
        for bar,val in zip(bars3,q_confs):
            ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{val}%",ha='center',va='bottom',color='white',fontsize=7)
        if q_confs:
            bi=q_confs.index(max(q_confs)); wi=q_confs.index(min(q_confs))
            ax3.annotate('BEST',xy=(bi,q_confs[bi]+4),ha='center',fontsize=6,color='#00e87a')
            ax3.annotate('WORST',xy=(wi,q_confs[wi]+4),ha='center',fontsize=6,color='#ff3d5a')

    plt.tight_layout(pad=1.5)
    fig.canvas.draw()
    buf=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8)
    w_fig,h_fig=fig.canvas.get_width_height()
    img_bgr=cv2.cvtColor(buf.reshape(h_fig,w_fig,4),cv2.COLOR_RGBA2BGR)
    plt.close(fig); return img_bgr

# ══════════════════════════════════════════════════════════════════════════════
# REPORT CARD
# ══════════════════════════════════════════════════════════════════════════════
def show_report_card(cap, tracker):
    grade,grade_col=tracker.grade(); avg_conf=tracker.avg_confidence()
    pct_smile=tracker.pct_smiling(); pct_gaze=tracker.pct_eye_contact()
    pct_post=tracker.pct_good_posture(); avg_wpm=tracker.avg_wpm()
    stress_pct=tracker.stress_pct(); dom_em=tracker.dominant_emotion()
    happy_pct=tracker.emotion_pct("happy"); neutral_pct=tracker.emotion_pct("neutral")
    dur=tracker.duration_secs(); m,s=divmod(dur,60)
    best_q,worst_q=tracker.best_worst_questions(); graph_img=generate_graphs(tracker)

    if stress_pct>=40:   em_summary=("You appeared quite stressed.",C_RED,"Take slow breaths. Pause 2s before each response.")
    elif stress_pct>=20: em_summary=("Some nervousness was visible.",C_YELLOW,"You recovered well. Practice pausing before answering.")
    elif happy_pct>=30:  em_summary=("You came across warm and positive!",C_GREEN,"Your natural expression is a strength.")
    elif neutral_pct>=70:em_summary=("Composed but a bit flat.",C_YELLOW,"Add more expression -- nod, smile, show enthusiasm.")
    else:                em_summary=("Emotional presence was balanced.",C_GREEN,"Work on showing more warmth and energy.")

    good_pts=[]
    if avg_conf>=70:    good_pts.append("Strong overall confidence -- you came across well")
    elif avg_conf>=55:  good_pts.append(f"Decent confidence ({avg_conf}%) -- solid foundation")
    if pct_smile>=50:   good_pts.append(f"Great smile ({pct_smile}%) -- very approachable")
    elif pct_smile>=30: good_pts.append(f"Good smiling ({pct_smile}%) -- keep it natural")
    if pct_gaze>=75:    good_pts.append(f"Excellent eye contact ({pct_gaze}%) -- looked engaged")
    elif pct_gaze>=55:  good_pts.append(f"Good eye contact ({pct_gaze}%) -- mostly on camera")
    if pct_post>=85:    good_pts.append(f"Excellent posture ({pct_post}%) -- upright throughout")
    elif pct_post>=65:  good_pts.append(f"Decent posture ({pct_post}%) -- generally upright")
    if avg_wpm>0 and WPM_IDEAL_LOW<=avg_wpm<=WPM_IDEAL_HIGH:
        good_pts.append(f"Perfect speaking pace ({avg_wpm} WPM) -- easy to follow")
    if stress_pct<15:   good_pts.append("Appeared calm and composed throughout")
    if happy_pct>=20:   good_pts.append(f"Showed positivity and warmth ({happy_pct}% happy)")
    if best_q: good_pts.append(f"Best Q ({best_q['avg_conf']}%): {best_q['text'][:36]}...")
    if not good_pts: good_pts.append("You practiced -- that is the most important first step!")

    bad_pts=[]
    if avg_conf<55:     bad_pts.append(f"Confidence ({avg_conf}%) needs work -- aim for 70%+")
    elif avg_conf<70:   bad_pts.append(f"Confidence ({avg_conf}%) close -- a bit more energy")
    if pct_smile<30:    bad_pts.append(f"Rarely smiled ({pct_smile}%) -- try to look warmer")
    if pct_gaze<55:     bad_pts.append(f"Eye contact ({pct_gaze}%) low -- look at camera lens")
    if pct_post<65:     bad_pts.append(f"Posture ({pct_post}%) -- sit upright, shoulders level")
    if avg_wpm>WPM_IDEAL_HIGH: bad_pts.append(f"Too fast ({avg_wpm} WPM) -- slow to 100-180 WPM")
    elif 0<avg_wpm<WPM_IDEAL_LOW: bad_pts.append(f"Too slow ({avg_wpm} WPM) -- be more energetic")
    if stress_pct>=20:  bad_pts.append(f"Stress visible {stress_pct}% -- breathe and slow down")
    if worst_q: bad_pts.append(f"Hardest Q ({worst_q['avg_conf']}%): {worst_q['text'][:36]}...")
    if not bad_pts: bad_pts.append("No major issues -- great session!")

    improve_tips=[]
    if stress_pct>=25:  improve_tips.append(("Calm down:","Before each answer take 1 deep breath and pause."))
    if pct_smile<40:    improve_tips.append(("Smile more:","Smile at mirror 30s before interview to relax."))
    if pct_gaze<60:     improve_tips.append(("Eye contact:","Stick a dot above your webcam to remind yourself."))
    if avg_conf<60:     improve_tips.append(("Confidence:","Record yourself daily. Watch it back. You improve fast."))
    if avg_wpm>WPM_IDEAL_HIGH: improve_tips.append(("Pace:","Put SLOW DOWN sticky note on your monitor."))
    if pct_post<70:     improve_tips.append(("Posture:","Sit at edge of chair -- forces you naturally upright."))
    if not improve_tips:improve_tips.append(("Keep it up:","You are performing well. Focus on consistency."))

    print("[INFO] Report card shown. Press any key to close.")
    while True:
        ret,raw=cap.read()
        if not ret: break
        raw=cv2.flip(raw,1); fh,fw=raw.shape[:2]
        ov=raw.copy(); cv2.rectangle(ov,(0,0),(fw,fh),(5,8,14),-1)
        cv2.addWeighted(ov,0.92,raw,0.08,0,raw); frame=raw

        title="SESSION REPORT CARD"
        tw=cv2.getTextSize(title,FONTB,0.85,2)[0][0]
        cv2.putText(frame,title,((fw-tw)//2,34),FONTB,0.85,C_ACCENT,2,cv2.LINE_AA)
        dur_str=f"Duration: {m:02d}:{s:02d}   Grade: {grade}   Confidence: {avg_conf}%   Mood: {dom_em.upper()}"
        dw=cv2.getTextSize(dur_str,FONT,0.32,1)[0][0]
        cv2.putText(frame,dur_str,((fw-dw)//2,52),FONT,0.32,grade_col,1,cv2.LINE_AA)

        cx,cy,r=46,105,36; cv2.circle(frame,(cx,cy),r,grade_col,3)
        cv2.circle(frame,(cx,cy),r-4,(18,22,32),-1)
        gw2=cv2.getTextSize(grade,FONTB,1.3,2)[0][0]
        cv2.putText(frame,grade,(cx-gw2//2,cy+12),FONTB,1.3,grade_col,2,cv2.LINE_AA)
        cv2.putText(frame,"GRADE",(cx-18,cy+26),FONT,0.26,C_DIM,1,cv2.LINE_AA)

        metrics=[("CONFIDENCE",f"{avg_conf}%",avg_conf>=70),("SMILING",f"{pct_smile}%",pct_smile>=40),
                 ("EYE CONTACT",f"{pct_gaze}%",pct_gaze>=70),("POSTURE",f"{pct_post}%",pct_post>=75),
                 ("SPEECH",f"{avg_wpm} WPM" if avg_wpm>0 else"N/A",WPM_IDEAL_LOW<=avg_wpm<=WPM_IDEAL_HIGH if avg_wpm>0 else True),
                 ("STRESS",f"{stress_pct}%",stress_pct<=15)]
        box_w=int((fw-100)//len(metrics))-4; box_h=48; box_y=62; bx0=90
        for i,(lbl,val,good) in enumerate(metrics):
            bx=bx0+i*(box_w+4)
            if i==5: col=C_RED if stress_pct>25 else(C_YELLOW if stress_pct>10 else C_GREEN)
            else:    col=C_GREEN if good else C_YELLOW
            ov2=frame.copy(); cv2.rectangle(ov2,(bx,box_y),(bx+box_w,box_y+box_h),(18,24,36),-1)
            cv2.addWeighted(ov2,0.85,frame,0.15,0,frame)
            cv2.rectangle(frame,(bx,box_y),(bx+box_w,box_y+box_h),col,1)
            lw2=cv2.getTextSize(lbl,FONT,0.24,1)[0][0]
            cv2.putText(frame,lbl,(bx+(box_w-lw2)//2,box_y+13),FONT,0.24,C_DIM,1,cv2.LINE_AA)
            vw2=cv2.getTextSize(val,FONTB,0.52,1)[0][0]
            cv2.putText(frame,val,(bx+(box_w-vw2)//2,box_y+37),FONTB,0.52,col,1,cv2.LINE_AA)

        em_y=box_y+box_h+7; em_text,em_col,em_tip=em_summary
        ov5=frame.copy(); cv2.rectangle(ov5,(6,em_y),(fw-6,em_y+32),(20,20,32),-1)
        cv2.addWeighted(ov5,0.80,frame,0.20,0,frame)
        cv2.rectangle(frame,(6,em_y),(fw-6,em_y+32),em_col,1)
        cv2.putText(frame,"EMOTION:  "+em_text,(14,em_y+12),FONTB,0.36,em_col,1,cv2.LINE_AA)
        cv2.putText(frame,em_tip,(14,em_y+26),FONT,0.28,C_WHITE,1,cv2.LINE_AA)

        panel_y=em_y+38; half_w2=fw//2-10; panel_h=int(fh*0.26)
        def draw_side(x1,x2,title_txt,items,col):
            ov3=frame.copy(); bg=(8,28,12) if col==C_GREEN else(28,8,12)
            cv2.rectangle(ov3,(x1,panel_y),(x2,panel_y+panel_h),bg,-1)
            cv2.addWeighted(ov3,0.75,frame,0.25,0,frame)
            cv2.rectangle(frame,(x1,panel_y),(x2,panel_y+panel_h),col,1)
            cv2.putText(frame,title_txt,(x1+8,panel_y+14),FONTB,0.38,col,1,cv2.LINE_AA)
            cv2.line(frame,(x1+8,panel_y+18),(x2-6,panel_y+18),col,1)
            y=panel_y+30
            for txt in items[:7]:
                words2=[]; line2=""
                for w in txt.split():
                    test2=(line2+" "+w).strip()
                    if len(test2)<=48: line2=test2
                    else:
                        if line2: words2.append(line2); line2=w
                if line2: words2.append(line2)
                cv2.circle(frame,(x1+10,y-3),3,col,-1)
                for li,ln in enumerate(words2[:2]):
                    cv2.putText(frame,ln,(x1+17,y+li*11),FONT,0.28,col,1,cv2.LINE_AA)
                y+=len(words2)*11+4
                if y>panel_y+panel_h-8: break
        draw_side(6,half_w2,"  WHAT WENT WELL",good_pts,C_GREEN)
        draw_side(half_w2+6,fw-6,"  NEEDS IMPROVEMENT",bad_pts,C_RED)

        tips_y=panel_y+panel_h+6; tips_h=int(fh*0.12)
        ov6=frame.copy(); cv2.rectangle(ov6,(6,tips_y),(fw-6,tips_y+tips_h),(14,20,36),-1)
        cv2.addWeighted(ov6,0.80,frame,0.20,0,frame)
        cv2.rectangle(frame,(6,tips_y),(fw-6,tips_y+tips_h),C_ACCENT,1)
        cv2.putText(frame,"  HOW TO IMPROVE FOR NEXT TIME",(12,tips_y+14),FONTB,0.37,C_ACCENT,1,cv2.LINE_AA)
        cv2.line(frame,(12,tips_y+17),(fw-10,tips_y+17),C_ACCENT,1)
        col_w3=(fw-20)//max(1,min(4,len(improve_tips)))
        for i,(heading,tip_txt) in enumerate(improve_tips[:4]):
            bx3=14+i*col_w3
            cv2.putText(frame,heading,(bx3,tips_y+30),FONTB,0.29,C_YELLOW,1,cv2.LINE_AA)
            words3=[]; line3=""
            for w in tip_txt.split():
                test3=(line3+" "+w).strip()
                if len(test3)<=30: line3=test3
                else:
                    if line3: words3.append(line3); line3=w
            if line3: words3.append(line3)
            for li,ln in enumerate(words3[:3]):
                cv2.putText(frame,ln,(bx3,tips_y+41+li*10),FONT,0.25,C_WHITE,1,cv2.LINE_AA)

        if graph_img is not None:
            gh2,gw2i=graph_img.shape[:2]; av_y=tips_y+tips_h+5; av_h=fh-av_y-18; av_w=fw-16
            if av_h>50:
                sc=min(av_w/gw2i,av_h/gh2); tw3=int(gw2i*sc); th3=int(gh2*sc)
                resized=cv2.resize(graph_img,(tw3,th3)); gx3=(fw-tw3)//2
                if av_y+th3<fh-14: frame[av_y:av_y+th3,gx3:gx3+tw3]=resized

        hint="Press any key to close"
        hw2=cv2.getTextSize(hint,FONT,0.32,1)[0][0]
        cv2.putText(frame,hint,((fw-hw2)//2,fh-8),FONT,0.32,C_DIM,1,cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME,frame)
        if cv2.waitKey(30)&0xFF!=255: break

# ══════════════════════════════════════════════════════════════════════════════
# PRE-INTERVIEW CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════
def show_checklist(cap):
    start=time.time(); item_delay=CHECKLIST_SECS/len(CHECKLIST)
    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1); h,w=frame.shape[:2]
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
        elapsed=time.time()-start; remaining=max(0,CHECKLIST_SECS-elapsed)
        title="GET READY FOR YOUR INTERVIEW"
        tw=cv2.getTextSize(title,FONTB,0.72,1)[0][0]
        cv2.putText(frame,title,((w-tw)//2,52),FONTB,0.72,C_ACCENT,1,cv2.LINE_AA)
        sub="Review these tips before we start:"
        sw=cv2.getTextSize(sub,FONT,0.44,1)[0][0]
        cv2.putText(frame,sub,((w-sw)//2,78),FONT,0.44,C_DIM,1,cv2.LINE_AA)
        items_shown=min(len(CHECKLIST),int(elapsed/item_delay)+1)
        col_w=w//2-40; y0=110
        for i,(cat,tip) in enumerate(CHECKLIST):
            if i>=items_shown: break
            col=0 if i<4 else 1; row=i%4
            bx=30+col*(col_w+40); by=y0+row*58
            filled_rect(frame,bx,by,bx+col_w,by+48,(30,30,42),alpha=0.8)
            cv2.rectangle(frame,(bx,by),(bx+col_w,by+48),(60,60,80),1)
            age=elapsed-i*item_delay
            cv2.putText(frame,"[OK]",(bx+8,by+22),FONT,0.38,C_GREEN if age>0.4 else C_YELLOW,1,cv2.LINE_AA)
            cv2.putText(frame,cat,(bx+52,by+17),FONTB,0.38,C_ACCENT,1,cv2.LINE_AA)
            words=[]; line=""
            for wrd in tip.split():
                test=(line+" "+wrd).strip()
                if len(test)<=36: line=test
                else:
                    if line: words.append(line); line=wrd
            if line: words.append(line)
            for li,ln in enumerate(words[:2]):
                cv2.putText(frame,ln,(bx+52,by+30+li*13),FONT,0.31,C_WHITE,1,cv2.LINE_AA)
        bar_y=h-44
        msg=f"Starting in {int(remaining)+1}s ...  (SPACE to skip)"
        mw=cv2.getTextSize(msg,FONT,0.48,1)[0][0]
        cv2.putText(frame,msg,((w-mw)//2,bar_y-8),FONT,0.48,C_WHITE,1,cv2.LINE_AA)
        bar_w=w-80; prog=1.0-remaining/CHECKLIST_SECS
        cv2.rectangle(frame,(40,bar_y),(40+bar_w,bar_y+8),(50,50,60),-1)
        fill=int(prog*bar_w)
        if fill>0:
            r=prog; bcol=(int(80*r),int(210*(1-r*0.5)),int(90*(1-r)))
            cv2.rectangle(frame,(40,bar_y),(40+fill,bar_y+8),bcol,-1)
        cv2.imshow(WINDOW_NAME,frame)
        key=cv2.waitKey(30)&0xFF
        if key in(ord(' '),ord('q')) or remaining<=0: break
    ret,frame=cap.read()
    if ret:
        frame=cv2.flip(frame,1); ov=frame.copy()
        cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),(0,0,0),-1)
        cv2.addWeighted(ov,0.5,frame,0.5,0,frame)
        msg="GO!"; mw=cv2.getTextSize(msg,FONTB,3.0,3)[0][0]
        cv2.putText(frame,msg,((frame.shape[1]-mw)//2,frame.shape[0]//2+20),FONTB,3.0,C_GREEN,3,cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME,frame); cv2.waitKey(900)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cap=cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): print("[ERROR] Cannot open webcam."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    show_checklist(cap)
    speech_monitor.start()
    session_tracker.start()

    resume_pdf=find_resume_pdf()
    if resume_pdf:
        print(f"[INFO] Resume: {resume_pdf}")
        qs=generate_resume_questions(extract_resume_text(resume_pdf))
        question_mgr.set_questions(qs)
    else:
        print("[INFO] No resume found. Place resume.pdf here to auto-load.")

    current_emotion="neutral"; last_emotion_time=0.0; emotion_lock=threading.Lock()
    yaw_buf=deque(maxlen=SMOOTHING_FRAMES); pitch_buf=deque(maxlen=SMOOTHING_FRAMES)
    roll_buf=deque(maxlen=SMOOTHING_FRAMES)
    posture_ok=True; posture_msg=""; posture_col=C_GREEN

    def async_emotion(frm):
        nonlocal current_emotion
        em=run_emotion(frm)
        with emotion_lock: current_emotion=em

    print("[INFO] Started.  I=questions  N=next  R=record  Q=quit+report")

    while True:
        ret,raw=cap.read()
        if not ret: break
        raw=cv2.flip(raw,1); fh,fw=raw.shape[:2]; frame=raw
        cam_w=fw-PANEL_W; rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)

        result=face_mesh.process(rgb)
        yaw=pitch=roll=0.0; gaze_l=gaze_r=0.5; smile=0.0; em="neutral"

        if result.multi_face_landmarks:
            lms=result.multi_face_landmarks[0]
            pr=compute_head_pose(lms.landmark,fw,fh)
            if pr:
                y_,p_,r_=pr
                yaw_buf.append(y_); pitch_buf.append(p_); roll_buf.append(r_)
                yaw=float(np.mean(yaw_buf)); pitch=float(np.mean(pitch_buf)); roll=float(np.mean(roll_buf))
            gaze_l,gaze_r=compute_gaze(lms.landmark,fw,fh)
            smile=compute_smile(lms.landmark)
            ear_l=eye_aspect_ratio(lms.landmark,L_EYE_TOP,L_EYE_BOT,133,33)
            ear_r=eye_aspect_ratio(lms.landmark,R_EYE_TOP,R_EYE_BOT,362,263)
            blink_tracker.update((ear_l+ear_r)/2)
            draw_face_overlay(frame,lms,fw,fh)
            head_y=fh//2+80 if question_mgr.active else fh//2
            lbl=f"Turn {'right' if yaw>0 else 'left'} -- face forward" if abs(yaw)>18 else "Good head position"
            lc=C_RED if abs(yaw)>18 else C_GREEN
            cv2.putText(frame,lbl,(20,head_y),FONT,0.6,lc,2,cv2.LINE_AA)
            now=time.time()
            if EMOTION_BACKEND and now-last_emotion_time>EMOTION_INTERVAL:
                last_emotion_time=now
                threading.Thread(target=async_emotion,args=(raw.copy(),),daemon=True).start()
            with emotion_lock: em=current_emotion
        else:
            cv2.putText(frame,"No face detected",(30,fh//2),FONT,0.8,C_RED,2,cv2.LINE_AA)

        pose_res=pose_detector.process(rgb)
        if pose_res.pose_landmarks:
            posture_ok,posture_msg,posture_col=compute_posture(pose_res.pose_landmarks,fw,fh)
            draw_posture_overlay(frame,pose_res.pose_landmarks,fw,fh,posture_ok,posture_msg,posture_col)
        else:
            posture_ok=True; posture_msg=""

        # Blink status — 5-second window
        b5,b_lbl,b_col=blink_tracker.status()

        # Push all toasts including blink
        push_toasts(yaw,pitch,roll,gaze_l,gaze_r,smile,em,posture_ok,b5,b_lbl,b_col)

        try:
            question_mgr.update(); question_mgr.draw(frame,fw,fh)
        except Exception as e:
            print(f"[WARN] Question error: {e}")

        bx1,bx2=20,cam_w-20; bar_y_top=125 if question_mgr.active else 22
        cv2.putText(frame,"GAZE",(bx1,bar_y_top-4),FONT,0.28,C_DIM,1,cv2.LINE_AA)
        h_bar(frame,bx1,bx2,bar_y_top,8,(gaze_l+gaze_r)/2,
              C_GREEN if 0.28<(gaze_l+gaze_r)/2<0.72 else C_RED)
        cv2.putText(frame,"SMILE",(bx1,fh-22),FONT,0.28,C_DIM,1,cv2.LINE_AA)
        h_bar(frame,bx1,bx2,fh-17,8,smile,C_GREEN if smile>0.15 else C_YELLOW)

        with emotion_lock: em=current_emotion
        confidence=compute_confidence(yaw,pitch,gaze_l,gaze_r,smile,em,b5)
        tips=build_tips(yaw,pitch,roll,gaze_l,gaze_r,smile,em,b5,b_lbl,b_col,posture_ok,posture_msg)
        draw_panel(frame,fw,fh,em,smile,yaw,pitch,tips,confidence,b5,b_lbl,b_col)

        wpm,_,_=speech_monitor.get()
        session_tracker.record(confidence,smile,gaze_l,gaze_r,posture_ok,wpm,em)
        recorder.draw_indicator(frame,fw,fh); recorder.write(frame)
        speech_monitor.draw(frame,fh,cam_w); toast_mgr.draw(frame,fw,fh)
        cv2.imshow(WINDOW_NAME,frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key in(ord('i'),ord('I')):
            if question_mgr.active:
                question_mgr.stop(); toast_mgr.push("Question mode OFF",C_DIM,"qmode",1)
            else:
                question_mgr.start()
                toast_mgr.push("Resume questions loaded!" if question_mgr.resume_mode else "Interview mode ON!",C_ACCENT,"qmode",1)
        elif key in(ord('n'),ord('N')):
            if question_mgr.active:
                question_mgr.next_question(); toast_mgr.push("Next question",C_ACCENT,"nextq",1)
        elif key in(ord('r'),ord('R')):
            if recorder.recording:
                recorder.stop(); toast_mgr.push("Recording saved!",C_GREEN,"rec",1)
            else:
                recorder.start(fw,fh); toast_mgr.push("Recording started -- R to stop",C_RED,"rec",1)

    if recorder.recording: recorder.stop()
    speech_monitor.stop(); session_tracker.finish()
    if session_tracker.frame_count>30: show_report_card(cap,session_tracker)
    cap.release(); cv2.destroyAllWindows()
    print(f"[INFO] Done. Blinks: {blink_tracker.total}  Grade: {session_tracker.grade()[0]}  Conf: {session_tracker.avg_confidence()}%")


if __name__ == "__main__":
    main()