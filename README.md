# 🛡️ AI Network Threat Detector

An AI-powered network intrusion detection system using **Random Forest** and **SHAP Explainability** trained on the CICIDS2017 benchmark dataset.

## 🚀 Live Demo
👉 [Click here to view live dashboard](https://ai-network-threat-detector-bsdvlzfur7hrt8mgkx58sf.streamlit.app)

<p align="center">
  <strong> Dashboard Preview</strong>
</p>
<img width="1303" height="699" alt="Screenshot from 2026-06-09 15-00-29" src="https://github.com/user-attachments/assets/de488e47-7562-49bf-9566-42689b4495e5" />
<br> <br>
<img width="1302" height="698" alt="Screenshot from 2026-06-09 15-00-35" src="https://github.com/user-attachments/assets/77b188e9-f138-4bd1-beb8-615e5d6e7932" />
<br> <br>
<img width="1304" height="697" alt="Screenshot from 2026-06-09 15-00-37" src="https://github.com/user-attachments/assets/bb489603-e0ae-4ef9-b9ac-a89bd7c82452" />
<br> <br>
<img width="1295" height="707" alt="Screenshot from 2026-06-09 15-00-38" src="https://github.com/user-attachments/assets/c716b46a-7341-49ed-ad1d-8cf48e540e83" />
<br> <br>
<img width="1306" height="692" alt="Screenshot from 2026-06-09 15-00-42" src="https://github.com/user-attachments/assets/59e4a92c-6566-4c1f-9e1e-d05a9d49a7a5" />

## 📌 About
This dashboard detects malicious network traffic using machine learning. Upload a network traffic CSV or use the built-in sample data to see real-time threat detection with explainable AI.

## ✨ Features
- 🔴 **Attack Detection** — Classifies network flows as BENIGN or DDoS
- 🧠 **Explainable AI** — SHAP values show exactly why each flow was flagged
- 📊 **Interactive Dashboard** — Traffic breakdown charts and confidence scores
- ⚙️ **Threshold Control** — Adjustable confidence threshold for alerts
- ⬇️ **Export** — Download alert report as CSV
- 🔜 **Coming Soon** — PortScan and BruteForce attack detection

## 🗂️ Dataset
Trained on [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity benchmark dataset with 225,000+ real network flows.

## 🛠️ Tech Stack
`Python` `Scikit-learn` `SHAP` `Streamlit` `Plotly` `Pandas`

## 🚀 Run Locally
```bash
git clone https://github.com/Sakshi-gg/AI-Network-Threat-Detector.git
cd AI-Network-Threat-Detector
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💻 Author
**Sakshi** — MTech AI & Cybersecurity | [GitHub](https://github.com/Sakshi-gg)

> ⚠️ Note: Achieves 100% accuracy on CICIDS2017 benchmark. Real-world performance may vary as dataset is lab-generated.
