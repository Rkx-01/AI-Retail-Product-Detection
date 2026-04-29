# Retail AI Pipeline - Main API Gateway
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import numpy as np
import traceback
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from werkzeug.utils import secure_filename

from detector import ProductDetector
from grouper import ProductGrouper
from utils import draw_and_save_visuals

app = Flask(__name__)

@app.errorhandler(500)
def internal_error(error):
    import traceback
    return jsonify({"error": "Internal Server Error", "traceback": traceback.format_exc()}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    import traceback
    return jsonify({"error": f"Unhandled Exception: {str(e)}", "traceback": traceback.format_exc()}), 500

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize AI models lazily
detector = None
grouper = None

def get_models():
    global detector, grouper
    if detector is None:
        print("🚀 Initializing YOLOv8 & Grouping models...")
        detector = ProductDetector()
        grouper = ProductGrouper()
        print("✅ Models ready!")
    return detector, grouper

# BASE URL FOR ASSETS
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Rkx-01/AI-Retail-Product-Detection/main"

# Premium UI Template
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Intelligence | YOLOv8 Pipeline</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #0f172a;
            --card-bg: rgba(255, 255, 255, 0.1);
            --text-main: #ffffff;
            --text-muted: #cbd5e1;
            --accent-yellow: #f59e0b;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg); 
            color: var(--text-main);
            line-height: 1.5;
            padding: 2rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }}

        .beta-banner {{
            background: rgba(245, 158, 11, 0.2);
            backdrop-filter: blur(8px);
            border: 1px solid var(--accent-yellow);
            color: var(--accent-yellow);
            padding: 12px 20px;
            border-radius: 12px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.875rem;
        }}
        .beta-badge {{
            background: #ef4444;
            color: white;
            font-weight: 700;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }}

        header {{ margin-bottom: 32px; }}
        header h1 {{ font-size: 2.5rem; font-weight: 700; color: #ffffff; letter-spacing: -0.025em; }}
        header p {{ color: var(--text-muted); font-size: 1.1rem; }}

        .main-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .upload-section {{
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 60px 40px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(255, 255, 255, 0.03);
            cursor: pointer;
        }}
        .upload-section:hover {{ 
            border-color: var(--primary); 
            background: rgba(37, 99, 235, 0.1);
            transform: translateY(-2px);
        }}
        
        #imageInput {{ display: none; }}

        .btn-process {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 24px;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);
        }}
        .btn-process:hover {{ background: var(--primary-hover); transform: scale(1.01); }}
        .btn-process:disabled {{ background: #475569; cursor: not-allowed; transform: none; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 32px;
            display: none;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }}
        .stat-val {{ font-size: 2rem; font-weight: 800; color: #3b82f6; }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-top: 4px; }}

        .video-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            overflow: hidden;
        }}
        .video-bg video {{
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
            opacity: 0.9;
        }}
        .video-bg::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, transparent 0%, rgba(15, 23, 42, 0.8) 100%);
        }}

        #resultContainer {{
            margin-top: 40px;
            display: none;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        
        .image-wrapper {{
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #outputImage {{ width: 100%; display: block; }}

        .loading-overlay {{
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 32px;
        }}
        .spinner {{
            width: 48px;
            height: 48px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        .gallery-item {{
            width: 90px;
            height: 90px;
            object-fit: cover;
            border-radius: 12px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s;
            opacity: 0.7;
        }}
        .gallery-item:hover {{ opacity: 1; border-color: var(--primary); transform: scale(1.05); }}

    </style>
</head>
<body>
    <div class="video-bg">
        <video autoplay muted loop playsinline id="bgVideo">
            <source src="{RAW_GITHUB_URL}/AI_Scans_Liquor_Store_Shelf.mp4" type="video/mp4">
        </video>
    </div>

    <div class="container">
        <div class="beta-banner">
            <span class="beta-badge">BETA VERSION</span>
            <span>System under development. Currently optimized for YOLO-formatted datasets and retail imagery.</span>
        </div>

        <header>
            <h1>Retail Intelligence</h1>
            <p>Advanced Inventory Scanning via YOLOv8</p>
        </header>

        <main class="main-card">
            <form id="uploadForm">
                <div class="upload-section" onclick="document.getElementById('imageInput').click()">
                    <p>Select Shelf Image</p>
                    <div id="fileName" style="margin-top: 10px; color: var(--primary);"></div>
                </div>
                <input type="file" id="imageInput" name="image" accept="image/*" required>
                <button type="submit" id="submitBtn" class="btn-process">Analyze Inventory</button>
            </form>

            <div id="loading" class="loading-overlay">
                <div class="spinner"></div>
                <p style="margin-top: 16px; font-weight: 600;">Processing Vision Pipeline...</p>
            </div>

            <div id="statsGrid" class="stats-grid">
                <div class="stat-card">
                    <div id="countVal" class="stat-val">0</div>
                    <div class="stat-label">SKUs Detected</div>
                </div>
                <div class="stat-card">
                    <div id="groupVal" class="stat-val">0</div>
                    <div class="stat-label">Unique Groups</div>
                </div>
            </div>

            <div style="margin-top: 40px; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 24px;">
                <h3 style="font-size: 0.9rem; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted);">Quick Test Samples</h3>
                <div id="sampleGallery" style="display: flex; gap: 16px; overflow-x: auto; padding-bottom: 12px;"></div>
            </div>

            <div id="resultContainer">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <h3 style="font-size: 1.25rem; font-weight: 700;">Detection Insight</h3>
                    <a id="downloadLink" href="#" download style="color: var(--primary); text-decoration: none; font-size: 0.875rem; font-weight: 600;">Save Analysis</a>
                </div>
                <div class="image-wrapper">
                    <img id="outputImage" src="" alt="Result">
                </div>
            </div>
        </main>
    </div>

    <script>
        const sampleGallery = document.getElementById('sampleGallery');
        const githubSamples = [
            "08d8f724d7a418745e5e1a7209da1916.jpg",
            "0e30879f0d654597228d9613afc343e0.jpg",
            "54f6f1aa68d6a5fef1baf9e013a290ab.jpg",
            "55ce59c2a91ffeb7c749f2c0d9bc096c.jpg",
            "8313a95add5aa77ba86373901c6b120e.jpg",
            "9a2d8faebb78b60a551ade70dcd27a53.jpg",
            "b3e73550486bb55e9d4cddeeb5b8a8be.webp",
            "eb0474bf5104e9862a310d5370169ad2.jpg"
        ];

        function loadSamples() {{
            githubSamples.forEach(filename => {{
                const img = document.createElement('img');
                const url = `{RAW_GITHUB_URL}/dataset/samples/` + filename;
                img.src = url;
                img.className = 'gallery-item';
                img.onclick = () => runSampleTest(url);
                sampleGallery.appendChild(img);
            }});
        }}
        loadSamples();

        async function runSampleTest(url) {{
            submitBtn.disabled = true;
            loading.style.display = 'flex';
            resultContainer.style.display = 'none';
            statsGrid.style.display = 'none';
            try {{
                const res = await fetch('/process-url', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ url: url }})
                }});
                const text = await res.text();
                let data;
                try {{
                    data = JSON.parse(text);
                }} catch (e) {{
                    console.error("Server Response:", text);
                    throw new Error("Server returned an invalid response (likely an HTML error page). Check browser console.");
                }}
                if (data.error) throw new Error(data.error + (data.traceback ? "\\n\\n" + data.traceback : ""));
                displayResults(data);
            }} catch (err) {{ alert('Analysis Error: ' + err.message); }}
            finally {{ submitBtn.disabled = false; loading.style.display = 'none'; }}
        }}

        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('imageInput');
        const fileNameDiv = document.getElementById('fileName');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('resultContainer');
        const statsGrid = document.getElementById('statsGrid');

        fileInput.onchange = () => {{ if (fileInput.files[0]) fileNameDiv.innerText = fileInput.files[0].name; }};

        form.onsubmit = async (e) => {{
            e.preventDefault();
            submitBtn.disabled = true;
            loading.style.display = 'flex';
            const formData = new FormData(form);
            try {{
                const res = await fetch('/process', {{ method: 'POST', body: formData }});
                const text = await res.text();
                let data;
                try {{
                    data = JSON.parse(text);
                }} catch (e) {{
                    console.error("Server Response:", text);
                    throw new Error("Server returned an invalid response (likely an HTML error page). Check browser console.");
                }}
                if (data.error) throw new Error(data.error + (data.traceback ? "\\n\\n" + data.traceback : ""));
                displayResults(data);
            }} catch (err) {{ alert('Analysis Error: ' + err.message); }}
            finally {{ submitBtn.disabled = false; loading.style.display = 'none'; }}
        }};

        function displayResults(data) {{
            if (data.output_image_path) {{
                document.getElementById('outputImage').src = '/' + data.output_image_path + '?t=' + new Date().getTime();
                document.getElementById('downloadLink').href = '/' + data.output_image_path;
                document.getElementById('countVal').innerText = data.products.length;
                document.getElementById('groupVal').innerText = new Set(data.products.map(p => p.group_id)).size;
                resultContainer.style.display = 'block';
                statsGrid.style.display = 'grid';
            }}
        }}
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process-url', methods=['POST'])
def process_url():
    try:
        data = request.get_json()
        resp = requests.get(data.get('url'), timeout=10)
        file_bytes = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to decode image from URL. It may be corrupt or an unsupported format."}), 400
        result = run_pipeline(image, "remote.jpg")
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_pipeline(image, filename):
    d, g = get_models()
    detected_items = d.detect_products(image)
    bboxes = [item['bbox'] for item in detected_items]
    group_ids = g.group_products(image, bboxes)
    final_products = []
    for item, gid in zip(detected_items, group_ids):
        final_products.append({"bbox": item['bbox'], "group_id": gid, "confidence": item['confidence']})
    safe_filename = secure_filename(filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"out_{safe_filename}")
    draw_and_save_visuals(image, final_products, output_path)
    return {"products": final_products, "output_image_path": output_path}

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Failed to decode uploaded image."}), 400
    result = run_pipeline(image, file.filename)
    return jsonify(result), 200

@app.route('/outputs/<filename>')
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # Match the port to what the user wants in README (8001)
    port = int(os.environ.get("PORT", 8001))
    app.run(host='0.0.0.0', port=port, debug=False)
