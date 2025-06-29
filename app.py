from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict
import json

app = Flask(__name__)

# === Load & Prepare Data ===
df = pd.read_csv("safety_data_updated.csv")
df = df.rename(columns={"area": "pin"}) if "area" in df.columns else df
df["pin"] = df["pin"].astype(str)

features = [
    "sex ratio", "r cases", "crimes", "wine shops", "men literacy",
    "porn access", "psyco", "desserted area", "ring roads",
    "slum areas", "season", "time of visit"
]

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df["safety_score"] = df[features].mean(axis=1)

safety_dict: Dict[str, float] = df.set_index("pin")["safety_score"].to_dict()
all_pins = set(safety_dict.keys())

# === Load Area Connectivity ===
with open("all_pincode_paths.json") as f:
    path_data = json.load(f)

graph = {}
for entry in path_data:
    for src, targets in entry.items():
        graph.setdefault(src, set()).update(targets)
        for tgt in targets:
            graph.setdefault(tgt, set()).add(src)

# === DFS Path Finder ===
def find_all_paths(graph, start: str, end: str, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]
    visited.add(start)

    if start == end:
        return [path]

    paths = []
    for node in graph.get(start, []):
        if node not in visited:
            newpaths = find_all_paths(graph, node, end, path, visited.copy())
            for newpath in newpaths:
                paths.append(newpath)
    return paths

# === Flask Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/route-finder')
def route_finder():
    return render_template('route_finder.html')

@app.route('/safety-map')
def safety_map():
    return render_template('safety_map.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/api/safe-route', methods=['POST'])
def safe_route():
    data = request.get_json()
    source = data['start']
    destination = data['end']

    if source not in all_pins or destination not in all_pins:
        return jsonify({'error': 'Invalid PIN codes'}), 400

    all_routes = find_all_paths(graph, source, destination)
    if not all_routes:
        return jsonify({'error': 'No routes found'}), 404

    scored_paths = []
    for path in all_routes:
        if all(p in safety_dict for p in path):
            mean_score = sum(safety_dict[p] for p in path) / len(path)
            score_level = (
                "🟢 Safe" if mean_score < 0.4 else
                "🟡 Moderate" if mean_score < 0.7 else
                "🔴 Risky"
            )
            scored_paths.append({
                "path": path,
                "mean_safety": round(mean_score, 3),
                "safety_code": score_level
            })

    if not scored_paths:
        return jsonify({'error': 'No valid paths with safety data'}), 404

    scored_paths = sorted(scored_paths, key=lambda x: x["mean_safety"])
    best = scored_paths[0]

    return jsonify({
        'best_path': best['path'],
        'mean_safety': best['mean_safety'],
        'safety_code': best['safety_code'],
        'all_paths': scored_paths
    })

if __name__ == '__main__':
    app.run(debug=True)

