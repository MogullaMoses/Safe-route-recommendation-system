"""
Safe Route Prediction System — Flask Backend
Uses: data/accidents.csv (Indian Road Accident Dataset, 3000 records)
Run : python app.py  →  open http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import math, os, requests, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(__file__), "data", "accidents.csv")
ORS_API_KEY    = os.getenv("ORS_API_KEY", "")   # optional – openrouteservice.org
RISK_RADIUS_KM = 80    # radius around each route segment to scan for accidents
SEGMENT_COUNT  = 8     # how many segments to split each route into

# ── CITY / STATE COORDINATES (lookup table since dataset has no lat/lon) ──────
CITY_COORDS = {
    "New Delhi": (28.6139, 77.2090), "Delhi": (28.6517, 77.2219),
    "Dwarka": (28.5921, 77.0460), "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319), "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739), "Prayagraj": (25.4358, 81.8463),
    "Mumbai": (19.0760, 72.8777), "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882), "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433), "Surat": (21.1702, 72.8311),
    "Ahmedabad": (23.0225, 72.5714), "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460), "Trichy": (10.7905, 78.7047),
    "Bengaluru": (12.9716, 77.5946), "Mysore": (12.2958, 76.6394),
    "Hubli": (15.3647, 75.1240), "Mangalore": (12.9141, 74.8560),
    "Jaipur": (26.9124, 75.7873), "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125), "Kota": (25.2138, 75.8648),
    "Kolkata": (22.5726, 88.3639), "Howrah": (22.5958, 88.2636),
    "Siliguri": (26.7271, 88.3953), "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577), "Gwalior": (26.2183, 78.1828),
    "Jabalpur": (23.1815, 79.9864), "Patna": (25.5941, 85.1376),
    "Ranchi": (23.3441, 85.3096), "Bhubaneswar": (20.2961, 85.8245),
    "Visakhapatnam": (17.6868, 83.2185), "Vijayawada": (16.5062, 80.6480),
    "Hyderabad": (17.3850, 78.4867), "Guntur": (16.3067, 80.4365),
    "Chandigarh": (30.7333, 76.7794), "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573), "Shimla": (31.1048, 77.1734),
    "Srinagar": (34.0837, 74.7973), "Jammu": (32.7266, 74.8570),
    "Dehradun": (30.3165, 78.0322), "Haridwar": (29.9457, 78.1642),
    "Guwahati": (26.1445, 91.7362), "Imphal": (24.8170, 93.9368),
    "Panaji": (15.4909, 73.8278), "Goa": (15.2993, 74.1240),
    "Thiruvananthapuram": (8.5241, 76.9366), "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
}

STATE_COORDS = {
    "Uttar Pradesh": (26.8467, 80.9462), "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6517, 77.2219), "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139), "Rajasthan": (27.0238, 74.2179),
    "Andhra Pradesh": (15.9129, 79.7400), "Gujarat": (22.2587, 71.1924),
    "West Bengal": (22.9868, 87.8550), "Madhya Pradesh": (22.9734, 78.6569),
    "Bihar": (25.0961, 85.3131), "Odisha": (20.9517, 85.0985),
    "Telangana": (17.1232, 79.2088), "Punjab": (31.1471, 75.3412),
    "Haryana": (29.0588, 76.0856), "Jharkhand": (23.6102, 85.2799),
    "Assam": (26.2006, 92.9376), "Chhattisgarh": (21.2787, 81.8661),
    "Kerala": (10.8505, 76.2711), "Uttarakhand": (30.0668, 79.0193),
    "Himachal Pradesh": (31.1048, 77.1734), "Goa": (15.2993, 74.1240),
    "Jammu and Kashmir": (33.7782, 76.5762), "Sikkim": (27.5330, 88.5122),
    "Tripura": (23.9408, 91.9882), "Meghalaya": (25.4670, 91.3662),
    "Manipur": (24.6637, 93.9063), "Nagaland": (26.1584, 94.5624),
    "Arunachal Pradesh": (28.2180, 94.7278), "Mizoram": (23.1645, 92.9376),
    "Puducherry": (11.9416, 79.8083), "Chandigarh": (30.7333, 76.7794),
    "Unknown": (20.5937, 78.9629),
}

def get_coords(city, state):
    if city and city != "Unknown" and city in CITY_COORDS:
        return CITY_COORDS[city]
    return STATE_COORDS.get(state, (20.5937, 78.9629))

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Assign coordinates from lookup
    coords = df.apply(lambda r: get_coords(r["City Name"], r["State Name"]), axis=1)
    df["Latitude"]  = coords.apply(lambda c: c[0])
    df["Longitude"] = coords.apply(lambda c: c[1])

    # Add small random jitter so nearby cities don't stack on same pixel
    np.random.seed(42)
    df["Latitude"]  += np.random.uniform(-0.8, 0.8, len(df))
    df["Longitude"] += np.random.uniform(-0.8, 0.8, len(df))

    # Normalise severity label  (Serious → Major for display)
    df["Severity_Display"] = df["Accident Severity"].replace({"Serious": "Major"})

    # Numeric month for sorting
    month_order = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                   "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    df["Month_Num"] = df["Month"].map(month_order).fillna(0).astype(int)

    log.info(f"Loaded {len(df)} accident records")
    return df

DF = load_data()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(min(a, 1)))

def accidents_near(lat, lon, radius_km=RISK_RADIUS_KM):
    dist = DF.apply(lambda r: haversine(lat, lon, r["Latitude"], r["Longitude"]), axis=1)
    return DF[dist <= radius_km]

def risk_label(count):
    if count <= 3:   return "safe"
    if count <= 10:  return "moderate"
    return "danger"

RISK_COLORS = {"safe":"#22c55e", "moderate":"#f59e0b", "danger":"#ef4444"}

def score_route(waypoints):
    segments, n = [], len(waypoints)
    step = max(1, n // SEGMENT_COUNT)
    indices = list(range(0, n, step))[:SEGMENT_COUNT]
    for i, idx in enumerate(indices):
        nxt = indices[i+1] if i+1 < len(indices) else n-1
        mid = (idx + nxt) // 2
        mlat, mlon = waypoints[mid]
        nearby = accidents_near(mlat, mlon)
        count  = len(nearby)
        fatal  = int((nearby["Accident Severity"] == "Fatal").sum())
        score  = count + fatal * 2
        label  = risk_label(count)
        segments.append({
            "coords":    [[p[0], p[1]] for p in waypoints[idx:nxt+1]],
            "risk":      label,
            "color":     RISK_COLORS[label],
            "accidents": count,
            "fatal":     fatal,
            "score":     score,
        })
    return segments, sum(s["score"] for s in segments)

def geocode(place):
    try:
        # Check our local lookup first (faster, no network needed)
        for name, coord in {**CITY_COORDS, **{k: v for k, v in STATE_COORDS.items()}}.items():
            if name.lower() == place.lower():
                return coord
        # Fall back to Nominatim
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{place}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "SafeRoutePredictionSystem/2.0"},
            timeout=6
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        log.error(f"Geocode error for '{place}': {e}")
    return None

def get_routes(src, dst):
    """ORS if key present, otherwise interpolated fallback."""
    if ORS_API_KEY:
        try:
            resp = requests.post(
                "https://api.openrouteservice.org/v2/directions/driving-car",
                headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                json={
                    "coordinates": [[src[1], src[0]], [dst[1], dst[0]]],
                    "alternative_routes": {"target_count": 3, "weight_factor": 1.6}
                },
                timeout=10
            )
            routes = []
            for route in resp.json().get("routes", []):
                coords = [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]]
                routes.append({
                    "coords":      coords,
                    "distance_km": round(route["summary"]["distance"] / 1000, 1),
                    "duration_h":  round(route["summary"]["duration"] / 3600, 2),
                })
            if routes:
                return routes
        except Exception as e:
            log.error(f"ORS error: {e}")
    return _fallback_routes(src, dst)

def _fallback_routes(src, dst):
    import math
    dist = haversine(*src, *dst)
    def interp(jitter_lat=0.0, jitter_lon=0.0, steps=20):
        pts = []
        for i in range(steps+1):
            t = i / steps
            lat = src[0] + (dst[0]-src[0])*t + jitter_lat * math.sin(math.pi*t)
            lon = src[1] + (dst[1]-src[1])*t + jitter_lon * math.sin(math.pi*t)
            pts.append([round(lat,4), round(lon,4)])
        return pts
    return [
        {"coords": interp(0.4, -0.3),  "distance_km": round(dist,1),        "duration_h": round(dist/65,2)},
        {"coords": interp(-0.5, 0.4),  "distance_km": round(dist*1.07,1),   "duration_h": round(dist*1.07/65,2)},
        {"coords": interp(0.15, 0.2),  "distance_km": round(dist*1.03,1),   "duration_h": round(dist*1.03/65,2)},
    ]

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dashboard")
def dashboard():
    total        = len(DF)
    danger_state = DF["State Name"].value_counts().idxmax()
    danger_city  = DF[DF["City Name"] != "Unknown"]["City Name"].value_counts().idxmax()
    fatal_count  = int((DF["Accident Severity"] == "Fatal").sum())
    alcohol_pct  = round(100 * (DF["Alcohol Involvement"] == "Yes").sum() / total, 1)
    return jsonify({
        "total_accidents":      total,
        "most_dangerous_state": danger_state,
        "high_risk_city":       danger_city,
        "fatal_accidents":      fatal_count,
        "alcohol_involved_pct": alcohol_pct,
    })

@app.route("/api/accidents/state")
def by_state():
    top = DF["State Name"].value_counts().head(8)
    return jsonify([{"state": s, "count": int(c)} for s, c in top.items()])

@app.route("/api/accidents/severity")
def by_severity():
    counts = DF["Severity_Display"].value_counts()
    return jsonify([{"severity": s, "count": int(c)} for s, c in counts.items()])

@app.route("/api/accidents/monthly")
def monthly():
    mn = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
    counts = DF.groupby("Month_Num").size()
    result = []
    for i, m in enumerate(mn, 1):
        result.append({"month": m[:3], "count": int(counts.get(i, 0))})
    return jsonify(result)

@app.route("/api/accidents/weather")
def by_weather():
    counts = DF["Weather Conditions"].value_counts()
    return jsonify([{"weather": w, "count": int(c)} for w, c in counts.items()])

@app.route("/api/accidents/roadtype")
def by_roadtype():
    counts = DF["Road Type"].value_counts()
    return jsonify([{"road": r, "count": int(c)} for r, c in counts.items()])

@app.route("/api/hotspots")
def hotspots():
    """Top accident clusters – group by rounded coordinates."""
    df2 = DF.copy()
    df2["lat_r"] = df2["Latitude"].round(1)
    df2["lon_r"] = df2["Longitude"].round(1)
    grp = df2.groupby(["lat_r","lon_r"]).agg(
        accidents=("Accident Severity","count"),
        fatal=("Accident Severity", lambda x:(x=="Fatal").sum()),
        state=("State Name","first"),
        city=("City Name","first"),
        road=("Road Type","first"),
        weather=("Weather Conditions","first"),
        casualties=("Number of Casualties","sum"),
    ).reset_index()
    top = grp.sort_values("accidents", ascending=False).head(40)
    return jsonify([{
        "lat":       float(r["lat_r"]),
        "lon":       float(r["lon_r"]),
        "accidents": int(r["accidents"]),
        "fatal":     int(r["fatal"]),
        "state":     r["state"],
        "city":      r["city"],
        "road":      r["road"],
        "weather":   r["weather"],
        "casualties":int(r["casualties"]),
        "risk":      risk_label(int(r["accidents"])),
    } for _, r in top.iterrows()])

@app.route("/api/route", methods=["POST"])
def find_route():
    body = request.get_json()
    if not body:
        return jsonify({"error": "JSON body required"}), 400
    source = body.get("source","").strip()
    dest   = body.get("destination","").strip()
    if not source or not dest:
        return jsonify({"error": "source and destination required"}), 400

    src_coord = geocode(source)
    dst_coord = geocode(dest)
    if not src_coord:
        return jsonify({"error": f"Could not find location: '{source}'"}), 400
    if not dst_coord:
        return jsonify({"error": f"Could not find location: '{dest}'"}), 400

    log.info(f"Route: {source}{src_coord} → {dest}{dst_coord}")
    raw_routes = get_routes(src_coord, dst_coord)

    # Step 1: score all routes first (no names yet)
    analyzed = []
    for i, rr in enumerate(raw_routes[:3]):
        segs, total_score = score_route(rr["coords"])
        analyzed.append({
            "id":          chr(65+i),
            "distance_km": rr["distance_km"],
            "duration_h":  rr["duration_h"],
            "total_score": total_score,
            "segments":    segs,
        })

    # Step 2: find safest (lowest score) and fastest (shortest distance)
    safest_idx  = min(range(len(analyzed)), key=lambda i: analyzed[i]["total_score"])
    fastest_idx = min(range(len(analyzed)), key=lambda i: analyzed[i]["distance_km"])

    # Step 3: assign names and colors based on actual scores
    for i in range(len(analyzed)):
        if i == safest_idx:
            analyzed[i]["name"]  = "Safest Route"
            analyzed[i]["color"] = "#22c55e"   # green
        elif i == fastest_idx:
            analyzed[i]["name"]  = "Fastest Route"
            analyzed[i]["color"] = "#3b82f6"   # blue
        else:
            analyzed[i]["name"]  = "Balanced Route"
            analyzed[i]["color"] = "#6b7280"   # grey

    log.info(f"Scores: {[r['total_score'] for r in analyzed]} → Safest=Route {chr(65+safest_idx)}, Fastest=Route {chr(65+fastest_idx)}")

    # Collect hotspot details along the safest route
    all_pts = [pt for seg in analyzed[safest_idx]["segments"] for pt in seg["coords"]]
    seen, route_hotspots = set(), []
    for pt in all_pts[::3]:
        nearby = accidents_near(pt[0], pt[1], radius_km=RISK_RADIUS_KM)
        for _, row in nearby.iterrows():
            key = (round(row["Latitude"],1), round(row["Longitude"],1))
            if key not in seen:
                seen.add(key)
                route_hotspots.append({
                    "lat":       key[0], "lon": key[1],
                    "severity":  row["Accident Severity"],
                    "road":      row["Road Type"],
                    "weather":   row["Weather Conditions"],
                    "city":      row["City Name"],
                    "state":     row["State Name"],
                    "casualties":int(row["Number of Casualties"]),
                    "fatalities":int(row["Number of Fatalities"]),
                })
    route_hotspots = route_hotspots[:25]

    return jsonify({
        "source":       source,
        "destination":  dest,
        "src_coord":    list(src_coord),
        "dst_coord":    list(dst_coord),
        "routes":       analyzed,
        "safest_index": safest_idx,
        "hotspots":     route_hotspots,
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
