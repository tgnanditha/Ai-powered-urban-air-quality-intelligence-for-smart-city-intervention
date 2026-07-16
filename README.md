# AirSense — Urban Air Quality Intelligence Platform
### ET AI Hackathon 2026 · Problem Statement 5

---

## Quick Start (Demo — no API keys needed)

Just open `index.html` in any browser. Done. You'll see:
- Live AQI map of Bengaluru (demo data, CPCB-structured)
- All 8 monitoring stations with real ward names
- Source attribution for every station
- 24-hour AI forecast chart
- Ranked enforcement priority list
- 4 city support: Bengaluru, Delhi, Mumbai, Kolkata
- Kannada attribution text

---

## Full Stack Setup

### 1. Install Python backend
```bash
pip install fastapi uvicorn httpx anthropic redis numpy pandas scikit-learn
```

### 2. Set API keys (optional — demo works without)
```bash
export ANTHROPIC_API_KEY=your_key_here
export CPCB_API_KEY=your_key_from_data.gov.in
export NASA_FIRMS_KEY=your_key_from_firms.modaps.eosdis.nasa.gov
```

### 3. Run backend
```bash
uvicorn backend:app --reload --port 8000
```

### 4. Connect frontend to backend
In `index.html`, change the `DEMO_MODE` flag at top of the script section to point to `http://localhost:8000`.

### 5. API endpoints
```
GET /api/stations/{city}           → All station readings
GET /api/attribution/{city}/{id}   → Agent 1: Source attribution
GET /api/forecast/{city}/{id}      → Agent 2: 24h AQI forecast
GET /api/enforcement/{city}        → Agent 3: Enforcement priorities
GET /api/city-summary/{city}       → Full dashboard data
GET /api/alert/{city}              → Active alerts (AQI > 150)
```

---

## Free API Keys to Get

| Source | What | URL | Time |
|--------|------|-----|------|
| CPCB (data.gov.in) | Live AQI station data | api.data.gov.in | 5 min |
| NASA FIRMS | Satellite thermal anomalies | firms.modaps.eosdis.nasa.gov/api | Instant |
| Anthropic Claude | AI attribution narratives | console.anthropic.com | 5 min |
| OpenWeatherMap | Wind + humidity forecasts | openweathermap.org/api | 5 min |

---

## Architecture

```
Data Sources          Backend Agents        Frontend
────────────          ──────────────        ────────
CPCB CAAQMS  ──┐
NASA FIRMS   ──┤─→  Agent 1: Attributor ──┐
IMD Weather  ──┤─→  Agent 2: Forecaster ──┼──→ Leaflet.js Map
BBMP Data    ──┤─→  Agent 3: Enforcer   ──┤    Attribution Panel
Traffic API  ──┘                          │    Forecast Chart
                    PostgreSQL/PostGIS ───┘    Enforcement List
                    Redis Cache
                    Claude API
```

---

## Judging Criteria Coverage

| Criterion | Weight | How AirSense wins |
|-----------|--------|-------------------|
| Innovation | 25% | Source attribution in natural language — no existing tool does this for Indian cities at ward level |
| Business Impact | 25% | 1.67M premature deaths/yr, live Bengaluru data, judges can see the problem outside the window |
| Technical Excellence | 20% | Multi-agent + geospatial ML + Claude API + real CPCB data + PostGIS |
| Scalability | 15% | Same architecture works for 900+ CAAQMS cities — add a city in 1 line |
| User Experience | 15% | Three personas (citizen/inspector/commissioner), mobile-first, Kannada alerts |

---

## Demo Script (3 minutes)

1. **Open the map** — "Bengaluru's AQI right now is 172 in Hebbal. Do you know why?"
2. **Click Hebbal station** — Show attribution: 42% construction, 35% traffic, live permit reference
3. **Click Peenya** — "AQI 278, Very Poor. Satellite shows thermal anomaly. Three unlicensed units."
4. **Switch to Enforcement layer** — "Here's where KSPCB inspectors should go today, ranked by impact."
5. **Click an enforcement pin** — Show evidence PDF download
6. **Scrub the forecast** — "At 6PM, AQI in Silk Board will reach 180. We can warn people now."
7. **Switch to Delhi** — "Same platform, new city, one dropdown."
8. **Close**: "Every day we don't act, 4,600 people die from air pollution in India. AirSense gives cities the intelligence to act."
