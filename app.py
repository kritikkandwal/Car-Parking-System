# app.py (AI-Enhanced)
from flask import Flask, render_template, jsonify, request, redirect
import random
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# AI Model Setup
class ParkingRecommender:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['distance', 'hour', 'covered']),
                ('cat', OneHotEncoder(), ['type'])
            ])
        self.is_trained = False

    def train(self, data):
        df = pd.DataFrame(data)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        X = self.preprocessor.fit_transform(df[['type', 'distance', 'covered', 'hour']])
        y = df['duration']
        self.model.fit(X, y)
        self.is_trained = True
        joblib.dump(self.model, 'ai_model.pkl')

recommender = ParkingRecommender()

# Simulated Database with AI Features
parking_lots = [
    {"id": i, "status": "free", "type": random.choice(['EV', 'Compact', 'SUV']), 
     "distance": random.randint(1, 100), "price": 5.0, "covered": random.choice([True, False]),
     "ai_score": 0, "score_breakdown": {}}
    for i in range(1, 21)
]

reservations = {}
user_account = {
    "balance": 100.0,
    "reservations": []
}

def calculate_dynamic_price(slot):
    hour = datetime.now().hour
    base_price = 5.0
    if 7 <= hour < 9 or 17 <= hour < 19:
        base_price *= 1.5
    if slot['covered']:
        base_price *= 1.2
    return round(base_price + (0.1 * slot['distance']), 2)

def hybrid_recommendation(slots):
    current_hour = datetime.now().hour
    for slot in slots:
        fuzzy_score = 0
        if slot['status'] == 'free': fuzzy_score += 50
        fuzzy_score -= slot['distance'] * 0.5
        if slot['covered']: fuzzy_score += 20
        
        if recommender.is_trained:
            input_data = pd.DataFrame([{
                'type': slot['type'],
                'distance': slot['distance'],
                'covered': slot['covered'],
                'hour': current_hour,
                'timestamp': datetime.now().isoformat()
            }])
            X = recommender.preprocessor.transform(input_data)
            ai_score = recommender.model.predict(X)[0]
        else:
            ai_score = 0
            
        slot['score'] = max(0, (fuzzy_score * 0.4) + (ai_score * 0.6))
        slot['score_breakdown'] = {
            'fuzzy': fuzzy_score,
            'ai': ai_score,
            'combined': slot['score']
        }
    return sorted(slots, key=lambda x: x['score'], reverse=True)

@app.route('/')
def index():
    return render_template('index.html', user_balance=user_account["balance"])

@app.route('/profile.html')
def profile():
    return render_template('profile.html', reservations=user_account["reservations"])

@app.route('/styles.css')
def styles():
    return app.send_static_file('styles.css')

@app.route('/reservation.html')
def reservation():
    return render_template('reservation.html')

@app.route('/submit-reservation', methods=['POST'])
def submit_reservation():
    data = request.json
    slot_id = int(data['slotId'])
    
    slot = next(s for s in parking_lots if s['id'] == slot_id)
    training_data = {
        'timestamp': datetime.now().isoformat(),
        'type': slot['type'],
        'distance': slot['distance'],
        'covered': slot['covered'],
        'duration': data['duration'],
        'vehicle': data['vehicle']
    }
    with open('training_data.json', 'a') as f:
        f.write(json.dumps(training_data) + '\n')
    
    reservation = {
        "slot_id": slot_id,
        "datetime": datetime.now().isoformat(),
        "duration": data['duration'],
        "vehicle": data['vehicle'],
        "location": "Downtown Lot A",
        "cost": slot['price'] * (int(data['duration']) / 60)
    }
    user_account["reservations"].append(reservation)
    return jsonify({"status": "success"})

@app.route('/parking-data')
def get_parking_data():
    try:
        with open('training_data.json', 'r') as f:
            data = [json.loads(line) for line in f]
        if len(data) > 50 and not recommender.is_trained:
            recommender.train(data)
    except FileNotFoundError:
        pass
    
    for slot in parking_lots:
        slot['price'] = calculate_dynamic_price(slot)
        if random.random() < 0.01 and slot['status'] == 'free':
            slot['status'] = 'occupied'
    return jsonify(hybrid_recommendation(parking_lots))

@app.route('/reserve/<int:slot_id>', methods=['POST'])
def reserve(slot_id):
    if slot_id < 1 or slot_id > len(parking_lots):
        return jsonify({"status": "failed", "message": "Invalid slot ID"})
    
    slot = parking_lots[slot_id - 1]
    
    if slot['status'] == 'free' and user_account["balance"] >= slot['price']:
        user_account["balance"] -= slot['price']
        slot['status'] = 'reserved'
        reservations[slot_id] = time.time() + 300
        
        return jsonify({
            "status": "success",
            "until": reservations[slot_id],
            "new_balance": user_account["balance"]
        })
    
    return jsonify({"status": "failed", "message": "Insufficient balance or slot not free"})

if __name__ == '__main__':
    app.run(debug=True)