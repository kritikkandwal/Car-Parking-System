# app.py
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

@app.route('/live-map')
def live_map():
    return render_template('live_map.html')

# this part is basically making the slot 
parking_lots = [
    {"id": i, 
     "status": "free", 
     "type": random.choice(['EV', 'Compact', 'SUV']), 
     "distance": random.randint(1, 100), 
     "price": 5.0, 
     "covered": random.choice([True, False]),
     "latitude": 31.2244 + random.uniform(-0.001, 0.001),
     "longitude": 75.7708 + random.uniform(-0.001, 0.001),
     "ai_score": 0, 
     "score_breakdown": {}}
    for i in range(1, 21) #by increasing this we can iincrease the no of slot
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

@app.route('/')  # Route for the front page
def frontpage():
    return render_template('frontpage.html')

@app.route('/wapis')  # this is just for profile.html to come back to index.html
def wapis_page():
    return render_template('index.html', user_balance=user_account["balance"])

@app.route('/start')  # Route for the main page (index.html)
def index():
    return render_template('index.html', user_balance=user_account["balance"])

@app.route('/profile.html')
def profile():
    return render_template('profile.html', 
        reservations=user_account["reservations"],
        user_balance=user_account["balance"])

@app.route('/styles.css')
def styles():
    return app.send_static_file('styles.css')

@app.route('/reservation.html')
def reservation():
    return render_template('reservation.html',
        user_balance=user_account["balance"])

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

@app.route('/slot/<int:slot_id>')
def get_slot(slot_id):
    slot = next((s for s in parking_lots if s['id'] == slot_id), None)
    return jsonify(slot if slot else {"error": "Slot not found"})

@app.route('/confirm-payment', methods=['POST'])
def confirm_payment():
    data = request.json
    slot = next(s for s in parking_lots if s['id'] == int(data['slotId']))
    
    # Validate payment
    expected_cost = slot['price'] * (int(data['duration']) / 60)
    if abs(float(data['amount']) - expected_cost) > 0.01:
        return jsonify({"status": "failed", "message": "Payment validation failed"})

    # Update system
    user_account["balance"] -= float(data['amount'])
    slot['status'] = 'reserved'
    user_account["reservations"].append({
        "slot_id": slot['id'],
        "datetime": datetime.now().isoformat(),
        "duration": data['duration'],
        "vehicle": data['vehicle'],
        "location": "Downtown Lot A",
        "cost": float(data['amount'])
    })
    
    # Updated app.py endpoint
@app.route('/confirm-reservation', methods=['POST'])
def confirm_reservation():
    data = request.json
    slot_id = int(data['slotId'])
    duration = int(data['duration'])
    vehicle = data['vehicle']
    operator = data['operator']
    amount = float(data['amount'])

    slot = next((s for s in parking_lots if s['id'] == slot_id), None)
    
    if not slot:
        return jsonify({"status": "failed", "message": "Invalid slot ID"})
    
    if slot['status'] != 'free':
        return jsonify({"status": "failed", "message": "Slot already occupied"})
    
    if user_account["balance"] < amount:
        return jsonify({"status": "failed", "message": "Insufficient quantum energy"})

    # Update system state
    user_account["balance"] -= amount
    slot['status'] = 'reserved'
    
    # Create reservation record
    reservation = {
        "slot_id": slot_id,
        "datetime": datetime.now().isoformat(),
        "duration": duration,
        "vehicle": vehicle,
        "operator": operator,
        "location": "Quantum Bay #7",
        "cost": round(amount, 2)
    }
    user_account["reservations"].append(reservation)
    
    # Add to training data
    training_data = {
        'timestamp': datetime.now().isoformat(),
        'type': slot['type'],
        'distance': slot['distance'],
        'covered': slot['covered'],
        'duration': duration,
        'vehicle': vehicle
    }
    with open('training_data.json', 'a') as f:
        f.write(json.dumps(training_data) + '\n')

    return jsonify({
        "status": "success",
        "new_balance": user_account["balance"],
        "reservation": reservation
    })

    return jsonify({"status": "success", "new_balance": user_account["balance"]})
if __name__ == '__main__':
    app.run(debug=True)
