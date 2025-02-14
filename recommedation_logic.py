def hybrid_recommendation(slots):
    current_hour = datetime.now().hour
    for slot in slots:
        # Fuzzy Logic Component
        fuzzy_score = 0
        if slot['status'] == 'free': fuzzy_score += 50
        fuzzy_score -= slot['distance'] * 0.5
        if slot['covered']: fuzzy_score += 20
        
        # AI/ML Component
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
            
        # Hybrid Combination (Weighted Average)
        slot['score'] = max(0, (fuzzy_score * 0.4) + (ai_score * 0.6))  # ← Hybrid Fusion
        slot['score_breakdown'] = {
            'fuzzy': fuzzy_score,
            'ai': ai_score,
            'combined': slot['score']
        }
    return sorted(slots, key=lambda x: x['score'], reverse=True)