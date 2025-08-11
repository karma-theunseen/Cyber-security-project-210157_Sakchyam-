"""
AI-Based Phishing Detection for Kathmandu College Students
Final Year Dissertation Project
Run this file in VS Code with Python 3.8+ environment
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, 
                           confusion_matrix, 
                           accuracy_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set up visualization style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_prepare_data():
    """Load and prepare the phishing dataset"""
    try:
        # Load dataset from UCI repository (replace with local data later)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
        df = pd.read_csv(url, skiprows=45, header=None)
        
        # Define column names based on dataset documentation
        columns = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report', 'Result'
        ]
        df.columns = columns
        
        # Convert target to binary (phishing = 0, legitimate = 1)
        df['Result'] = df['Result'].map({-1: 0, 1: 1})
        
        # Add simulated Nepal-specific features (replace with actual data)
        np.random.seed(42)
        df['nepali_keyword_usage'] = np.random.uniform(0, 1, size=len(df))
        df['academic_theme'] = np.random.randint(0, 3, size=len(df))
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e} - pgishing_ktm_detection.py:55")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    try:
        # Separate features and target
        X = df.drop('Result', axis=1)
        y = df['Result']
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        print(f"Error in preprocessing: {e} - pgishing_ktm_detection.py:77")
        return None, None, None, None, None

def train_model(X_train, y_train):
    """Train the Random Forest classifier"""
    try:
        # Initialize model with balanced class weights
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    
    except Exception as e:
        print(f"Error training model: {e} - pgishing_ktm_detection.py:99")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Print classification report
        print("\nModel Evaluation Report: - pgishing_ktm_detection.py:109")
        print("======================== - pgishing_ktm_detection.py:110")
        print(classification_report(y_test, y_pred))
        
        # Print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy:.2%} - pgishing_ktm_detection.py:115")
        
        # Generate confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Phishing', 'Legitimate'],
                    yticklabels=['Phishing', 'Legitimate'])
        plt.title('Phishing Detection Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Saved confusion matrix to 'confusion_matrix.png' - pgishing_ktm_detection.py:128")
        
        return y_pred
    
    except Exception as e:
        print(f"Error evaluating model: {e} - pgishing_ktm_detection.py:133")
        return None

def analyze_features(model, feature_names):
    """Analyze and visualize feature importance"""
    try:
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create importance dataframe
        feat_imp = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot top 15 features
        plt.figure()
        sns.barplot(x='Importance', y='Feature', 
                   data=feat_imp.head(15))
        plt.title('Top 15 Important Features for Phishing Detection')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Saved feature importance plot to 'feature_importance.png' - pgishing_ktm_detection.py:157")
        
        # Save to CSV for educational strategy development
        feat_imp.to_csv('feature_importances.csv', index=False)
        print("Saved feature importances to 'feature_importances.csv' - pgishing_ktm_detection.py:161")
        
        return feat_imp
    
    except Exception as e:
        print(f"Error in feature analysis: {e} - pgishing_ktm_detection.py:166")
        return None

def generate_recommendations(feature_imp):
    """Generate cybersecurity education recommendations"""
    try:
        print("\nCybersecurity Education Recommendations for Kathmandu Colleges: - pgishing_ktm_detection.py:172")
        print("=============================================================== - pgishing_ktm_detection.py:173")
        
        # Get top features
        top_features = feature_imp.head(5)['Feature'].tolist()
        
        # Generate recommendations based on features
        recommendations = []
        
        if 'SSLfinal_State' in top_features:
            recommendations.append(
                "1. Teach students to verify SSL certificates and look for HTTPS in URLs"
            )
        
        if 'URL_Length' in top_features:
            recommendations.append(
                "2. Train students to identify suspiciously long or short URLs"
            )
        
        if 'having_Sub_Domain' in top_features:
            recommendations.append(
                "3. Educate about subdomain spoofing (e.g., 'tribhuvan.university.fake.com')"
            )
        
        # Add Nepal-specific recommendations
        recommendations.extend([
            "4. Develop Nepali-language examples of phishing attempts",
            "5. Create simulations using local scam patterns (scholarship, job offers)",
            "6. Focus on mobile phishing (common on platforms like Khalti, eSewa)",
            "7. Conduct regular phishing simulation exercises",
            "8. Establish a cybersecurity awareness campaign in Social Engineering and Nepali"
        ])
        
        # Print recommendations
        for rec in recommendations:
            print(f"{rec} - pgishing_ktm_detection.py:207")
        
        # Save to file
        with open('education_recommendations.txt', 'w') as f:
            f.write("\n".join(recommendations))
        print("\nSaved recommendations to 'education_recommendations.txt' - pgishing_ktm_detection.py:212")
    
    except Exception as e:
        print(f"Error generating recommendations: {e} - pgishing_ktm_detection.py:215")

def save_model_and_artifacts(model, scaler):
    """Save model and preprocessing artifacts"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('model_artifacts', exist_ok=True)
        
        # Save model
        joblib.dump(model, 'model_artifacts/phishing_model.pkl')
        
        # Save scaler
        joblib.dump(scaler, 'model_artifacts/scaler.pkl')
        
        print("\nSaved model and artifacts to 'model_artifacts/' directory - pgishing_ktm_detection.py:229")
    
    except Exception as e:
        print(f"Error saving artifacts: {e} - pgishing_ktm_detection.py:232")

def main():
    """Main execution function"""
    print("Starting Phishing Detection Project for Kathmandu Students - pgishing_ktm_detection.py:236")
    print("========================================================= - pgishing_ktm_detection.py:237")
    
    # Step 1: Load and prepare data
    print("\n[1/5] Loading and preparing data... - pgishing_ktm_detection.py:240")
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Step 2: Preprocess data
    print("\n[2/5] Preprocessing data... - pgishing_ktm_detection.py:246")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    if X_train is None:
        return
    
    # Step 3: Train model
    print("\n[3/5] Training model... - pgishing_ktm_detection.py:252")
    model = train_model(X_train, y_train)
    if model is None:
        return
    
    # Step 4: Evaluate model
    print("\n[4/5] Evaluating model... - pgishing_ktm_detection.py:258")
    y_pred = evaluate_model(model, X_test, y_test)
    if y_pred is None:
        return
    
    # Step 5: Analyze features and generate recommendations
    print("\n[5/5] Analyzing features and generating recommendations... - pgishing_ktm_detection.py:264")
    feature_imp = analyze_features(model, df.drop('Result', axis=1).columns)
    if feature_imp is not None:
        generate_recommendations(feature_imp)
    
    # Save model and artifacts
    save_model_and_artifacts(model, scaler)
    
    print("\nProcess completed successfully! - pgishing_ktm_detection.py:272")

if __name__ == "__main__":
    main()