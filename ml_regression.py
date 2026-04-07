import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import StandardScaler

def plot_model_insights(model, X_test, y_test, feature_names):
    """
    Generates presentation-ready graphs using Seaborn and Matplotlib.
    """
    print("Generating visual insights...")
    
    # Set a sleek, professional style
    sns.set_theme(style="darkgrid")
    
    # Create a wide figure with 2 side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: FEATURE IMPORTANCE ---
    # Extract importances from the trained model
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    # Draw a horizontal bar plot
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=axes[0], hue='Feature', palette='magma', legend=False)
    axes[0].set_title('What Triggers Gridlock? (Feature Importance)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Impact on Prediction', fontsize=12)
    axes[0].set_ylabel('')

    # --- PLOT 2: ACTUAL VS. PREDICTED ---
    predictions = model.predict(X_test)
    
    # Scatter plot of reality vs AI guess
    axes[1].scatter(y_test, predictions, alpha=0.4, color='#00E5FF', edgecolor='black')
    
    # Draw the "Perfect Prediction" line (y = x)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], color='#FF0055', linestyle='--', linewidth=3, label='Perfect Prediction')
    
    axes[1].set_title('AI Accuracy: Actual vs. Predicted Time', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Actual Seconds To Gridlock', fontsize=12)
    axes[1].set_ylabel('AI Predicted Seconds', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    # Save the figure so you can put it in your presentation slides
    plt.savefig('model_insights.png', dpi=300)
    plt.show()

def train_and_save_model(csv_path="/home/lappy/Documents/Learnings/Winning/baseline_chaos_data.csv", model_path="jam_predictor.pkl"):
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        return

    # Map text weather to numbers
    weather_mapping = {'Clear': 0, 'Rain': 1, 'Fog': 2}
    if df['Weather_Condition'].dtype == object:
        df['Weather_Condition'] = df['Weather_Condition'].map(weather_mapping)

    features = [
        'Weather_Condition', 'Hour_Of_Day', 'Density_Lane1', 
         'Density_Lane3', 'Total_Throughput', 'Global_Speed_Variance', 
        'Global_Avg_Speed','Aggression_Profile','Compliance_Profile','Current_Impatience'
    ]
    target = 'Seconds_To_Gridlock' 
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model Error (RMSE): +/- {rmse:.2f} seconds")

    # --- TRIGGER THE GRAPHS HERE ---
    plot_model_insights(rf_model, X_test, y_test, features)

    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print("Done! Model is ready for the dashboard.")
    

def predict_jam(current_state_dict, model_path="jam_predictor.pkl"):
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
    except FileNotFoundError:
         return "Error: Model not trained yet."

    state_df = pd.DataFrame([current_state_dict])
    prediction = loaded_model.predict(state_df)[0]
    return prediction

def predict_jam_from_csv(csv_path, model_path="jam_predictor.pkl"):
    """
    Reads an entire CSV file of traffic states and predicts the time to gridlock
    for every single row simultaneously.
    """
    try:
        # 1. Wake up the AI Brain
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
    except FileNotFoundError:
         return "Error: Model not trained yet."

    # 2. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"Error: Could not find the file '{csv_path}'."

    # 3. Define the exact features the AI expects (Must match training!)
    features = [
        'Weather_Condition', 'Hour_Of_Day', 'Hard_Braking_Count', 
        'Global_Speed_Variance', 'Global_Avg_Speed', 'Footpath_Count', 
        'Current_Impatience', 'Aggression_Profile', 'Compliance_Profile'
    ]

    # 4. Safety Check: Make sure the CSV actually has these columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        return f"Error: The CSV is missing these required columns -> {missing_cols}"

    # 5. Filter the DataFrame to ONLY the columns the AI needs
    X_live = df[features]

    # 6. Predict the entire CSV at once (Batch Prediction)
    predictions = loaded_model.predict(X_live)

    # 7. Add the predictions as a brand new column to the DataFrame
    df['Predicted_Seconds_To_Gridlock'] = predictions
    
    # Return the updated DataFrame
    return df['Predicted_Seconds_To_Gridlock']

if __name__ == "__main__":
    train_and_save_model()
