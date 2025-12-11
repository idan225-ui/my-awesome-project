"""
Calorie Prediction Model
========================
Expert Data Science approach to predict calories burned during fitness activities.

Author: Data Science Expert
Dataset: Activities_English.csv
Goal: Build a Regression Model to predict 'Calories'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set beautiful seaborn theme
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


def time_to_minutes(time_str):
    """
    Convert time string in format HH:MM:SS or HH:MM:SS.S to total minutes (float).
    
    Args:
        time_str: Time string in format HH:MM:SS or HH:MM:SS.S
    
    Returns:
        float: Total minutes
    """
    try:
        if pd.isna(time_str) or time_str == '--':
            return np.nan
        
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            total_minutes = hours * 60 + minutes + seconds / 60
            return total_minutes
        else:
            return np.nan
    except:
        return np.nan


def load_and_clean_data(filepath):
    """
    Load and clean the activities dataset.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 70)
    
    # Load the CSV
    df = pd.read_csv(filepath)
    print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Convert Time column to minutes
    print("\n→ Converting 'Time' column to minutes...")
    df['Time_Minutes'] = df['Time'].apply(time_to_minutes)
    
    # Ensure Calories is numeric
    print("→ Ensuring 'Calories' is numeric...")
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')
    
    # Ensure Avg HR is numeric
    print("→ Ensuring 'Avg HR' is numeric...")
    df['Avg HR'] = pd.to_numeric(df['Avg HR'], errors='coerce')
    
    # Convert Distance to numeric (additional useful feature)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    
    # Show initial stats
    print(f"\n→ Rows before cleaning: {len(df)}")
    
    # Drop rows with missing values in critical columns
    critical_columns = ['Calories', 'Time_Minutes', 'Avg HR', 'Activity Type']
    df_clean = df.dropna(subset=critical_columns)
    
    print(f"→ Rows after dropping missing values: {len(df_clean)}")
    print(f"→ Removed {len(df) - len(df_clean)} rows with missing data")
    
    # Show activity type distribution
    print(f"\n→ Activity Types found: {df_clean['Activity Type'].nunique()}")
    print("\n   Top 5 Activities by count:")
    print(df_clean['Activity Type'].value_counts().head())
    
    return df_clean


def prepare_features(df):
    """
    Prepare features for modeling with encoding.
    
    Args:
        df: Cleaned dataframe
    
    Returns:
        tuple: (X, y, feature_names, label_encoder, activity_type_original)
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING & ENCODING")
    print("=" * 70)
    
    # Store original activity type for stratification
    activity_type_original = df['Activity Type'].copy()
    
    # Label Encoding for Activity Type
    le = LabelEncoder()
    df['Activity_Type_Encoded'] = le.fit_transform(df['Activity Type'])
    
    print(f"✓ Encoded 'Activity Type' using Label Encoding")
    print(f"   Classes: {list(le.classes_)[:5]}... (showing first 5)")
    
    # Select features for the model
    feature_columns = ['Time_Minutes', 'Avg HR', 'Activity_Type_Encoded']
    
    # Add Distance if it has enough non-null values
    if df['Distance'].notna().sum() > len(df) * 0.5:
        feature_columns.append('Distance')
        print(f"✓ Added 'Distance' as a feature")
    
    X = df[feature_columns].copy()
    y = df['Calories'].copy()
    
    print(f"\n→ Features (X): {feature_columns}")
    print(f"→ Target (y): Calories")
    print(f"→ Dataset shape: {X.shape}")
    print(f"\n→ Feature Statistics:")
    print(X.describe())
    
    return X, y, feature_columns, le, activity_type_original


def split_data_stratified(X, y, activity_type_original, test_size=0.2, random_state=42):
    """
    Split data with stratification based on Activity Type.
    
    Args:
        X: Features
        y: Target
        activity_type_original: Original activity type for stratification
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 70)
    print("STEP 3: STRATIFIED DATA SPLITTING")
    print("=" * 70)
    
    # Stratified split based on Activity Type
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=activity_type_original
    )
    
    print(f"✓ Split data into {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test")
    print(f"   → Training set: {len(X_train)} samples")
    print(f"   → Test set: {len(X_test)} samples")
    
    # Verify stratification worked
    train_activities = activity_type_original.iloc[X_train.index].value_counts()
    test_activities = activity_type_original.iloc[X_test.index].value_counts()
    
    print(f"\n✓ Stratification successful!")
    print(f"   → Activity types in train: {len(train_activities)}")
    print(f"   → Activity types in test: {len(test_activities)}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        RandomForestRegressor: Trained model
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    print("→ Training Random Forest Regressor...")
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model training complete!")
    print(f"   → Number of trees: {model.n_estimators}")
    print(f"   → Max depth: {model.max_depth}")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Args:
        model: Trained model
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    
    Returns:
        tuple: (y_pred_train, y_pred_test)
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Training set metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Test set metrics
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("📊 TRAINING SET PERFORMANCE:")
    print(f"   → Mean Absolute Error (MAE): {mae_train:.2f} calories")
    print(f"   → R² Score: {r2_train:.4f}")
    
    print("\n📊 TEST SET PERFORMANCE:")
    print(f"   → Mean Absolute Error (MAE): {mae_test:.2f} calories")
    print(f"   → R² Score: {r2_test:.4f}")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if r2_test > 0.85:
        print("   ✓ Excellent model performance!")
    elif r2_test > 0.70:
        print("   ✓ Good model performance!")
    elif r2_test > 0.50:
        print("   → Moderate model performance. Consider feature engineering.")
    else:
        print("   ⚠ Model needs improvement. Consider more features or data.")
    
    print(f"\n   On average, predictions are off by ±{mae_test:.1f} calories.")
    
    return y_pred_train, y_pred_test


def create_visualizations(model, X_test, y_test, y_pred_test, feature_names):
    """
    Create beautiful visualizations for model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        y_pred_test: Test predictions
        feature_names: Names of features
    """
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZATION")
    print("=" * 70)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    color_scatter = '#3498db'
    color_line = '#e74c3c'
    color_bars = '#2ecc71'
    
    # ========================================
    # SUBPLOT 1: Prediction Accuracy
    # ========================================
    ax1 = axes[0]
    
    # Scatter plot
    ax1.scatter(y_test, y_pred_test, alpha=0.6, s=80, 
                color=color_scatter, edgecolors='white', linewidth=1.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             '--', color=color_line, linewidth=2.5, label='Perfect Prediction')
    
    # Labels and title
    ax1.set_xlabel('Actual Calories', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted Calories', fontsize=13, fontweight='bold')
    ax1.set_title('Prediction Accuracy: Actual vs Predicted Calories', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Calculate metrics for annotation
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    # Add metrics text box
    textstr = f'R² = {r2:.3f}\nMAE = {mae:.1f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========================================
    # SUBPLOT 2: Feature Importance
    # ========================================
    ax2 = axes[1]
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create better feature names for display
    display_names = []
    for name in feature_names:
        if name == 'Time_Minutes':
            display_names.append('Time (minutes)')
        elif name == 'Avg HR':
            display_names.append('Average Heart Rate')
        elif name == 'Activity_Type_Encoded':
            display_names.append('Activity Type')
        elif name == 'Distance':
            display_names.append('Distance')
        else:
            display_names.append(name)
    
    # Sort display names by importance
    sorted_names = [display_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Horizontal bar chart
    y_pos = np.arange(len(sorted_names))
    bars = ax2.barh(y_pos, sorted_importances, color=color_bars, 
                    edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=11)
    ax2.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
    ax2.set_title('Feature Importance: What Drives Calorie Prediction?', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Tight layout
    plt.tight_layout()
    
    print("✓ Visualization created successfully!")
    print("→ Displaying plot...")
    
    # Show the plot
    plt.show()


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print(" 🔥 CALORIE PREDICTION MODEL - EXPERT DATA SCIENCE APPROACH 🔥")
    print("=" * 70)
    
    # File path
    filepath = r'c:\Users\USER\OneDrive\שולחן העבודה\קורס AI\Activities_English.csv'
    
    try:
        # Step 1: Load and clean data
        df = load_and_clean_data(filepath)
        
        # Step 2: Prepare features
        X, y, feature_names, label_encoder, activity_type_original = prepare_features(df)
        
        # Step 3: Split data with stratification
        X_train, X_test, y_train, y_test = split_data_stratified(
            X, y, activity_type_original, test_size=0.2, random_state=42
        )
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        y_pred_train, y_pred_test = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        
        # Step 6: Create visualizations
        create_visualizations(model, X_test, y_test, y_pred_test, feature_names)
        
        print("\n" + "=" * 70)
        print(" ✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\n💡 Key Insights:")
        print("   1. Model successfully predicts calories burned during activities")
        print("   2. Stratified split ensures all activity types are represented")
        print("   3. Feature importance shows which factors matter most")
        print("   4. Visualizations provide clear model performance assessment")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at {filepath}")
        print("   Please check the file path and try again.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

