import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import
import matplotlib.pyplot as plt




def detect_ddos_attack(df, threshold_factor=2.0):
    """
    Detect DDoS attacks based on multiple indicators and return attack details.
    Returns: (is_ddos, attack_type, confidence, details)
    """
    results = {
        'is_ddos': False,
        'attack_type': 'Normal',
        'confidence': 0.0,
        'indicators': []
    }
    
    # Calculate attack indicators
    indicators = []
    
    # 1. Check for high packet rate (typical in volumetric attacks)
    if 'Packets_Per_Second' in df.columns:
        pkt_rate = df['Packets_Per_Second'].mean()
        if pkt_rate > 1000:  # Threshold for high packet rate
            confidence = min(0.8, (pkt_rate - 1000) / 10000)
            indicators.append({
                'name': 'High Packet Rate',
                'value': f"{pkt_rate:.2f} pkt/s",
                'confidence': confidence
            })
    
    # 2. Check for SYN flood (common in DDoS)
    if 'SYN_Count' in df.columns and 'ACK_Count' in df.columns:
        syn_ack_ratio = (df['SYN_Count'].sum() + 1) / (df['ACK_Count'].sum() + 1)
        if syn_ack_ratio > 3.0:  # High SYN to ACK ratio
            confidence = min(0.9, (syn_ack_ratio - 3) / 10)
            indicators.append({
                'name': 'High SYN/ACK Ratio',
                'value': f"{syn_ack_ratio:.2f}",
                'confidence': confidence
            })
    
    # 3. Check for unique IPs (reflection/amplification attack)
    if 'Unique_IPs_Per_Second' in df.columns:
        unique_ips = df['Unique_IPs_Per_Second'].mean()
        if unique_ips > 100:  # High number of unique IPs
            confidence = min(0.8, (unique_ips - 100) / 1000)
            indicators.append({
                'name': 'High Unique IPs',
                'value': f"{unique_ips:.0f} IPs/s",
                'confidence': confidence
            })
    
    # 4. Check for small packet size (common in UDP flood)
    if 'Packet_Size' in df.columns:
        avg_pkt_size = df['Packet_Size'].mean()
        if avg_pkt_size < 100:  # Very small packets
            confidence = 0.7
            indicators.append({
                'name': 'Small Packet Size',
                'value': f"{avg_pkt_size:.0f} bytes",
                'confidence': confidence
            })
    
    # Calculate overall confidence and determine attack type
    if indicators:
        results['is_ddos'] = True
        results['confidence'] = max(ind['confidence'] for ind in indicators)
        results['indicators'] = indicators
        
        # Determine most likely attack type based on indicators
        if any('High SYN/ACK' in ind['name'] for ind in indicators):
            results['attack_type'] = 'SYN Flood'
        elif any('Small Packet' in ind['name'] for ind in indicators):
            results['attack_type'] = 'UDP/ICMP Flood'
        elif any('High Unique IPs' in ind['name'] for ind in indicators):
            results['attack_type'] = 'Reflection/Amplification'
        else:
            results['attack_type'] = 'Generic DDoS'
    
    return results

def detect_anomalies(test_path, model_dir="model", results_dir="static/results", threshold_factor=2.0):
    """
    Detect anomalies and DDoS attacks using the trained Autoencoder model.
    Saves CSV results, plots, and detailed attack analysis.
    """
    # --- Ensure model and scaler exist ---
    model_path = os.path.join(model_dir, "autoencoder.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained model or scaler not found. Please train first.")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    # --- Load and preprocess test dataset ---
    df = pd.read_csv(test_path)
    
    # Ensure we have required columns for DDoS detection
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Select only numeric columns for the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_test = df[numeric_cols].values
    
    # Handle empty or invalid data
    if len(X_test) == 0:
        raise ValueError("No numeric data found in the input file.")
    
    try:
        X_scaled = scaler.transform(X_test)
    except ValueError as e:
        raise ValueError(f"Data scaling failed. Make sure the input data matches the training format. Error: {str(e)}")

    # --- Predict reconstruction and calculate errors ---
    reconstructed = model.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    print("\n=== DEBUG: Reconstruction Error Stats ===")
    print(f"Min error: {np.min(reconstruction_error):.6f}")
    print(f"Max error: {np.max(reconstruction_error):.6f}")
    print(f"Mean error: {np.mean(reconstruction_error):.6f}")
    # --- Calculate anomaly threshold ---
    threshold = np.mean(reconstruction_error) + 1.0 * np.std(reconstruction_error)  # 95th percentile as threshold
    print("\n=== DEBUG: Threshold Info ===")
    print(f"Threshold (95th percentile): {threshold:.6f}")
    print(f"Errors above threshold: {np.sum(reconstruction_error > threshold)} / {len(reconstruction_error)}")
    # --- Label anomalies ---
    df["Reconstruction_Error"] = reconstruction_error
    df["Anomaly_Score"] = reconstruction_error / threshold  # Normalized score
    df["Is_Anomaly"] = df["Anomaly_Score"] > 1.0

    # --- DDoS Attack Detection ---
    ddos_results = detect_ddos_attack(df)
    print("\n=== DEBUG: DDoS Detection Results ===")
    print(f"Is DDoS: {ddos_results['is_ddos']}")
    print(f"Attack Type: {ddos_results['attack_type']}")
    print(f"Confidence: {ddos_results['confidence']:.2f}")
    print(f"Indicators Found: {[i['name'] for i in ddos_results['indicators']]}")
    # Add DDoS detection results to the dataframe
    df["Attack_Type"] = ddos_results['attack_type'] if ddos_results['is_ddos'] else 'Normal'
    df["Attack_Confidence"] = ddos_results['confidence'] if ddos_results['is_ddos'] else 0.0
    
    # --- Save detailed detection results ---
    os.makedirs(results_dir, exist_ok=True)
    
    # Save full results
    result_path = os.path.join(results_dir, "detection_results.csv")
    df.to_csv(result_path, index=False)
    
    # Save summary of detected attacks
    summary_path = os.path.join(results_dir, "attack_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== DDoS Attack Detection Summary ===\n\n")
            if ddos_results['is_ddos']:
                f.write(f"[!] DETECTED: {ddos_results['attack_type']} (Confidence: {ddos_results['confidence']*100:.1f}%)\n\n")
                f.write("Attack Indicators:\n")
                for indicator in ddos_results['indicators']:
                    f.write(f"- {indicator['name']}: {indicator['value']} (Confidence: {indicator['confidence']*100:.1f}%)\n")
            else:
                f.write("[+] No DDoS attack detected. Network traffic appears normal.\n")
    except Exception as e:
        # Fallback to ASCII-only output if UTF-8 fails
        with open(summary_path, 'w') as f:
            f.write("=== DDoS Attack Detection Summary ===\n\n")
            if ddos_results['is_ddos']:
                f.write(f"[!] DETECTED: {ddos_results['attack_type']} (Confidence: {ddos_results['confidence']*100:.1f}%)\n\n")
                f.write("Attack Indicators:\n")
                for indicator in ddos_results['indicators']:
                    f.write(f"- {indicator['name']}: {indicator['value']} (Confidence: {indicator['confidence']*100:.1f}%)\n")
            else:
                f.write("[+] No DDoS attack detected. Network traffic appears normal.\n")
   
    # --- Generate visualizations ---
    # Create and save histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7, color='blue')
    plt.axvline(threshold, color='red', linestyle='--', 
               label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save histogram plot with a consistent filename
    hist_filename = 'error_hist.png'
    hist_path = os.path.join(results_dir, hist_filename)
    
    # Remove existing file if it exists
    if os.path.exists(hist_path):
        try:
            os.remove(hist_path)
        except OSError:
            pass
            
    # Save the new plot
    plt.savefig(hist_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Return the relative path for web access
    hist_web_path = os.path.join('results', hist_filename)
    
    # Create and save time series plot if timestamp is available
    time_plot_path = None
    if 'Timestamp' in df.columns:
        plt.figure(figsize=(12, 4))
        plt.scatter(df['Timestamp'], df['Anomaly_Score'], 
                   c=df['Is_Anomaly'].map({True: 'red', False: 'blue'}), 
                   alpha=0.6, s=10)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Anomaly Threshold')
        plt.title('Anomaly Scores Over Time')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        time_plot_path = os.path.join(results_dir, 'time_series.png')
        plt.savefig(time_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    # Generate time series plot for key metrics if timestamp is available
    if 'Timestamp' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Packets per second
        plt.subplot(3, 1, 1)
        if 'Packets_Per_Second' in df.columns:
            plt.plot(df['Timestamp'], df['Packets_Per_Second'], 'b-', alpha=0.7)
            plt.title('Packets Per Second')
            plt.ylabel('Packets/s')
        
        # Plot 2: Unique IPs per second
        plt.subplot(3, 1, 2)
        if 'Unique_IPs_Per_Second' in df.columns:
            plt.plot(df['Timestamp'], df['Unique_IPs_Per_Second'], 'g-', alpha=0.7)
            plt.title('Unique Source IPs Per Second')
            plt.ylabel('IPs/s')
        
        # Plot 3: Anomaly scores
        plt.subplot(3, 1, 3)
        plt.scatter(df['Timestamp'], df['Anomaly_Score'], 
                   c=df['Is_Anomaly'].map({True: 'red', False: 'blue'}), 
                   alpha=0.6, s=10)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        plt.title('Anomaly Detection Results')
        plt.ylabel('Anomaly Score')
        
        plt.tight_layout()
        ts_plot_path = os.path.join(results_dir, "time_series_analysis.png")
        plt.savefig(ts_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    # Calculate statistics
    n_anomalies = sum(df["Is_Anomaly"])
    total = len(df)
    anomaly_percent = (n_anomalies / total) * 100 if total > 0 else 0
    
    # Prepare return dictionary with web-accessible paths
    result = {
        'results_csv': result_path,
        'error_plot': hist_web_path,  # Web-accessible path to the histogram
        'attack_summary': summary_path,
        'n_anomalies': n_anomalies,
        'total_samples': total,
        'anomaly_percent': anomaly_percent,
        'threshold': threshold,
        'is_ddos': ddos_results['is_ddos'],
        'attack_type': ddos_results['attack_type'],
        'confidence': ddos_results['confidence'],
        'indicators': ddos_results['indicators']
    }
    
    # Add time series plot path if available
    if time_plot_path is not None:
        result['time_series_plot'] = time_plot_path
    
    return result
