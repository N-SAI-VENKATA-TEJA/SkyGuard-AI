import pandas as pd

def evaluate_step7():
    print("==================================================")
    print("STEP 7: SENSOR HEALTH EVALUATION")
    print("==================================================")

    df = pd.read_csv('data/processed/step7_sensor_health.csv')
    
    print("\n1. Overall Data Quality Status")
    print(df['data_quality_status'].value_counts(normalize=True).map('{:.2%}'.format))
    
    print("\n2. Overall Sensor Status (Temperature)")
    print(df['temperature_status'].value_counts(normalize=True).map('{:.2%}'.format))
    print("\nOverall Sensor Status (Pressure)")
    print(df['pressure_status'].value_counts(normalize=True).map('{:.2%}'.format))
    print("\nOverall Sensor Status (Humidity)")
    print(df['humidity_status'].value_counts(normalize=True).map('{:.2%}'.format))
    
    print("\n3. Maintenance Recommendations")
    print(df['maintenance_status'].value_counts(normalize=True).map('{:.2%}'.format))

    print("\n4. Fault Type Classifications (Anomalous Rows Only)")
    anomalies = df[df['anomaly_flag'] == True]
    print(anomalies['classified_fault_type'].value_counts())
    
    print("\n5. Average Health during DATA_LOSS vs Good Data")
    dl = df[df['data_quality_status'] == 'DATA_LOSS']
    gd = df[df['data_quality_status'] == 'GOOD']
    print(f"Mean Temp Health (GOOD): {gd['temperature_health'].mean():.1f}")
    print(f"Mean Temp Health (DATA_LOSS): {dl['temperature_health'].mean():.1f}")
    
    print("\n6. Sensor Attribution (Anomalous Rows Only)")
    print(anomalies['affected_sensor'].value_counts())

if __name__ == '__main__':
    evaluate_step7()
