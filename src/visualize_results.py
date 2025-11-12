# src/visualize_results.py
import pandas as pd
import folium
import os

os.makedirs('results', exist_ok=True)
df = pd.read_csv('results/predictions_with_probs.csv')
center = [df['lat'].mean(), df['lon'].mean()]
m = folium.Map(location=center, zoom_start=8)
for _, r in df.iterrows():
    prob = r.get('pred_prob', 0)
    color = 'red' if prob>0.5 else 'blue'
    folium.CircleMarker([r['lat'], r['lon']], radius=4, color=color, fill=True,
                        popup=f"prob={prob:.2f}").add_to(m)
m.save('results/predicted_pollution_map.html')
print("Saved map to results/predicted_pollution_map.html")
