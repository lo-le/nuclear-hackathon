import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('avalon_nuclear.csv')

# Create panic_mode variable
df['panic_mode'] = ((df['avalon_evac_recommendation'] == 1) |
                    (df['avalon_shutdown_recommendation'] == 1)) & \
                   (df['true_risk_level'] <= 2)
df['panic_mode'] = df['panic_mode'].astype(int)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Title
fig.suptitle('Dataset Overview: Avalon Nuclear Safety Monitoring System',
             fontsize=20, fontweight='bold', y=0.98)

# 1. Dataset Info Box (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
info_text = f"""
DATASET SUMMARY
━━━━━━━━━━━━━━━━━
Records: {len(df):,}
Features: {len(df.columns)}
Countries: {df['country'].nunique()}
Time Span: {df['year'].min()}–{df['year'].max()}
Missing Values: 0
"""
ax1.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Panic Mode Distribution (top center-left)
ax2 = fig.add_subplot(gs[0, 1])
panic_counts = df['panic_mode'].value_counts()
colors = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax2.pie(panic_counts, labels=['Normal', 'Panic Mode'],
                                     autopct='%1.1f%%', colors=colors, startangle=90,
                                     textprops={'fontsize': 11, 'weight': 'bold'})
ax2.set_title('Panic Mode Distribution\n(n=180 panic cases)', fontsize=11, weight='bold')

# 3. True Risk Level Distribution (top center-right)
ax3 = fig.add_subplot(gs[0, 2])
risk_counts = df['true_risk_level'].value_counts().sort_index()
bars = ax3.bar(risk_counts.index, risk_counts.values, color=['#3498db', '#f39c12', '#e74c3c'])
ax3.set_xlabel('True Risk Level', fontsize=10, weight='bold')
ax3.set_ylabel('Count', fontsize=10, weight='bold')
ax3.set_title('True Risk Level Distribution', fontsize=11, weight='bold')
ax3.set_xticks([0, 1, 2, 3])
for i, (idx, val) in enumerate(risk_counts.items()):
    ax3.text(idx, val + 50, str(val), ha='center', fontsize=9, weight='bold')

# 4. Incident Occurrence (top right)
ax4 = fig.add_subplot(gs[0, 3])
incident_counts = df['incident_occurred'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax4.bar(['No Incident', 'Incident'], incident_counts.values, color=colors)
ax4.set_ylabel('Count', fontsize=10, weight='bold')
ax4.set_title('Actual Incidents Occurred', fontsize=11, weight='bold')
for i, v in enumerate(incident_counts.values):
    ax4.text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9, weight='bold')

# 5. External Pressure Variables (middle row - full width)
ax5 = fig.add_subplot(gs[1, :])
external_vars = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']
data_to_plot = [df[var].values for var in external_vars]
bp = ax5.boxplot(data_to_plot, labels=['Public Anxiety', 'Social Media Rumors', 'Regulatory Scrutiny'],
                  patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#9b59b6', '#e67e22']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Score', fontsize=11, weight='bold')
ax5.set_title('External Pressure Variables Distribution (Key Tipping Point Signals)',
              fontsize=12, weight='bold')
ax5.grid(True, alpha=0.3)

# 6. Temporal Distribution (bottom left)
ax6 = fig.add_subplot(gs[2, 0:2])
year_counts = df['year'].value_counts().sort_index()
ax6.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4)
ax6.fill_between(year_counts.index, year_counts.values, alpha=0.3)
ax6.set_xlabel('Year', fontsize=10, weight='bold')
ax6.set_ylabel('Number of Records', fontsize=10, weight='bold')
ax6.set_title('Temporal Distribution of Observations', fontsize=11, weight='bold')
ax6.grid(True, alpha=0.3)

# 7. Top Countries (bottom center-right)
ax7 = fig.add_subplot(gs[2, 2:])
top_countries = df['country'].value_counts().head(10)
bars = ax7.barh(range(len(top_countries)), top_countries.values)
ax7.set_yticks(range(len(top_countries)))
ax7.set_yticklabels(top_countries.index, fontsize=9)
ax7.set_xlabel('Number of Records', fontsize=10, weight='bold')
ax7.set_title('Top 10 Countries by Observation Count', fontsize=11, weight='bold')
ax7.invert_yaxis()
for i, v in enumerate(top_countries.values):
    ax7.text(v + 5, i, str(v), va='center', fontsize=8)

# Add footer note
fig.text(0.5, 0.01, 'panic_mode = 1 when Avalon recommends evacuation/shutdown despite true_risk_level ≤ 2',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('slide3_data_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Visualization saved as 'slide3_data_overview.png'")
print(f"\nDataset Statistics:")
print(f"  Total records: {len(df):,}")
print(f"  Panic mode cases: {df['panic_mode'].sum()} ({df['panic_mode'].sum()/len(df)*100:.2f}%)")
print(f"  Normal cases: {len(df) - df['panic_mode'].sum()} ({(len(df)-df['panic_mode'].sum())/len(df)*100:.2f}%)")
print(f"  Unique countries: {df['country'].nunique()}")
print(f"  Reactor types: {df['reactor_type_code'].nunique()}")
