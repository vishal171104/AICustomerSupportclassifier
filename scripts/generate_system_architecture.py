import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyBboxPatch, Rectangle, ArrowStyle
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Function to draw a block
def draw_block(ax, center_x, center_y, width, height, text, color, edgecolor='black', text_color='black'):
    # x,y is bottom left
    x = center_x - width/2
    y = center_y - height/2
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor=edgecolor, lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(center_x, center_y, text, ha='center', va='center', color=text_color, fontweight='bold', fontsize=11, zorder=3)
    return x, y, width, height

# Set up coordinates
layer_labels = ["Client Layer", "Application Layer (FastAPI)", "Machine Learning & Data Layer"]

# Draw Layer Boundaries
ax.add_patch(Rectangle((0.02, 0.05), 0.25, 0.85, fill=False, edgecolor='gray', linestyle='--', lw=2, zorder=1))
ax.text(0.145, 0.92, "Client Layer", ha='center', fontweight='bold', fontsize=14, color='gray')

ax.add_patch(Rectangle((0.30, 0.05), 0.35, 0.85, fill=False, edgecolor='teal', linestyle='--', lw=2, zorder=1))
ax.text(0.475, 0.92, "Application Layer / Gateway", ha='center', fontweight='bold', fontsize=14, color='teal')

ax.add_patch(Rectangle((0.68, 0.05), 0.30, 0.85, fill=False, edgecolor='purple', linestyle='--', lw=2, zorder=1))
ax.text(0.83, 0.92, "Data & ML Core", ha='center', fontweight='bold', fontsize=14, color='purple')

# Draw Blocks
# Client Layer
draw_block(ax, 0.145, 0.75, 0.2, 0.12, "Chrome Extension\n(Gmail Sync UI)", "lightblue")
draw_block(ax, 0.145, 0.45, 0.2, 0.12, "Dashboard UI\n(Agent Terminal)", "lightblue")

# App Layer
draw_block(ax, 0.475, 0.75, 0.25, 0.12, "REST API Gateway\n(FastAPI Server)", "lightgreen")
draw_block(ax, 0.475, 0.45, 0.25, 0.12, "Live Inference Engine\n(Prediction & Confidence)", "mediumaquamarine")
draw_block(ax, 0.475, 0.15, 0.25, 0.12, "Background Daemon\n(Drift Detection Math)", "darkseagreen")

# ML Layer
draw_block(ax, 0.83, 0.75, 0.2, 0.12, "SQLite Database\n(Tickets & Corrections)", "wheat")
draw_block(ax, 0.83, 0.45, 0.2, 0.15, "Model Pipelines\n- TF-IDF Vectorizer\n- SVM/LogReg Ensemble", "plum")
draw_block(ax, 0.83, 0.15, 0.2, 0.12, "Offline Retraining Script\n(HITL Injection)", "thistle")

# Arrows helper
def draw_arrow(ax, start, end, label=""):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='black', lw=2), zorder=1)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.02
        ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred', zorder=4)

def draw_bi_arrow(ax, start, end, label=""):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->,head_width=0.4,head_length=0.6", color='black', lw=2), zorder=1)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.02
        ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred', zorder=4)

# Draw arrows
# Extension -> Gateway
draw_arrow(ax, (0.245, 0.75), (0.35, 0.75), "HTTP POST")

# Gateway -> Database
draw_bi_arrow(ax, (0.6, 0.75), (0.73, 0.75), "Read/Write Tickets")

# Gateway -> Inference Engine
draw_arrow(ax, (0.475, 0.69), (0.475, 0.51), "Pass Text")

# Dashboard -> Gateway (API)
draw_bi_arrow(ax, (0.245, 0.45), (0.35, 0.45), "Fetch/Correct")
draw_arrow(ax, (0.35, 0.48), (0.5, 0.69), "") # Dashboard hitting gateway for other things

# Inference Engine -> Model Pipelines
draw_bi_arrow(ax, (0.6, 0.45), (0.73, 0.45), "Load Pickle Binary")

# Inference Engine -> Drift Daemon
draw_arrow(ax, (0.475, 0.39), (0.475, 0.21), "Send Signal")

# Dashboard Feedback -> Offline Retrain -> Pickles
draw_arrow(ax, (0.93, 0.75), (1.0, 0.75), "") # Just bending the line visually 
ax.plot([0.93, 0.96, 0.96, 0.93], [0.75, 0.75, 0.15, 0.15], color='black', lw=2, zorder=1) # connect DB to Retraining
ax.annotate("", xy=(0.93, 0.15), xytext=(0.96, 0.15), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='black', lw=2), zorder=1)
ax.text(0.97, 0.45, "Cron/Trigger", rotation=90, va='center', fontweight='bold', color='darkred', zorder=4)

# Retraining -> Override Model Pipelines
draw_arrow(ax, (0.83, 0.21), (0.83, 0.375), "Overwrite .pkl")

plt.title("System Architecture: End-to-End Triage Workflow", pad=20, fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/system_architecture.png', dpi=300)
plt.close()

print("System architecture diagram generated successfully.")
