import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# ---------------------------------------------------------
# 1. Extension Architecture Diagram
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Extension Container Box
ax.add_patch(plt.Rectangle((0.1, 0.2), 0.45, 0.6, fill=False, edgecolor='black', linestyle='--', lw=2))
ax.text(0.325, 0.75, "Chrome Extension (\"TicketIQ Sync\")", ha='center', va='center', fontweight='bold', fontsize=11)

# Components
boxes = [
    ("popup.html / popup.js\n(User Interface)", (0.325, 0.6), "lightblue"),
    ("background.js\n(Service Worker for Polling)", (0.325, 0.35), "moccasin"),
    ("Gmail REST API\n(External)", (0.8, 0.6), "lightgreen"),
    ("FastAPI Backend\n(http://localhost:8000)", (0.8, 0.35), "plum")
]

for text, pos, color in boxes:
    ax.add_patch(plt.Rectangle((pos[0]-0.15, pos[1]-0.08), 0.3, 0.16, facecolor=color, edgecolor='black'))
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
# popup to background
plt.annotate("", xy=(0.325, 0.43), xytext=(0.325, 0.52), arrowprops=dict(arrowstyle="<->", lw=1.5))
# background to Gmail
plt.annotate("", xy=(0.65, 0.6), xytext=(0.475, 0.4), arrowprops=dict(arrowstyle="<->", lw=1.5))
# background to FastAPI
plt.annotate("", xy=(0.65, 0.35), xytext=(0.475, 0.35), arrowprops=dict(arrowstyle="->", lw=1.5))

plt.title("Chrome Extension Component Interaction Model", fontweight='bold', y=0.9)
plt.tight_layout()
plt.savefig('figures/extension_architecture.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. Docker Topology Diagram
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Host Machine box
ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False, edgecolor='black', lw=2))
ax.text(0.5, 0.85, "Host Machine", ha='center', va='center', fontweight='bold', fontsize=12)

# Bridge Network box
ax.add_patch(plt.Rectangle((0.1, 0.5), 0.8, 0.25, fill=False, edgecolor='blue', linestyle='--', lw=2))
ax.text(0.5, 0.7, "Docker Bridge Network (customersupportticketai_default)", ha='center', va='center', color='blue')

# Services
services = [
    ("API Service\n(Container: api-1)\nPort: 8000", (0.3, 0.58), "plum"),
    ("Dashboard UI\n(via FastAPI)", (0.7, 0.58), "lightblue")
]

for text, pos, color in services:
    ax.add_patch(plt.Rectangle((pos[0]-0.15, pos[1]-0.06), 0.3, 0.12, facecolor=color, edgecolor='black'))
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')

# Host Volumes
volumes = [
    ("./data\n(SQLite + CSV)", (0.3, 0.25), "lightgreen"),
    ("./model\n(Pickled Pipelines)", (0.7, 0.25), "moccasin")
]

for text, pos, color in volumes:
    # Use generic rectangle for mounted directories
    ax.add_patch(plt.Rectangle((pos[0]-0.15, pos[1]-0.06), 0.3, 0.12, fill=False, edgecolor=color, hatch='//', lw=2))
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')

# Links
# API to data
plt.annotate("Volume Mount", xy=(0.3, 0.31), xytext=(0.3, 0.52), arrowprops=dict(arrowstyle="<->", lw=1.5, ls='--'), ha='center')
# API to model
plt.annotate("Volume Mount", xy=(0.7, 0.31), xytext=(0.3, 0.52), arrowprops=dict(arrowstyle="<->", lw=1.5, ls='--'), ha='center')

plt.title("Docker Container Topology and Volumes", fontweight='bold', y=0.95)
plt.tight_layout()
plt.savefig('figures/docker_topology.png', dpi=300)
plt.close()

print("Architecture diagrams generated successfully!")
