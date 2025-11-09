# Step 1: Load & Prepare Data for Pitch Control
from kloppy import skillcorner
from scipy.special import expit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- Constants ---
MATCH_ID = '1925299'
TIME_WINDOW_SEC = 300  # First 5 minutes

# --- 1️⃣ Load Match Data ---
print(f"Loading match {MATCH_ID} from SkillCorner...")
dataset = skillcorner.load_open_data(
    match_id=MATCH_ID,
    coordinates="skillcorner"
)
print("Data loaded successfully.")

# --- 2️⃣ Convert to DataFrame ---
df = dataset.to_df()  # Wide-format DataFrame with player positions per timestamp
print(f"DataFrame shape: {df.shape}")

# --- 3️⃣ Filter by Time Window ---
df['timestamp_sec'] = df['timestamp'].dt.total_seconds()
df_window = df[df['timestamp_sec'] <= TIME_WINDOW_SEC].copy()
print(f"Filtered DataFrame shape (first {TIME_WINDOW_SEC} seconds): {df_window.shape}")

# --- 4️⃣ Optional: Check sample data ---
print(df_window.head())

# --- 5️⃣ Identify Players ---
# Get all player IDs from metadata
player_ids = [str(p.player_id) for team in dataset.metadata.teams for p in team.players]
print(f"Total players found: {len(player_ids)}")

# --- 5️⃣ Identify Players (Fixed) ---
# Only include players who have at least one _x column in df_window
player_ids_in_df = set()
for col in df_window.columns:
    if col.endswith('_x'):
        player_id = col.split('_')[0]
        player_ids_in_df.add(player_id)

player_ids_in_df = list(player_ids_in_df)
print(f"Players with actual data in this window: {len(player_ids_in_df)}")

# --- 6️⃣ Handle Missing Data (Fixed) ---
position_cols = [f"{pid}_x" for pid in player_ids_in_df] + [f"{pid}_y" for pid in player_ids_in_df]

# Now all columns exist in df_window, safe to drop NaNs
df_window = df_window.dropna(subset=position_cols, how='all')
print("Data cleaned: NaNs for off-pitch players dropped.")

# --- Step 2: Compute Player Velocities ---
# We'll calculate dx/dt, dy/dt, and speed for each player

# Time difference between frames in seconds
df_window['dt'] = df_window['timestamp_sec'].diff().fillna(0.0)

player_velocities = {}

for pid in player_ids_in_df:
    x_col = f"{pid}_x"
    y_col = f"{pid}_y"

    # Compute differences
    dx = df_window[x_col].diff().fillna(0.0)
    dy = df_window[y_col].diff().fillna(0.0)

    # Compute velocity components
    vx = dx / df_window['dt']
    vy = dy / df_window['dt']

    # Compute total speed
    speed = np.sqrt(vx**2 + vy**2)

    # Store in dict
    player_velocities[pid] = {
        'vx': vx.values,
        'vy': vy.values,
        'speed': speed.values
    }


# ----------------------------
# Step 3: Pitch Control - Code
# ----------------------------


# --- Parameters (tweakable) ---
GRID_RES = 2.0       # meters per cell (2m x 2m)
V_MAX = 7.0          # assumed max speed for players (m/s)
T_REACTION = 0.2     # reaction time (s)
LAMBDA = 0.1         # decay constant for exp(-lambda * t)
EPS = 1e-3           # small epsilon to avoid division by zero

# --- Helper: build player -> team map from metadata ---
def build_player_team_map(metadata):
    """
    Returns dict: {player_id_str: team_id}
    """
    mapping = {}
    for team in metadata.teams:
        for p in team.players:
            mapping[str(p.player_id)] = team.team_id
    return mapping

player_team_map = build_player_team_map(dataset.metadata)

def compute_pitch_control_for_frame(df_frame, player_team_map, velocities, lambda_=0.05):
    X, Y = np.meshgrid(np.linspace(-52.5, 52.5, 105), np.linspace(-34, 34, 68))
    teamA_control = np.zeros_like(X)
    teamB_control = np.zeros_like(X)

    # Identify teams
    teams = list({v for v in player_team_map.values()})
    if len(teams) < 2:
        return X, Y, np.zeros_like(X), df_frame
    teamA_id, teamB_id = teams[:2]

    gamma = 1.0
    t0 = 0.4
    min_speed = 0.5

    for player_id in player_team_map:
        col_x = f"{player_id}_x"
        col_y = f"{player_id}_y"
        if col_x not in df_frame.columns or col_y not in df_frame.columns:
            continue

        pos_x = df_frame[col_x].values[0]
        pos_y = df_frame[col_y].values[0]
        if np.isnan(pos_x) or np.isnan(pos_y):
            continue

        vx = velocities[player_id]['vx'][-1]
        vy = velocities[player_id]['vy'][-1]
        speed = max(np.sqrt(vx**2 + vy**2), min_speed)

        dist = np.sqrt((X - pos_x)**2 + (Y - pos_y)**2)
        reaction_time = 0.7 if player_team_map[player_id] == teamA_id else 0.8
        t_arrive = reaction_time + dist / speed
        influence = expit(-gamma * (t_arrive - t0))

        if player_team_map[player_id] == teamA_id:
            teamA_control += influence
        else:
            teamB_control += influence

    control_grid = teamA_control / (teamA_control + teamB_control + 1e-6)
    return X, Y, control_grid, df_frame


# --- Plotting helper ---
# def plot_pitch_control(X, Y, control_grid, df_frame, player_team_map, title="Pitch Control (Team A Probability)"):
#     plt.figure(figsize=(10, 7))
#     extent = [X.min(), X.max(), Y.min(), Y.max()]

#     # Pitch control heatmap
#     plt.imshow(control_grid, origin='lower', extent=extent, aspect='auto', cmap='coolwarm', vmin=0, vmax=1, alpha=0.8)
#     plt.colorbar(label='P(Team A controls)')
#     # Plot players
#     teamA_players = [pid for pid, t in player_team_map.items() if t == list(set(player_team_map.values()))[0]]
#     teamB_players = [pid for pid, t in player_team_map.items() if t == list(set(player_team_map.values()))[1]]

# # Plot Team A (Blue)
#     for pid in teamA_players:
#       if f"{pid}_x" in df_frame.columns and f"{pid}_y" in df_frame.columns:
#         plt.scatter(df_frame[f"{pid}_x"], df_frame[f"{pid}_y"], 
#                     color='blue', s=90, edgecolor='white', marker='o')
# # Add legend entry for Team A
#     plt.scatter([], [], color='blue', s=90, edgecolor='white', marker='o', label='Team A')

# # Plot Team B (Red)
#     for pid in teamB_players:
#        if f"{pid}_x" in df_frame.columns and f"{pid}_y" in df_frame.columns:
#         plt.scatter(df_frame[f"{pid}_x"], df_frame[f"{pid}_y"], 
#                     color='red', s=90, edgecolor='white', marker='o')
# # Add legend entry for Team B
#     plt.scatter([], [], color='red', s=90, edgecolor='white', marker='o', label='Team B')

# # Plot Ball (Black, Football Marker)
#     if 'ball_x' in df_frame.columns and 'ball_y' in df_frame.columns:
#         plt.scatter(df_frame['ball_x'], df_frame['ball_y'], 
#                 color='black', s=120, marker='o', label='Ball')

# # Add legend (top-right corner)
#     plt.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black', fontsize=10)


#     # # Plot players
#     # teamA_players = [pid for pid, t in player_team_map.items() if t == list(set(player_team_map.values()))[0]]
#     # teamB_players = [pid for pid, t in player_team_map.items() if t == list(set(player_team_map.values()))[1]]

#     # for pid in teamA_players:
#     #     if f"{pid}_x" in df_frame.columns and f"{pid}_y" in df_frame.columns:
#     #         plt.scatter(df_frame[f"{pid}_x"], df_frame[f"{pid}_y"], color='blue', s=90, edgecolor='white')
#     # for pid in teamB_players:
#     #     if f"
    
#     plt.title(title)
#     plt.xlabel('x (m)')
#     plt.ylabel('y (m)')
#     plt.legend(['Team A', 'Team B', 'Ball'], loc='upper right')
#     plt.tight_layout()
#     plt.show()

def plot_pitch_control(X, Y, control_grid, df_frame, player_team_map, title="Pitch Control (Team A Probability)"):
    plt.figure(figsize=(10, 7))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # Pitch control heatmap
    plt.imshow(control_grid, origin='lower', extent=extent, aspect='auto',
               cmap='coolwarm', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(label='P(Team A controls)')

    # Identify teams
    teamA_players = [pid for pid, t in player_team_map.items()
                     if t == list(set(player_team_map.values()))[0]]
    teamB_players = [pid for pid, t in player_team_map.items()
                     if t == list(set(player_team_map.values()))[1]]

    # --- Plot Team A (Blue) ---
    for pid in teamA_players:
        if f"{pid}_x" in df_frame.columns and f"{pid}_y" in df_frame.columns:
            plt.scatter(df_frame[f"{pid}_x"], df_frame[f"{pid}_y"],
                        color='blue', s=90, edgecolor='white', marker='o')
    plt.scatter([], [], color='blue', s=90, edgecolor='white', marker='o', label='Team A')

    # --- Plot Team B (Red) ---
    for pid in teamB_players:
        if f"{pid}_x" in df_frame.columns and f"{pid}_y" in df_frame.columns:
            plt.scatter(df_frame[f"{pid}_x"], df_frame[f"{pid}_y"],
                        color='red', s=90, edgecolor='white', marker='o')
    plt.scatter([], [], color='red', s=90, edgecolor='white', marker='o', label='Team B')

    # --- Plot Ball (Black) ---
    if 'ball_x' in df_frame.columns and 'ball_y' in df_frame.columns:
        plt.scatter(df_frame['ball_x'], df_frame['ball_y'],
                    color='black', s=120, marker='o', label='Ball')

    # --- Legend ---
    plt.legend(loc='upper right', facecolor='white', framealpha=1,
               edgecolor='black', fontsize=10)

    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.show()


frames_to_plot = [180, 200, 220]
for f in frames_to_plot:
    frame_df = df_window.iloc[[f]]
    X, Y, control_grid, frame_used = compute_pitch_control_for_frame(
        df_frame=frame_df,
        player_team_map=player_team_map,
        velocities=player_velocities,
        lambda_=LAMBDA
    )
    plot_pitch_control(X, Y, control_grid,frame_df,player_team_map ,title=f"Pitch Control - Frame {f}")