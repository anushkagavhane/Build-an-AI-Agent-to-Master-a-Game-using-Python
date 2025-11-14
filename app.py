import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque

# Set up the page
st.set_page_config(page_title="AI Game Agent", page_icon="🤖", layout="wide")

st.title("🎮 AI Game Agent - Watch It Learn and Plan Paths!")
st.write("See the AI robot learn to navigate the maze and show its path to the goal!")

# Improved Game Environment
class SmartGame:
    def __init__(self):
        self.size = 5
        self.reset()
    
    def reset(self):
        self.player = [0, 0]  # Start position
        self.goal = [4, 4]    # Goal position
        self.obstacles = [[1, 1], [2, 3], [3, 1], [1, 3]]  # More walls
        self.traps = [[2, 1], [3, 3]]  # Traps that end game
        self.rewards = [[0, 3], [2, 0], [4, 2]]  # Bonus points
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.won = False
        self.path = []  # Store the path taken
        return self.player
    
    def move(self, action):
        if self.game_over:
            return True
            
        self.steps += 1
        new_pos = self.player.copy()
        
        # Move based on action (0=Up, 1=Right, 2=Down, 3=Left)
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.size-1, new_pos[1] + 1)
        elif action == 2:  # Down
            new_pos[0] = min(self.size-1, new_pos[0] + 1)
        elif action == 3:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Record the move in path
        self.path.append(new_pos.copy())
        
        # Check if move is valid
        if new_pos in self.obstacles:
            self.score -= 5  # Penalty for hitting wall
            return False
        else:
            self.player = new_pos
            self.score -= 1  # Small penalty for each move
            
        # Check traps
        if self.player in self.traps:
            self.score -= 20
            self.game_over = True
            return True
            
        # Check rewards
        if self.player in self.rewards:
            self.score += 10
            self.rewards.remove(self.player)  # Remove collected reward
            
        # Check if reached goal
        if self.player == self.goal:
            self.score += 50  # Big reward for winning!
            self.game_over = True
            self.won = True
            return True
            
        # Check if too many steps
        if self.steps >= 30:
            self.game_over = True
            return True
            
        return False
    
    def get_optimal_path(self, q_table):
        """Calculate the optimal path using the Q-table"""
        path = []
        current_pos = self.player.copy()
        visited = set()
        
        for _ in range(20):  # Limit steps to avoid infinite loops
            if tuple(current_pos) in visited:
                break
            visited.add(tuple(current_pos))
            path.append(current_pos.copy())
            
            # Check if reached goal
            if current_pos == self.goal:
                break
                
            # Get best action from Q-table
            action = np.argmax(q_table[current_pos[0], current_pos[1]])
            
            # Move to next position
            if action == 0:  # Up
                current_pos[0] = max(0, current_pos[0] - 1)
            elif action == 1:  # Right
                current_pos[1] = min(self.size-1, current_pos[1] + 1)
            elif action == 2:  # Down
                current_pos[0] = min(self.size-1, current_pos[0] + 1)
            elif action == 3:  # Left
                current_pos[1] = max(0, current_pos[1] - 1)
                
            # Stop if hitting obstacle
            if current_pos in self.obstacles:
                break
                
        return path

# Q-Learning AI Agent
class QLearningAI:
    def __init__(self, grid_size=5, num_actions=4):
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.q_table = np.zeros((grid_size, grid_size, num_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        self.training_history = []
        
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state[0], state[1], action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state[0], next_state[1]])
        
        # Update Q-value
        self.q_table[state[0], state[1], action] += self.learning_rate * (target - current_q)
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
    
    def get_action_values(self, state):
        """Get Q-values for all actions at current state"""
        return self.q_table[state[0], state[1]]
    
    def get_policy(self):
        """Get the optimal policy from Q-table"""
        policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                policy[i, j] = np.argmax(self.q_table[i, j])
        return policy

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = SmartGame()
    st.session_state.ai = QLearningAI()
    st.session_state.training = False
    st.session_state.training_complete = False
    st.session_state.results = []
    st.session_state.demo_mode = False
    st.session_state.show_path = False
    st.session_state.show_q_values = False

# Sidebar
st.sidebar.header("🎯 Game Controls")

if st.sidebar.button("🔄 New Game"):
    st.session_state.game.reset()
    st.session_state.demo_mode = False
    st.rerun()

# Manual controls
st.sidebar.subheader("Manual Play")
col1, col2, col3, col4 = st.sidebar.columns(4)
with col1:
    if st.button("⬆️ Up"):
        if not st.session_state.game.game_over:
            st.session_state.game.move(0)
            st.rerun()
with col2:
    if st.button("➡️ Right"):
        if not st.session_state.game.game_over:
            st.session_state.game.move(1)
            st.rerun()
with col3:
    if st.button("⬇️ Down"):
        if not st.session_state.game.game_over:
            st.session_state.game.move(2)
            st.rerun()
with col4:
    if st.button("⬅️ Left"):
        if not st.session_state.game.game_over:
            st.session_state.game.move(3)
            st.rerun()

# AI Controls
st.sidebar.subheader("🤖 AI Controls")

if st.sidebar.button("🚀 Train AI (100 episodes)"):
    st.session_state.training = True
    st.session_state.results = []

# Toggle options
col_opt1, col_opt2 = st.sidebar.columns(2)
with col_opt1:
    show_path = st.checkbox("Show AI Path", value=st.session_state.show_path)
with col_opt2:
    show_q_values = st.checkbox("Show Q-Values", value=st.session_state.show_q_values)

if show_path != st.session_state.show_path:
    st.session_state.show_path = show_path
    st.rerun()

if show_q_values != st.session_state.show_q_values:
    st.session_state.show_q_values = show_q_values
    st.rerun()

# Main display
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎮 Game Board")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the game board
    for i in range(5):
        for j in range(5):
            cell_color = 'white'
            # Add subtle grid
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, 
                               facecolor=cell_color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add elements
            if [i, j] == st.session_state.game.player:
                ax.text(j, i, '🤖', ha='center', va='center', fontsize=30, weight='bold')
            elif [i, j] == st.session_state.game.goal:
                ax.text(j, i, '🎯', ha='center', va='center', fontsize=30, weight='bold')
            elif [i, j] in st.session_state.game.obstacles:
                ax.text(j, i, '🚫', ha='center', va='center', fontsize=22)
            elif [i, j] in st.session_state.game.traps:
                ax.text(j, i, '💀', ha='center', va='center', fontsize=22)
            elif [i, j] in st.session_state.game.rewards:
                ax.text(j, i, '💰', ha='center', va='center', fontsize=22)
    
    # Show AI's planned path if trained and option enabled
    if st.session_state.training_complete and st.session_state.show_path:
        optimal_path = st.session_state.game.get_optimal_path(st.session_state.ai.q_table)
        
        # Draw path lines
        for k in range(len(optimal_path) - 1):
            start = optimal_path[k]
            end = optimal_path[k + 1]
            ax.plot([start[1], end[1]], [start[0], end[0]], 'g-', linewidth=3, alpha=0.6)
            ax.plot([start[1], end[1]], [start[0], end[0]], 'go-', markersize=8, alpha=0.4)
    
    # Show actual path taken
    if len(st.session_state.game.path) > 1:
        path_x = [pos[1] for pos in st.session_state.game.path]
        path_y = [pos[0] for pos in st.session_state.game.path]
        ax.plot(path_x, path_y, 'b--', alpha=0.4, linewidth=2)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # So 0,0 is top-left
    ax.set_xticks([])
    ax.set_yticks([])
    
    title = "Maze - AI Path Planning" if st.session_state.show_path else "Maze - Guide 🤖 to 🎯"
    ax.set_title(title, fontsize=16, weight='bold')
    
    # Add legend
    legend_elements = []
    if st.session_state.show_path:
        legend_elements.extend([
            plt.Line2D([0], [0], color='green', linewidth=3, label='AI Planned Path'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Actual Path')
        ])
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    st.pyplot(fig)

with col_right:
    st.subheader("📊 Game Info")
    
    # Game status
    if st.session_state.game.game_over:
        if st.session_state.game.won:
            st.success("🏆 **VICTORY!** AI reached the goal!")
        else:
            st.error("💥 **GAME OVER** - AI failed!")
    
    st.metric("Score", st.session_state.game.score)
    st.metric("Steps", st.session_state.game.steps)
    st.metric("Player Position", f"{st.session_state.game.player}")
    st.metric("Remaining Rewards", len(st.session_state.game.rewards))
    
    # Game state
    if st.session_state.game.game_over:
        status = "❌ Game Over" if not st.session_state.game.won else "✅ Victory!"
    else:
        status = "🟢 Playing..."
    st.metric("Game Status", status)
    
    # Show AI's current decision if trained
    if st.session_state.training_complete and not st.session_state.game.game_over:
        st.subheader("🤖 AI Decision Making")
        
        current_state = st.session_state.game.player
        q_values = st.session_state.ai.get_action_values(current_state)
        best_action = np.argmax(q_values)
        
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        action_emojis = ["⬆️", "➡️", "⬇️", "⬅️"]
        
        st.write("**Q-values for current position:**")
        for i, (action, emoji) in enumerate(zip(action_names, action_emojis)):
            highlight = "**" if i == best_action else ""
            st.write(f"{emoji} {action}: {q_values[i]:.2f} {highlight}")

# Training Section
if st.session_state.training and not st.session_state.training_complete:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Train the AI
    wins = 0
    for episode in range(100):
        state = st.session_state.game.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 30:
            action = st.session_state.ai.choose_action(state)
            next_state = st.session_state.game.player.copy()
            old_score = st.session_state.game.score
            
            done = st.session_state.game.move(action)
            reward = st.session_state.game.score - old_score
            
            st.session_state.ai.update(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            
            if done and st.session_state.game.won:
                wins += 1
        
        # Record results
        st.session_state.results.append({
            'episode': episode + 1,
            'score': st.session_state.game.score,
            'steps': steps,
            'won': st.session_state.game.won,
            'exploration_rate': st.session_state.ai.exploration_rate
        })
        
        # Update progress
        progress = (episode + 1) / 100
        progress_bar.progress(progress)
        status_text.text(f"Training... {episode + 1}/100\nWins: {wins}")
    
    st.session_state.training = False
    st.session_state.training_complete = True
    st.sidebar.success(f"✅ Training complete! {wins}/100 wins")
    st.rerun()

# Watch AI Play
st.sidebar.subheader("🎮 Watch AI Play")

if st.sidebar.button("▶️ Watch Trained AI Play"):
    if st.session_state.training_complete:
        st.session_state.demo_mode = True
        st.session_state.game.reset()
        st.session_state.show_path = True  # Auto-show path during demo
    else:
        st.sidebar.warning("Please train the AI first!")

# Demo mode - Watch AI play
if st.session_state.demo_mode and not st.session_state.game.game_over:
    demo_placeholder = st.empty()
    
    with demo_placeholder.container():
        st.info("🤖 AI is playing...")
        
        # AI makes a move
        state = st.session_state.game.player
        action = np.argmax(st.session_state.ai.q_table[state[0], state[1]])
        
        # Show AI's reasoning
        q_values = st.session_state.ai.get_action_values(state)
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        action_emojis = ["⬆️", "➡️", "⬇️", "⬅️"]
        
        st.write("**AI's Decision Process:**")
        for i, (action_name, emoji) in enumerate(zip(action_names, action_emojis)):
            highlight = "🏆" if i == action else ""
            st.write(f"{emoji} {action_name}: {q_values[i]:.2f} {highlight}")
        
        st.session_state.game.move(action)
        
        # Auto-refresh for next move
        time.sleep(1.5)
        st.rerun()
    
    if st.session_state.game.game_over:
        if st.session_state.game.won:
            st.balloons()
            st.success("🎉 AI successfully completed the maze!")
        else:
            st.error("❌ AI failed to reach the goal")

# Show Q-values visualization
if st.session_state.training_complete and st.session_state.show_q_values:
    st.subheader("🧠 AI's Learned Q-Values")
    
    # Create a visualization of the Q-table
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for action in range(4):
        row = action // 2
        col = action % 2
        q_values = st.session_state.ai.q_table[:, :, action]
        
        im = axes[row, col].imshow(q_values, cmap='RdYlGn', aspect='equal')
        axes[row, col].set_title(f'{action_names[action]} Action Q-values')
        axes[row, col].set_xticks(range(5))
        axes[row, col].set_yticks(range(5))
        
        # Add values to cells
        for i in range(5):
            for j in range(5):
                text = axes[row, col].text(j, i, f'{q_values[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

# Show training results
if st.session_state.results:
    st.subheader("📈 Training Results")
    
    # Calculate statistics
    total_wins = sum(1 for r in st.session_state.results if r['won'])
    avg_score = np.mean([r['score'] for r in st.session_state.results])
    avg_steps = np.mean([r['steps'] for r in st.session_state.results])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wins", f"{total_wins}/100")
    with col2:
        st.metric("Win Rate", f"{(total_wins/100)*100:.1f}%")
    with col3:
        st.metric("Average Score", f"{avg_score:.1f}")
    with col4:
        st.metric("Average Steps", f"{avg_steps:.1f}")
    
    # Plot learning progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scores over time
    episodes = [r['episode'] for r in st.session_state.results]
    scores = [r['score'] for r in st.session_state.results]
    ax1.plot(episodes, scores, alpha=0.7)
    ax1.set_title("Scores per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    
    # Exploration rate
    exploration_rates = [r['exploration_rate'] for r in st.session_state.results]
    ax2.plot(episodes, exploration_rates, color='orange', alpha=0.7)
    ax2.set_title("Exploration Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Exploration Rate")
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Final AI status
if st.session_state.training_complete:
    st.sidebar.info(f"🤖 AI Ready! Exploration: {st.session_state.ai.exploration_rate:.3f}")

# How to interpret the path
if st.session_state.show_path:
    st.info("""
    **🎯 Path Interpretation:**
    - **Green line**: AI's planned optimal path to goal
    - **Blue dashed line**: Actual path taken by AI
    - When they match: AI is following its optimal plan!
    - When they differ: AI is exploring or adapting to obstacles
    """)


