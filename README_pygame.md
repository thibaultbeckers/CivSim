# Civilization Simulation - Pygame Version

A high-performance, real-time visualization of the civilization simulation using pygame instead of matplotlib.

## 🚀 Features

- **Real-time grid visualization** with smooth 60 FPS rendering
- **Large grid focus** - 800x800 pixel main simulation window
- **Interactive controls**: Pause/Play, Stop buttons
- **Live charts**: Population, energy, strength distribution, and daily statistics
- **Comprehensive legend** explaining all visual elements
- **Keyboard shortcuts**: Space (pause/play), S (stop), Escape (quit)
- **Same simulation logic** as the original, just better visualization
- **Individual chart saving** - saves plots as PNG files instead of showing them all

## 📦 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python main_pygame.py
   ```

## 🎮 Controls

### **Mouse Controls**
- **Pause/Play Button**: Click to pause or resume simulation
- **Stop Button**: Click to stop simulation and generate final charts

### **Keyboard Shortcuts**
- **Spacebar**: Toggle pause/play
- **S key**: Stop simulation
- **Escape**: Quit simulation

## 🎨 Visual Elements

### **Grid World (Main Focus)**
- **Large 800x800 pixel grid** as the central visualization
- **Food Zones**: Different shades of green (low/medium/high richness)
- **Carrots**: Orange triangles (^) representing available food
- **Agents**: Colored circles with energy-based fill and action-based outlines
- **Grid Lines**: Dark gray lines (150,150,150) for excellent visibility

### **Agent Colors**
- **Fill Color**: Based on energy level (purple=low, blue=medium, yellow=high)
- **Outline Color**: Based on current action
  - Blue: Seeking food
  - Gray: Exploring
  - Green: Foraging
  - Red: In combat

### **Live Charts**
- **Population**: Blue line showing agent count over time
- **Energy**: Green line showing average energy over time
- **Strength Distribution**: Orange histogram showing current agent strength spread
- **Daily Statistics**: Purple chart showing fights, deaths, births, and carrots per day

### **Legend**
- Complete explanation of all visual elements
- Positioned to the right of the grid for easy reference

## 🔧 Technical Details

- **Resolution**: 1800x1200 pixels (optimized for grid focus)
- **Frame Rate**: 60 FPS for smooth animation
- **Grid Size**: 800x800 pixels (main focus of the interface)
- **Performance**: Much faster than matplotlib version
- **Memory**: Efficient sprite and surface management
- **Grid Lines**: Dark gray (150,150,150) for excellent visibility

## 📊 Final Analysis & Chart Saving

When the simulation stops (either manually or naturally), it automatically saves comprehensive analysis charts as PNG files in a `simulation_results/` directory:

### **Individual Charts Saved:**
- **`population_evolution.png`** - Population trends over time
- **`combat_frequency.png`** - Fight frequency patterns  
- **`energy_evolution.png`** - Energy level evolution
- **`max_energy_evolution.png`** - Maximum energy trait evolution
- **`metabolism_evolution.png`** - Metabolism rate evolution
- **`move_cost_evolution.png`** - Movement cost evolution
- **`perception_evolution.png`** - Perception range evolution
- **`strength_evolution.png`** - Combat strength evolution
- **`aggression_evolution.png`** - Aggression level evolution
- **`repro_threshold_evolution.png`** - Reproduction threshold evolution
- **`repro_cost_evolution.png`** - Reproduction cost evolution
- **`summary_statistics.png`** - 4-panel overview of key metrics

### **Chart Features:**
- **High Resolution**: 150 DPI for publication quality
- **Professional Styling**: Clean titles, proper labels, grid lines
- **Individual Files**: Easy to use in reports, presentations, or analysis
- **Automatic Organization**: All charts saved in dedicated folder

## 🆚 Comparison with Matplotlib Version

| Feature | Matplotlib | Pygame |
|---------|------------|---------|
| **Performance** | ~10-15 FPS | 60 FPS |
| **Responsiveness** | Slower updates | Real-time |
| **User Experience** | Static plots | Interactive |
| **Memory Usage** | Higher | Lower |
| **Setup Complexity** | Simple | Simple |
| **Chart Output** | Display only | Save as files |
| **Layout** | Fixed | Grid-focused |
| **Grid Visibility** | Standard | Enhanced |

## 🎯 Layout Improvements

### **Grid-Focused Design:**
- **800x800 pixel grid** as the main visualization element
- **Larger window** (1800x1200) to accommodate the grid
- **Better cell visibility** with larger individual cells
- **Optimized spacing** between all UI elements

### **Enhanced Charts:**
- **Live Strength Distribution**: Real-time histogram of agent strength values
- **Daily Statistics**: Current day's fights, deaths, births, and carrots
- **Population & Energy**: Time series of key metrics
- **Compact layout** to the right of the grid

### **Visual Balance:**
- **Darker grid lines** (150,150,150) for excellent visibility
- **Consistent spacing** between all components
- **Better font sizing** for readability
- **Optimized chart dimensions** for the available space

## 🆕 New Features in This Version

### **Live Strength Distribution:**
- **Real-time histogram** showing current agent strength spread
- **10 bins** for clear visualization of strength distribution
- **Dynamic scaling** based on current population
- **Orange color scheme** for easy identification

### **Daily Statistics Panel:**
- **Current day metrics**: Fights, deaths, births, carrots
- **Live updates** as the simulation progresses
- **Compact display** with clear labeling
- **Purple theme** to distinguish from other charts

### **Enhanced Grid Visibility:**
- **Much larger grid** (800x800 vs previous 500x500)
- **Darker grid lines** for better cell separation
- **Larger cells** for easier agent and carrot identification
- **Main focus** of the entire interface

## 🐛 Troubleshooting

### **Common Issues**

1. **"pygame module not found"**
   - Run: `pip install pygame`

2. **"numpy module not found"**
   - Run: `pip install numpy`

3. **"matplotlib module not found"**
   - Run: `pip install matplotlib`
   - Note: Only needed for final charts, not for the main simulation

4. **Window doesn't appear**
   - Check if pygame is properly installed
   - Try running from command line: `python main_pygame.py`

5. **Performance issues**
   - Close other applications to free up system resources
   - The simulation automatically adjusts to maintain 60 FPS

6. **Charts not saving**
   - Ensure matplotlib is installed: `pip install matplotlib`
   - Check write permissions in the current directory
   - Look for `simulation_results/` folder after simulation ends

7. **Grid too large for screen**
   - The interface is designed for 1800x1200 resolution
   - Consider reducing your display scaling if on Windows
   - The grid will automatically scale down if needed

## 🎯 Customization

### **Modifying Colors**
Edit the `COLORS` dictionary in `pygame_monitor.py` to change the visual appearance.

### **Adjusting Layout**
Modify the layout constants in the `PygameMonitor.__init__` method to change window size and element positioning.

### **Adding New Charts**
Extend the `_init_charts()` method to add new data visualization.

### **Chart Saving Options**
Modify the `final_charts()` function in `main_pygame.py` to change chart styles, sizes, or save formats.

### **Grid Size Adjustment**
Change `self.grid_size` in the `PygameMonitor.__init__` method to make the grid larger or smaller.

## 📝 File Structure

- `pygame_monitor.py` - Pygame visualization system with grid-focused layout
- `main_pygame.py` - Main simulation with pygame monitor and chart saving
- `requirements.txt` - Python dependencies
- `README_pygame.md` - This documentation
- `simulation_results/` - Generated charts (created after simulation)

## 🚀 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run simulation: `python main_pygame.py`
3. Use Space to pause/play, S to stop
4. Watch the simulation evolve in the large grid window!
5. Monitor live strength distribution and daily statistics
6. When stopped, find your charts in `simulation_results/` folder
7. Close window or press Escape to quit

## ✨ What's New in This Version

- **🎯 Grid-Focused Layout**: 800x800 pixel main simulation window
- **📊 Live Strength Distribution**: Real-time histogram of agent strengths
- **📈 Daily Statistics Panel**: Current day's fights, deaths, births, carrots
- **👁️ Enhanced Grid Visibility**: Darker grid lines and larger cells
- **💾 Chart Saving**: Individual PNG files instead of blocking display
- **🎨 Better Visual Balance**: Optimized spacing and sizing for all elements

Enjoy your high-performance civilization simulation with the grid as the main focus! 🎮✨
