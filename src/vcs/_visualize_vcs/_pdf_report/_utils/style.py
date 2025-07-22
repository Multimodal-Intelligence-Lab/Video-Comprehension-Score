import matplotlib.pyplot as plt

def setup_matplotlib_style():
    """Configure matplotlib styling for professional reports"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2
    # Modern color palette - professional blues and grays
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#3a6c8f', '#7aadcc', '#c9d7df', 
                                                      '#2c3e50', '#34495e', '#7f8c8d', '#bdc3c7'])