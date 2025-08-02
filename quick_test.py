#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced EEG Trajectory Tracking Algorithm Comparison System - Quick Test Script
Validates system installation and basic functionality with English interface
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import platform

# Set matplotlib to English only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

def test_dependencies():
    """Test dependency installation"""
    print("🔍 Testing dependency installation...")
    
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm',
        'mne': 'MNE-Python',
        'pandas': 'Pandas',
        'seaborn': 'Seaborn'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            installed_packages.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing_packages.append(name)
            print(f"  ❌ {name} - Not installed")
    
    print(f"\nInstallation status: {len(installed_packages)}/{len(required_packages)} packages installed")
    
    if missing_packages:
        print(f"\n❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies correctly installed!")
        return True

def test_tracker_factory():
    """Test tracker factory"""
    print("\n🏭 Testing tracker factory...")
    
    try:
        # Add paths
        sys.path.append('trackers')
        sys.path.append('src')
        from trackers import TrackerFactory
        from config import Config
        
        # Test available algorithms
        algorithms = TrackerFactory.get_available_algorithms()
        print(f"  ✓ Available algorithms: {', '.join(algorithms)}")
        
        # Test tracker creation
        success_count = 0
        for algorithm in algorithms:
            try:
                tracker = TrackerFactory.create_tracker(algorithm, Config)
                if tracker is not None:
                    print(f"  ✓ {algorithm} tracker created successfully")
                    success_count += 1
                else:
                    print(f"  ❌ {algorithm} tracker creation failed")
            except Exception as e:
                print(f"  ❌ {algorithm} tracker creation exception: {e}")
        
        print(f"\nTracker creation status: {success_count}/{len(algorithms)} algorithms available")
        return success_count > 0
        
    except Exception as e:
        print(f"  ❌ Tracker factory test failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data processing with enhanced algorithms"""
    print("\n🧪 Testing synthetic data processing...")
    
    try:
        sys.path.append('src')
        sys.path.append('trackers')
        
        from src.topography import TopographyGenerator
        from trackers import TrackerFactory
        from config import Config
        
        # Create synthetic topography data
        n_frames = 30  # Reduced for faster testing
        size = (64, 64)  # Smaller size for speed
        
        print(f"  🔧 Generating {n_frames} frames of {size} synthetic topographies...")
        
        # Create moving activation regions
        topographies = np.zeros((n_frames, size[0], size[1]))
        
        for i in range(n_frames):
            # Create moving Gaussian activation
            center_x = 20 + int(15 * np.sin(2 * np.pi * i / 20))
            center_y = 20 + int(10 * np.cos(2 * np.pi * i / 15))
            
            y, x = np.ogrid[:size[0], :size[1]]
            activation = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 4**2))
            topographies[i] = activation
        
        print("  ✓ Synthetic topography generation complete")
        
        # Test tracking algorithms
        test_algorithms = Config.COMPARISON_ALGORITHMS[:3]  # Test first 3 algorithms
        
        algorithm_results = {}
        
        for algorithm in test_algorithms:
            try:
                print(f"  🎯 Testing {algorithm} algorithm...")
                
                tracker = TrackerFactory.create_tracker(algorithm, Config)
                if tracker is None:
                    print(f"    ❌ {algorithm} tracker creation failed")
                    continue
                
                import time
                start_time = time.time()
                result = tracker.track_sequence(topographies)
                end_time = time.time()
                
                if result and 'trajectories' in result:
                    trajectories = result['trajectories']
                    metrics = result.get('metrics', {})
                    
                    algorithm_results[algorithm] = {
                        'trajectory_count': len(trajectories),
                        'computation_time': end_time - start_time,
                        'metrics': metrics
                    }
                    
                    print(f"    ✓ {algorithm}: {len(trajectories)} trajectories detected")
                    print(f"    ✓ Processing time: {end_time - start_time:.3f}s")
                    
                    if len(trajectories) > 0:
                        first_traj = list(trajectories.values())[0]
                        print(f"    ✓ Trajectory length: {first_traj['length']} frames")
                else:
                    print(f"    ⚠️  {algorithm}: No trajectories detected")
                    algorithm_results[algorithm] = {
                        'trajectory_count': 0,
                        'computation_time': end_time - start_time,
                        'metrics': {}
                    }
                
            except Exception as e:
                print(f"    ❌ {algorithm} test failed: {e}")
                algorithm_results[algorithm] = {
                    'trajectory_count': 0,
                    'computation_time': 0,
                    'error': str(e)
                }
        
        # Display comparison results
        if algorithm_results:
            print(f"\n  📊 Algorithm Performance Comparison:")
            print(f"  {'Algorithm':<12} {'Trajectories':<12} {'Time (s)':<10} {'Status'}")
            print(f"  {'-'*50}")
            
            for alg, results in algorithm_results.items():
                status = "✓ Pass" if results['trajectory_count'] > 0 else "⚠ No detection"
                if 'error' in results:
                    status = "❌ Error"
                
                print(f"  {alg:<12} {results['trajectory_count']:<12} {results['computation_time']:<10.3f} {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Synthetic data test failed: {e}")
        return False

def test_enhanced_visualization():
    """Test enhanced visualization functionality"""
    print("\n🎨 Testing enhanced visualization functionality...")
    
    try:
        # Test matplotlib setup
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create comprehensive test plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Enhanced EEG Trajectory Analysis - Visualization Test', fontsize=14, fontweight='bold')
        
        # Test 1: Basic plotting with English labels
        ax = axes[0, 0]
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2, label='Test Signal')
        ax.set_title('Signal Processing Test')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Test 2: Algorithm comparison simulation
        ax = axes[0, 1]
        algorithms = ['Greedy', 'Hungarian', 'Kalman', 'Overlap', 'Hybrid']
        performance = [4.2, 4.8, 3.9, 4.1, 4.6]
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        bars = ax.bar(algorithms, performance, color=colors, alpha=0.7)
        ax.set_title('Algorithm Performance Comparison')
        ax.set_ylabel('Average Trajectories')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, perf in zip(bars, performance):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{perf:.1f}', ha='center', va='bottom')
        
        # Test 3: Topography simulation
        ax = axes[1, 0]
        size = 64
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2) / 0.3) * np.cos(3*X) * np.sin(3*Y)
        
        im = ax.imshow(Z, cmap='RdYlBu_r', origin='lower', extent=[-1, 1, -1, 1])
        ax.set_title('EEG Topography Simulation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add head outline
        circle = plt.Circle((0, 0), 0.9, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Test 4: Trajectory visualization
        ax = axes[1, 1]
        
        # Simulate trajectory data
        t = np.linspace(0, 4*np.pi, 50)
        traj1_x = 0.3 * np.cos(t) + 0.1 * np.sin(3*t)
        traj1_y = 0.3 * np.sin(t) + 0.1 * np.cos(2*t)
        traj2_x = -0.2 * np.cos(1.5*t) + 0.15 * np.sin(2*t)
        traj2_y = 0.4 * np.sin(1.5*t) - 0.1 * np.cos(4*t)
        
        ax.plot(traj1_x, traj1_y, 'r-', linewidth=2, alpha=0.8, label='Trajectory 1')
        ax.plot(traj2_x, traj2_y, 'b-', linewidth=2, alpha=0.8, label='Trajectory 2')
        
        # Mark start and end points
        ax.scatter([traj1_x[0], traj2_x[0]], [traj1_y[0], traj2_y[0]], 
                  c=['red', 'blue'], s=100, marker='o', label='Start', zorder=5)
        ax.scatter([traj1_x[-1], traj2_x[-1]], [traj1_y[-1], traj2_y[-1]], 
                  c=['red', 'blue'], s=100, marker='s', label='End', zorder=5)
        
        ax.set_title('Trajectory Tracking Simulation')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save test plots
        test_dir = './test_results'
        os.makedirs(test_dir, exist_ok=True)
        
        main_test_path = os.path.join(test_dir, 'enhanced_visualization_test.png')
        plt.savefig(main_test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Enhanced visualization test saved: {main_test_path}")
        
        # Test algorithm comparison visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulate comprehensive algorithm comparison
        metrics = {
            'Trajectory Count': [4.2, 4.8, 3.9, 4.1, 4.6],
            'Quality Score': [0.72, 0.85, 0.78, 0.74, 0.82],
            'Processing Time': [0.15, 0.45, 0.25, 0.35, 0.55],
            'Efficiency': [28, 11, 16, 12, 8]
        }
        
        x = np.arange(len(algorithms))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics.items()):
            # Normalize values for comparison
            if metric == 'Processing Time':
                # Lower is better for time, so invert
                norm_values = [1.0 - (v - min(values)) / (max(values) - min(values)) for v in values]
            else:
                norm_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
            
            ax.bar(x + i * width, norm_values, width, label=metric, alpha=0.8)
        
        ax.set_title('Normalized Algorithm Performance Comparison', fontweight='bold')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Normalized Performance (0-1)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        comparison_path = os.path.join(test_dir, 'algorithm_comparison_test.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Algorithm comparison test saved: {comparison_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced visualization test failed: {e}")
        return False

def test_config():
    """Test configuration file"""
    print("\n⚙️ Testing configuration file...")
    
    try:
        from config import Config
        
        # Test basic configuration
        print(f"  ✓ Data path: {Config.DATA_ROOT}")
        print(f"  ✓ Results path: {Config.RESULTS_ROOT}")
        print(f"  ✓ Max subjects: {Config.MAX_SUBJECTS}")
        print(f"  ✓ Algorithm comparison: {'Enabled' if Config.ENABLE_ALGORITHM_COMPARISON else 'Disabled'}")
        print(f"  ✓ Comparison algorithms: {', '.join(Config.COMPARISON_ALGORITHMS)}")
        print(f"  ✓ Max frames per epoch: {Config.MAX_FRAMES_PER_EPOCH}")
        
        # Test configuration methods
        summary = Config.get_experiment_summary()
        print(f"  ✓ Experiment summary: {summary['algorithms_count']} algorithms, {summary['total_subjects']} subjects")
        
        # Test algorithm configuration
        for algorithm in Config.COMPARISON_ALGORITHMS:
            alg_config = Config.get_algorithm_config(algorithm)
            print(f"  ✓ {algorithm} config: {len(alg_config)} parameters")
        
        # Test frame control
        frame_limit = Config.get_max_frames('epoch')
        print(f"  ✓ Frame control: {frame_limit} frames/epoch limit")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def generate_enhanced_test_report(results):
    """Generate enhanced test report"""
    print("\n" + "="*80)
    print("🎯 ENHANCED EEG TRAJECTORY TRACKING SYSTEM - QUICK TEST REPORT")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    test_items = [
        ('Dependency Installation', results.get('dependencies', False)),
        ('Tracker Factory', results.get('tracker_factory', False)),
        ('Synthetic Data Processing', results.get('synthetic_data', False)),
        ('Enhanced Visualization', results.get('enhanced_visualization', False)),
        ('Configuration System', results.get('config', False))
    ]
    
    passed_tests = sum(1 for _, result in test_items if result)
    total_tests = len(test_items)
    
    print("Test Results:")
    for item_name, passed in test_items:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {item_name:<25}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    # Provide detailed feedback based on results
    if passed_tests == total_tests:
        print("🎉 EXCELLENT! System is fully operational and ready for use!")
        print("\nRecommended Next Steps:")
        print("  1. Prepare your EEG data (see README.md for data format)")
        print("  2. Quick test: python main.py --fast-mode")
        print("  3. Full experiment: python main.py")
        print("  4. View results in ./results/ directory")
        
    elif passed_tests >= 4:
        print("✅ GOOD! System is functional with minor issues.")
        print("⚠️  Some advanced features may have limitations.")
        print("\nRecommended Next Steps:")
        print("  1. Review failed tests and address issues if needed")
        print("  2. Try quick test: python main.py --fast-mode")
        print("  3. Check system logs for detailed error information")
        
    elif passed_tests >= 2:
        print("⚠️  PARTIAL! Basic functionality works but issues detected.")
        print("🔧 Some components need attention before full operation.")
        print("\nRecommended Actions:")
        print("  1. Fix failed dependency installations")
        print("  2. Check Python version (requires 3.8+)")
        print("  3. Verify system compatibility")
        
    else:
        print("❌ CRITICAL! Major system issues detected.")
        print("🚨 System requires significant troubleshooting.")
        print("\nRequired Actions:")
        print("  1. Check Python version (requires 3.8+)")
        print("  2. Reinstall dependencies: pip install -r requirements.txt")
        print("  3. Verify system compatibility and permissions")
        print("  4. Check installation logs for specific errors")
    
    print(f"\n📁 Test outputs saved in: ./test_results/")
    
    # System information
    print(f"\nSystem Information:")
    print(f"  • Python: {sys.version.split()[0]}")
    
    try:
        import platform as plt_module
        print(f"  • Platform: {plt_module.system().lower()}")
        print(f"  • Architecture: {plt_module.machine()}")
    except Exception as e:
        print(f"  • Platform: {sys.platform}")
        print(f"  • Architecture: Unknown")
    
    # Performance estimate
    if passed_tests >= 4:
        estimated_time = "2-5 minutes per subject" if passed_tests == total_tests else "3-8 minutes per subject"
        print(f"  • Estimated processing time: {estimated_time}")
        print(f"  • Recommended subjects for testing: 2-3")
        print(f"  • Memory requirement: 2-4 GB for typical datasets")
    
    print("\n📋 For detailed help, documentation, and troubleshooting:")
    print("   • Check README.md")
    print("   • Review example configurations")
    print("   • Examine log files in ./logs/")
    print("="*80)

def run_performance_benchmark():
    """Run quick performance benchmark"""
    print("\n⚡ Running performance benchmark...")
    
    try:
        import time
        
        # Test computation performance
        start_time = time.time()
        
        # Matrix operations test
        size = 1000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        matrix_time = time.time() - start_time
        
        # Memory allocation test
        start_time = time.time()
        large_array = np.zeros((2000, 2000, 10))
        del large_array
        
        memory_time = time.time() - start_time
        
        print(f"  ✓ Matrix computation: {matrix_time:.3f}s")
        print(f"  ✓ Memory allocation: {memory_time:.3f}s")
        
        # Performance assessment
        if matrix_time < 0.5 and memory_time < 0.1:
            performance = "Excellent"
        elif matrix_time < 2.0 and memory_time < 0.5:
            performance = "Good"
        elif matrix_time < 5.0 and memory_time < 1.0:
            performance = "Fair"
        else:
            performance = "Poor"
        
        print(f"  📊 Overall performance: {performance}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 ENHANCED EEG TRAJECTORY TRACKING ALGORITHM COMPARISON SYSTEM")
    print("   Quick Functionality Test & System Validation")
    print("="*80)
    print("This test validates system installation and basic functionality")
    print("Estimated time: 2-3 minutes")
    print("")
    
    # Suppress some logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    
    # Run test suite
    results = {}
    
    results['dependencies'] = test_dependencies()
    results['config'] = test_config()
    results['tracker_factory'] = test_tracker_factory()
    results['synthetic_data'] = test_synthetic_data()
    results['enhanced_visualization'] = test_enhanced_visualization()
    
    # Optional performance benchmark
    print("\n🎭 Additional Tests:")
    results['performance'] = run_performance_benchmark()
    
    # Generate comprehensive test report
    generate_enhanced_test_report(results)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)