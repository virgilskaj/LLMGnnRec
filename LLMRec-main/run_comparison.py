#!/usr/bin/env python3
"""
Comparison script to run both original LLMRec and enhanced version with EmerG GNN integration
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_experiment(use_enhanced=False, dataset='netflix', additional_args=''):
    """
    Run experiment with either original or enhanced model
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if use_enhanced:
        script_name = 'main_enhanced.py'
        exp_name = f'enhanced_{dataset}_{timestamp}'
        print(f"🚀 Running Enhanced LLMRec with EmerG GNN integration...")
    else:
        script_name = 'main.py'
        exp_name = f'original_{dataset}_{timestamp}'
        print(f"📊 Running Original LLMRec...")
    
    # Build command
    cmd = [
        'python', script_name,
        '--dataset', dataset,
        '--title', exp_name
    ]
    
    # Add additional arguments
    if additional_args:
        cmd.extend(additional_args.split())
    
    # Add enhanced-specific arguments
    if use_enhanced:
        cmd.extend([
            '--use_enhanced_gnn', 'True',
            '--gnn_layers', '3',
            '--use_attention', 'True',
            '--attention_heads', '4',
            '--graph_reg_weight', '0.01',
            '--feature_interaction_weight', '0.1'
        ])
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        print(f"✅ Experiment {exp_name} completed!")
        print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
        if result.stderr:
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            
        return result.returncode == 0, exp_name
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {exp_name} timed out after 2 hours")
        return False, exp_name
    except Exception as e:
        print(f"❌ Experiment {exp_name} failed: {e}")
        return False, exp_name

def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--dataset', default='netflix', choices=['netflix', 'movielens'], 
                       help='Dataset to use')
    parser.add_argument('--run_both', action='store_true', 
                       help='Run both original and enhanced versions')
    parser.add_argument('--enhanced_only', action='store_true', 
                       help='Run only enhanced version')
    parser.add_argument('--original_only', action='store_true', 
                       help='Run only original version')
    parser.add_argument('--additional_args', default='', 
                       help='Additional arguments to pass to the script')
    
    args = parser.parse_args()
    
    print("🔬 LLMRec vs Enhanced LLMRec Comparison Experiment")
    print("=" * 80)
    
    results = {}
    
    if args.run_both or args.original_only:
        print("\n📊 Running Original LLMRec...")
        success, exp_name = run_experiment(
            use_enhanced=False, 
            dataset=args.dataset, 
            additional_args=args.additional_args
        )
        results['original'] = {'success': success, 'experiment': exp_name}
    
    if args.run_both or args.enhanced_only:
        print("\n🚀 Running Enhanced LLMRec with EmerG GNN...")
        success, exp_name = run_experiment(
            use_enhanced=True, 
            dataset=args.dataset, 
            additional_args=args.additional_args
        )
        results['enhanced'] = {'success': success, 'experiment': exp_name}
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for model_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{model_type.upper():>10}: {status} - {result['experiment']}")
    
    print("\n💡 Check the log files for detailed results and performance metrics!")

if __name__ == '__main__':
    main()