#!/usr/bin/env python3
"""
Performance boost runner with proven configurations
Designed to beat baseline: R@20=0.0829, N@20=0.0347
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run performance boost experiments')
    parser.add_argument('--strategy', choices=['quick', 'conservative', 'aggressive', 'optimal'], 
                       default='optimal', help='Enhancement strategy')
    parser.add_argument('--dataset', default='netflix', help='Dataset to use')
    
    args = parser.parse_args()
    
    print("🚀 LLMRec Performance Boost Runner")
    print("=" * 50)
    print(f"📊 Baseline: R@20=0.0829, N@20=0.0347")
    print(f"🎯 Strategy: {args.strategy}")
    print("=" * 50)
    
    if args.strategy == 'quick':
        print("⚡ Quick Test Strategy")
        cmd = f"python3 quick_performance_test.py"
        
    elif args.strategy == 'conservative':
        print("🛡️ Conservative Enhancement Strategy")
        cmd = f"python3 main_optimized_enhanced.py --dataset {args.dataset} --use_enhanced_gnn True --graph_reg_weight 0.001 --lr 0.0002 --title conservative_boost"
        
    elif args.strategy == 'aggressive':
        print("🔥 Aggressive Enhancement Strategy")
        cmd = f"python3 main_optimized_enhanced.py --dataset {args.dataset} --use_enhanced_gnn True --lr 0.0005 --embed_size 128 --model_cat_rate 0.1 --layers 2 --title aggressive_boost"
        
    elif args.strategy == 'optimal':
        print("🎯 Optimal Strategy (Recommended)")
        cmd = f"python3 main_optimized_enhanced.py --dataset {args.dataset} --use_enhanced_gnn True --lr 0.0003 --embed_size 128 --model_cat_rate 0.05 --user_cat_rate 4.0 --graph_reg_weight 0.005 --title optimal_boost"
    
    print(f"\n🔧 Running command:")
    print(f"   {cmd}")
    print("\n" + "=" * 50)
    
    # Execute command
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n✅ Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed. Check the output above for details.")
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)