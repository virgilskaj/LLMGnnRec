#!/usr/bin/env python3
"""
Fixed enhanced version runner with better error handling
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run fixed enhanced LLMRec')
    parser.add_argument('--version', choices=['simple', 'complex'], default='simple',
                       help='Version to run: simple (stable) or complex (experimental)')
    parser.add_argument('--dataset', default='netflix', help='Dataset to use')
    
    args = parser.parse_args()
    
    if args.version == 'simple':
        print("🚀 Running Simple Enhanced LLMRec (Stable Version)")
        print("   Features: EmerG-inspired item attention + multi-modal fusion")
        os.system(f'python3 main_simple_enhanced.py --dataset {args.dataset}')
    else:
        print("🧪 Running Complex Enhanced LLMRec (Experimental Version)")  
        print("   Features: Full EmerG GNN integration + item-specific graphs")
        os.system(f'python3 main_enhanced.py --dataset {args.dataset}')

if __name__ == '__main__':
    main()