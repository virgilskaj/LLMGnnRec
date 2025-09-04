#!/usr/bin/env python3
"""
Validate that we can reproduce the baseline performance
Target: R@20=0.0829, N@20=0.0347
"""

import os
import subprocess
import sys

def run_baseline_validation():
    """Run baseline validation to ensure we can reproduce original performance"""
    print("🔍 Baseline Validation Test")
    print("=" * 50)
    print("📊 Target: R@20=0.0829, N@20=0.0347")
    print("🎯 Goal: Reproduce original performance exactly")
    print("=" * 50)
    
    # Test 1: Original model (should match baseline)
    print("\n📊 Test 1: Original Model")
    cmd1 = ['python3', 'main.py', '--dataset', 'netflix', '--epoch', '50', '--title', 'baseline_validation']
    
    print(f"Command: {' '.join(cmd1)}")
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=3600)
        
        if result1.returncode == 0:
            print("✅ Original model ran successfully")
            
            # Extract performance
            lines = result1.stdout.split('\n')
            for line in lines:
                if 'Test_Recall@20:' in line:
                    print(f"📈 Result: {line.strip()}")
                    break
        else:
            print("❌ Original model failed")
            print(f"Error: {result1.stderr[-500:]}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run original model: {e}")
        return False
    
    # Test 2: Enhanced model with enhancement disabled (should match baseline)
    print("\n🔧 Test 2: Enhanced Model (Enhancement Disabled)")
    cmd2 = ['python3', 'main.py', '--dataset', 'netflix', '--epoch', '50', '--use_enhanced_gnn', 'False', '--title', 'enhanced_disabled']
    
    print(f"Command: {' '.join(cmd2)}")
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=3600)
        
        if result2.returncode == 0:
            print("✅ Enhanced model (disabled) ran successfully")
            
            # Extract performance
            lines = result2.stdout.split('\n')
            for line in lines:
                if 'Test_Recall@20:' in line:
                    print(f"📈 Result: {line.strip()}")
                    break
        else:
            print("❌ Enhanced model (disabled) failed")
            print(f"Error: {result2.stderr[-500:]}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run enhanced model (disabled): {e}")
        return False
    
    # Test 3: Enhanced model with very light enhancement
    print("\n🚀 Test 3: Enhanced Model (Light Enhancement)")
    cmd3 = ['python3', 'main.py', '--dataset', 'netflix', '--epoch', '50', '--use_enhanced_gnn', 'True', '--title', 'enhanced_light']
    
    print(f"Command: {' '.join(cmd3)}")
    
    try:
        result3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=3600)
        
        if result3.returncode == 0:
            print("✅ Enhanced model (enabled) ran successfully")
            
            # Extract performance
            lines = result3.stdout.split('\n')
            for line in lines:
                if 'Test_Recall@20:' in line:
                    print(f"📈 Result: {line.strip()}")
                    # Parse and compare
                    try:
                        recall_str = line.split('Test_Recall@20:')[1].split(',')[0].strip()
                        recall_20 = float(recall_str)
                        baseline = 0.0829
                        improvement = (recall_20 - baseline) / baseline * 100
                        
                        if improvement > 0:
                            print(f"🎉 SUCCESS! Improvement: +{improvement:.2f}%")
                        else:
                            print(f"📉 Need tuning: {improvement:.2f}%")
                    except:
                        pass
                    break
        else:
            print("❌ Enhanced model (enabled) failed")
            print(f"Error: {result3.stderr[-500:]}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run enhanced model (enabled): {e}")
        return False
    
    return True

def main():
    success = run_baseline_validation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ VALIDATION COMPLETED")
        print("\n💡 Next Steps:")
        print("   1. If Test 1 & 2 match baseline → Enhancement framework is correct")
        print("   2. If Test 3 shows improvement → Success!")
        print("   3. If Test 3 shows decline → Need parameter tuning")
        print("\n🚀 Try these if enhancement needs tuning:")
        print("   python3 main.py --use_enhanced_gnn True --lr 0.0005")
        print("   python3 main.py --use_enhanced_gnn True --embed_size 128") 
        print("   python3 main.py --use_enhanced_gnn True --model_cat_rate 0.05")
    else:
        print("❌ VALIDATION FAILED")
        print("   Check the error messages above")
    
    return 0 if success else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)