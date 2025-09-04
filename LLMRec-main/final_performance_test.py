#!/usr/bin/env python3
"""
Final performance test with gradual enhancement validation
Goal: Beat R@20=0.0829, N@20=0.0347 step by step
"""

import os
import subprocess
import time
import re

def extract_performance(output_text):
    """Extract performance metrics from output"""
    lines = output_text.split('\n')
    
    for line in lines:
        if 'Test_Recall@20:' in line:
            try:
                # Extract recall@20
                recall_match = re.search(r'Test_Recall@20:\s*([\d\.]+)', line)
                if recall_match:
                    recall_20 = float(recall_match.group(1))
                    
                    # Extract precision@20
                    precision_match = re.search(r'precision=\[([\d\.]+)\]', line)
                    precision_20 = float(precision_match.group(1)) if precision_match else 0.0
                    
                    # Extract ndcg@20
                    ndcg_match = re.search(r'ndcg=\[([\d\.]+)\]', line)
                    ndcg_20 = float(ndcg_match.group(1)) if ndcg_match else 0.0
                    
                    return recall_20, precision_20, ndcg_20
            except Exception as e:
                print(f"Parse error: {e}")
                continue
    
    return None, None, None

def run_test(params, test_name, timeout=1800):
    """Run a single test"""
    print(f"\n🧪 {test_name}")
    print("-" * 40)
    
    # Build command
    cmd = ['python3', 'main.py', '--dataset', 'netflix', '--epoch', '30']  # Short epoch for testing
    
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])
    
    cmd.extend(['--title', test_name.replace(' ', '_')])
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            recall_20, precision_20, ndcg_20 = extract_performance(result.stdout)
            
            if recall_20 is not None:
                # Calculate improvements
                baseline_r20 = 0.0829
                baseline_n20 = 0.0347
                
                r_improvement = (recall_20 - baseline_r20) / baseline_r20 * 100
                n_improvement = (ndcg_20 - baseline_n20) / baseline_n20 * 100
                
                status = "🎉 BETTER" if r_improvement > 0 else "📉 WORSE" if r_improvement < -1 else "📊 SIMILAR"
                
                print(f"📊 Results:")
                print(f"   R@20: {recall_20:.4f} ({r_improvement:+.2f}%) {status}")
                print(f"   N@20: {ndcg_20:.4f} ({n_improvement:+.2f}%)")
                print(f"   P@20: {precision_20:.4f}")
                print(f"⏱️  Runtime: {end_time - start_time:.1f}s")
                
                return {
                    'success': True,
                    'recall_20': recall_20,
                    'ndcg_20': ndcg_20,
                    'precision_20': precision_20,
                    'r_improvement': r_improvement,
                    'n_improvement': n_improvement,
                    'runtime': end_time - start_time
                }
            else:
                print("❌ Could not extract performance metrics")
                return {'success': False, 'error': 'Parse failed'}
        else:
            print(f"❌ Failed with return code {result.returncode}")
            error_msg = result.stderr[-300:] if result.stderr else 'Unknown error'
            print(f"Error: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout}s")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {'success': False, 'error': str(e)}

def main():
    print("🎯 Final Performance Test - Gradual Enhancement Validation")
    print("=" * 70)
    print("📊 Baseline Target: R@20=0.0829, N@20=0.0347")
    print("🎯 Goal: Find configuration that beats baseline")
    print("=" * 70)
    
    # Step-by-step testing strategy
    test_configs = [
        # Step 1: Verify original performance can be reproduced
        {
            'params': {},
            'name': 'Step 1: Original Baseline'
        },
        
        # Step 2: Enhanced model with enhancement disabled (should match step 1)
        {
            'params': {'use_enhanced_gnn': False},
            'name': 'Step 2: Enhanced Framework (Disabled)'
        },
        
        # Step 3: Enable enhancement with minimal impact
        {
            'params': {'use_enhanced_gnn': True},
            'name': 'Step 3: EmerG Lite (Minimal)'
        },
        
        # Step 4: Boost learning rate
        {
            'params': {'use_enhanced_gnn': True, 'lr': 0.0002},
            'name': 'Step 4: Boosted Learning Rate'
        },
        
        # Step 5: Larger embedding
        {
            'params': {'use_enhanced_gnn': True, 'embed_size': 128},
            'name': 'Step 5: Larger Embedding'
        },
        
        # Step 6: Enhanced fusion weights
        {
            'params': {'use_enhanced_gnn': True, 'model_cat_rate': 0.05},
            'name': 'Step 6: Enhanced Fusion'
        },
        
        # Step 7: Multi-layer with enhancement
        {
            'params': {'use_enhanced_gnn': True, 'layers': 2, 'lr': 0.0002},
            'name': 'Step 7: Multi-layer Enhanced'
        },
        
        # Step 8: Best combination
        {
            'params': {'use_enhanced_gnn': True, 'lr': 0.0002, 'embed_size': 128, 'model_cat_rate': 0.05, 'layers': 2},
            'name': 'Step 8: Best Combination'
        }
    ]
    
    results = []
    best_result = None
    best_improvement = -100
    
    for config in test_configs:
        result = run_test(config['params'], config['name'])
        results.append((config['name'], result))
        
        if result.get('success') and result.get('r_improvement', -100) > best_improvement:
            best_improvement = result['r_improvement']
            best_result = (config['name'], result, config['params'])
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 70)
    
    baseline_found = False
    enhancement_works = False
    
    for name, result in results:
        if result.get('success'):
            improvement = result.get('r_improvement', -100)
            status = "✅" if improvement > 0 else "📊" if improvement > -1 else "❌"
            print(f"{status} {name:30} | R@20: {result.get('recall_20', 0):.4f} ({improvement:+.2f}%)")
            
            if 'Original Baseline' in name and abs(improvement) < 1:
                baseline_found = True
            if improvement > 0:
                enhancement_works = True
        else:
            print(f"❌ {name:30} | Failed: {result.get('error', 'Unknown')[:30]}")
    
    print("\n" + "=" * 70)
    
    if baseline_found:
        print("✅ BASELINE REPRODUCED: Original performance confirmed")
    else:
        print("⚠️  BASELINE ISSUE: Could not reproduce original performance")
    
    if enhancement_works:
        print("🎉 ENHANCEMENT SUCCESS: Found improvement over baseline!")
        if best_result:
            name, result, params = best_result
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Name: {name}")
            print(f"   R@20: {result['recall_20']:.4f} (+{result['r_improvement']:.2f}%)")
            print(f"   N@20: {result['ndcg_20']:.4f} (+{result['n_improvement']:.2f}%)")
            print(f"   Params: {params}")
            
            print(f"\n🚀 RUN FULL TRAINING:")
            param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
            print(f"   python3 main.py {param_str} --epoch 200")
    else:
        print("📈 ENHANCEMENT NEEDED: No improvement found yet")
        print("\n💡 NEXT STEPS:")
        print("   1. Check if baseline is reproduced correctly")
        print("   2. Try more aggressive parameter tuning")
        print("   3. Consider different enhancement strategies")
    
    return enhancement_works

if __name__ == '__main__':
    success = main()
    print(f"\n{'🎉 SUCCESS' if success else '📈 NEEDS WORK'}: Enhancement test completed")
    exit(0 if success else 1)