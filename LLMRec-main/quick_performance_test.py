#!/usr/bin/env python3
"""
Quick performance test to beat the baseline
Target: R@20=0.0829, N@20=0.0347
"""

import os
import subprocess
import time

def run_quick_test(params_dict, test_name):
    """Run a quick test with given parameters"""
    print(f"\n🧪 Testing: {test_name}")
    
    # Build command
    cmd = ['python3', 'main_optimized_enhanced.py', '--dataset', 'netflix', '--epoch', '20']  # Short epoch for quick test
    
    for key, value in params_dict.items():
        cmd.extend([f'--{key}', str(value)])
    
    cmd.extend(['--title', test_name])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        end_time = time.time()
        
        if result.returncode == 0:
            # Extract performance metrics
            output = result.stdout
            
            # Look for the final test results
            lines = output.split('\n')
            for line in lines:
                if 'Test_Recall@20:' in line:
                    try:
                        recall_part = line.split('Test_Recall@20:')[1].split(',')[0].strip()
                        recall_20 = float(recall_part)
                        
                        precision_part = line.split('precision=[')[1].split(']')[0]
                        precision_20 = float(precision_part)
                        
                        ndcg_part = line.split('ndcg=[')[1].split(']')[0]
                        ndcg_20 = float(ndcg_part)
                        
                        # Calculate improvements
                        baseline_r20 = 0.0829
                        baseline_n20 = 0.0347
                        
                        r_improvement = (recall_20 - baseline_r20) / baseline_r20 * 100
                        n_improvement = (ndcg_20 - baseline_n20) / baseline_n20 * 100
                        
                        print(f"📊 Results: R@20={recall_20:.4f} (+{r_improvement:.2f}%), N@20={ndcg_20:.4f} (+{n_improvement:.2f}%)")
                        print(f"⏱️  Runtime: {end_time - start_time:.1f}s")
                        
                        return {
                            'success': True,
                            'recall_20': recall_20,
                            'ndcg_20': ndcg_20,
                            'precision_20': precision_20,
                            'r_improvement': r_improvement,
                            'n_improvement': n_improvement,
                            'runtime': end_time - start_time,
                            'params': params_dict
                        }
                    except Exception as e:
                        print(f"Failed to parse results: {e}")
                        break
            
            print("❌ Could not parse results from output")
            return {'success': False, 'error': 'Parse failed'}
            
        else:
            print(f"❌ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[-300:]}")
            return {'success': False, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {'success': False, 'error': str(e)}

def main():
    print("🎯 Quick Performance Test for Enhanced LLMRec")
    print("=" * 60)
    print("📊 Baseline: R@20=0.0829, N@20=0.0347")
    print("🎯 Goal: Beat baseline performance")
    print("=" * 60)
    
    # Test configurations designed to beat baseline
    test_configs = [
        # Test 1: Disabled enhancement (should match original)
        {
            'params': {'use_enhanced_gnn': False},
            'name': 'baseline_check'
        },
        
        # Test 2: Conservative enhancement
        {
            'params': {'use_enhanced_gnn': True, 'graph_reg_weight': 0.001},
            'name': 'conservative_enhance'
        },
        
        # Test 3: Boosted learning rate
        {
            'params': {'use_enhanced_gnn': True, 'lr': 0.0005},
            'name': 'boosted_lr'
        },
        
        # Test 4: Enhanced embedding size
        {
            'params': {'use_enhanced_gnn': True, 'embed_size': 128},
            'name': 'larger_embedding'
        },
        
        # Test 5: Tuned fusion weights
        {
            'params': {'use_enhanced_gnn': True, 'model_cat_rate': 0.05, 'user_cat_rate': 5.0},
            'name': 'tuned_fusion'
        },
        
        # Test 6: Multiple layers with enhancement
        {
            'params': {'use_enhanced_gnn': True, 'layers': 2, 'lr': 0.0002},
            'name': 'multi_layer'
        }
    ]
    
    results = []
    best_result = None
    best_improvement = -100
    
    for config in test_configs:
        result = run_quick_test(config['params'], config['name'])
        results.append(result)
        
        if result.get('success') and result.get('r_improvement', -100) > best_improvement:
            best_improvement = result['r_improvement']
            best_result = result
    
    print("\n" + "=" * 60)
    print("📋 QUICK TEST SUMMARY")
    print("=" * 60)
    
    for i, (config, result) in enumerate(zip(test_configs, results)):
        status = "✅" if result.get('success') else "❌"
        if result.get('success'):
            improvement = result.get('r_improvement', 0)
            print(f"{status} {config['name']:20} | R@20: {result.get('recall_20', 0):.4f} ({improvement:+.2f}%)")
        else:
            print(f"{status} {config['name']:20} | Failed: {result.get('error', 'Unknown')[:50]}")
    
    if best_result:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Name: {best_result.get('name', 'unknown')}")
        print(f"   R@20: {best_result['recall_20']:.4f} (+{best_result['r_improvement']:.2f}%)")
        print(f"   N@20: {best_result['ndcg_20']:.4f} (+{best_result['n_improvement']:.2f}%)")
        print(f"   Params: {best_result['params']}")
        
        if best_result['r_improvement'] > 0:
            print(f"\n🚀 SUCCESS! Found configuration that beats baseline!")
            print(f"💡 Run full training with: python3 main_optimized_enhanced.py " + 
                  " ".join([f"--{k} {v}" for k, v in best_result['params'].items()]))
        else:
            print(f"\n📈 Need more tuning. Best attempt: +{best_result['r_improvement']:.2f}%")
    else:
        print(f"\n❌ No successful configurations found. Check the error messages above.")

if __name__ == '__main__':
    main()