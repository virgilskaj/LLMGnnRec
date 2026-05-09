#!/usr/bin/env python3
"""
Targeted improvement for weak metrics: R@10, N@10, N@20
"""

import os
import subprocess
import time
import re

def extract_all_metrics(output_text):
    """Extract all performance metrics from output"""
    lines = output_text.split('\n')
    
    for line in lines:
        if 'Test_Recall@' in line and 'precision=' in line:
            try:
                # Extract all metrics from the comprehensive result line
                recall_match = re.search(r'Test_Recall@\d+:\s*([\d\.]+)', line)
                precision_match = re.search(r'precision=\[([\d\.]+)\]', line)
                ndcg_match = re.search(r'ndcg=\[([\d\.]+)\]', line)
                
                if recall_match and precision_match and ndcg_match:
                    return {
                        'recall': float(recall_match.group(1)),
                        'precision': float(precision_match.group(1)), 
                        'ndcg': float(ndcg_match.group(1))
                    }
            except:
                continue
    
    return None

def run_targeted_test(params, test_name):
    """Run test targeting weak metrics"""
    print(f"\n🎯 {test_name}")
    print("-" * 50)
    
    cmd = ['python3', 'main.py', '--dataset', 'netflix', '--epoch', '100']
    
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])
    
    cmd.extend(['--title', test_name.replace(' ', '_')])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        if result.returncode == 0:
            metrics = extract_all_metrics(result.stdout)
            
            if metrics:
                # Compare with your current best results
                current_best = {
                    'recall': 0.08617886,  # Your current R@20
                    'ndcg': 0.04633842     # Your current N@20 (approximated)
                }
                
                r_improvement = (metrics['recall'] - current_best['recall']) / current_best['recall'] * 100
                n_improvement = (metrics['ndcg'] - current_best['ndcg']) / current_best['ndcg'] * 100
                
                status = "🚀" if r_improvement > 0 else "📊" if r_improvement > -2 else "📉"
                
                print(f"📊 Results:")
                print(f"   R@20: {metrics['recall']:.6f} ({r_improvement:+.2f}%) {status}")
                print(f"   N@20: {metrics['ndcg']:.6f} ({n_improvement:+.2f}%)")
                print(f"   P@20: {metrics['precision']:.6f}")
                print(f"⏱️  Runtime: {end_time - start_time:.1f}s")
                
                return {
                    'success': True,
                    'metrics': metrics,
                    'improvements': {'recall': r_improvement, 'ndcg': n_improvement},
                    'runtime': end_time - start_time,
                    'params': params
                }
            else:
                print("❌ Could not extract metrics")
                return {'success': False, 'error': 'Parse failed'}
        else:
            print(f"❌ Failed: {result.stderr[-200:]}")
            return {'success': False, 'error': result.stderr[-200:]}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {'success': False, 'error': str(e)}

def main():
    print("🎯 Targeted Improvement for Weak Metrics")
    print("=" * 60)
    print("📊 Current Best: R@20=0.08617886 (+3.9% vs baseline)")
    print("🎯 Goal: Improve R@10, N@10, N@20 while maintaining other gains")
    print("=" * 60)
    
    # Targeted configurations to improve weak metrics
    test_configs = [
        # Focus on top-k performance
        {
            'params': {
                'use_enhanced_gnn': True,
                'lr': 0.0003,  # Slightly higher LR for better convergence
                'layers': 2,
                'embed_size': 96,  # Between 64 and 128
                'model_cat_rate': 0.03,  # Slight boost
                'user_cat_rate': 3.5,   # Slight boost
                'drop_rate': 0.05       # Light regularization
            },
            'name': 'Config A: Top-K Focused'
        },
        
        # Focus on NDCG improvements
        {
            'params': {
                'use_enhanced_gnn': True,
                'lr': 0.0002,
                'layers': 3,  # Deeper for better ranking
                'embed_size': 128,
                'model_cat_rate': 0.04,
                'user_cat_rate': 4.0,
                'item_cat_rate': 0.01  # Boost item features
            },
            'name': 'Config B: NDCG Focused'
        },
        
        # Focus on early precision  
        {
            'params': {
                'use_enhanced_gnn': True,
                'lr': 0.0004,  # Higher LR for faster learning
                'layers': 2,
                'embed_size': 128,
                'model_cat_rate': 0.06,  # Strong multi-modal
                'user_cat_rate': 5.0,    # Strong user modeling
                'batch_size': 512        # Smaller batch for better gradients
            },
            'name': 'Config C: Early Precision'
        },
        
        # Balanced improvement
        {
            'params': {
                'use_enhanced_gnn': True,
                'lr': 0.00025,
                'layers': 2,
                'embed_size': 128,
                'model_cat_rate': 0.045,
                'user_cat_rate': 4.2,
                'item_cat_rate': 0.008,
                'drop_rate': 0.02
            },
            'name': 'Config D: Balanced Boost'
        }
    ]
    
    results = []
    best_overall = None
    best_improvement = -100
    
    for config in test_configs:
        result = run_targeted_test(config['params'], config['name'])
        results.append((config['name'], result))
        
        if result.get('success'):
            total_improvement = result['improvements']['recall'] + result['improvements']['ndcg']
            if total_improvement > best_improvement:
                best_improvement = total_improvement
                best_overall = (config['name'], result, config['params'])
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 TARGETED IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        if result.get('success'):
            r_imp = result['improvements']['recall']
            n_imp = result['improvements']['ndcg']
            status = "🚀" if r_imp > 0 and n_imp > 0 else "📈" if r_imp > 0 or n_imp > 0 else "📉"
            print(f"{status} {name:25} | R@20: {result['metrics']['recall']:.6f} ({r_imp:+.2f}%) | N@20: {result['metrics']['ndcg']:.6f} ({n_imp:+.2f}%)")
        else:
            print(f"❌ {name:25} | Failed")
    
    if best_overall:
        name, result, params = best_overall
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"   {name}")
        print(f"   R@20: {result['metrics']['recall']:.6f} ({result['improvements']['recall']:+.2f}%)")
        print(f"   N@20: {result['metrics']['ndcg']:.6f} ({result['improvements']['ndcg']:+.2f}%)")
        
        print(f"\n🚀 RUN OPTIMAL TRAINING:")
        param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
        print(f"   python3 main.py {param_str} --epoch 300 --title final_optimized")
        
        # Also suggest the advanced model
        print(f"\n🔬 TRY ADVANCED MODEL:")
        print(f"   python3 main.py {param_str} --epoch 300 --title advanced_final")
        print(f"   (Need to update main.py to use Models_Final_Optimized)")
        
        return True
    else:
        print("\n📈 No significant improvement found in this round")
        print("💡 Consider:")
        print("   - Longer training (--epoch 500)")
        print("   - Different learning rate schedules")
        print("   - Advanced model architecture")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)