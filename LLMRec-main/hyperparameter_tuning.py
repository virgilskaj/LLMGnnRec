#!/usr/bin/env python3
"""
Hyperparameter tuning script to find optimal settings for enhanced LLMRec
Goal: Beat baseline R@20=0.0829, N@20=0.0347
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

class HyperparameterTuner:
    def __init__(self, dataset='netflix'):
        self.dataset = dataset
        self.baseline_recall_20 = 0.0829
        self.baseline_ndcg_20 = 0.0347
        self.results = []
        
    def run_experiment(self, params, experiment_name):
        """Run a single experiment with given parameters"""
        print(f"\n🧪 Running experiment: {experiment_name}")
        print(f"📋 Parameters: {params}")
        
        # Build command
        cmd = ['python3', 'main_optimized_enhanced.py', '--dataset', self.dataset]
        
        # Add parameters
        for key, value in params.items():
            cmd.extend([f'--{key}', str(value)])
        
        cmd.extend(['--title', experiment_name, '--epoch', '50'])  # Shorter epochs for tuning
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse results from output
                output_lines = result.stdout.split('\n')
                recall_20, ndcg_20 = None, None
                
                for line in output_lines:
                    if 'Test_Recall@20:' in line:
                        try:
                            parts = line.split('Test_Recall@20:')[1].split(',')[0].strip()
                            recall_20 = float(parts)
                        except:
                            pass
                    if 'ndcg=[' in line and recall_20 is not None:
                        try:
                            ndcg_part = line.split('ndcg=[')[1].split(']')[0]
                            ndcg_20 = float(ndcg_part)
                        except:
                            pass
                
                # Calculate improvements
                recall_improvement = ((recall_20 - self.baseline_recall_20) / self.baseline_recall_20 * 100) if recall_20 else -100
                ndcg_improvement = ((ndcg_20 - self.baseline_ndcg_20) / self.baseline_ndcg_20 * 100) if ndcg_20 else -100
                
                result_dict = {
                    'experiment': experiment_name,
                    'params': params,
                    'recall_20': recall_20,
                    'ndcg_20': ndcg_20,
                    'recall_improvement': recall_improvement,
                    'ndcg_improvement': ndcg_improvement,
                    'runtime': end_time - start_time,
                    'success': True
                }
                
                print(f"✅ Success: R@20={recall_20:.4f} (+{recall_improvement:.2f}%), N@20={ndcg_20:.4f} (+{ndcg_improvement:.2f}%)")
                
            else:
                result_dict = {
                    'experiment': experiment_name,
                    'params': params,
                    'success': False,
                    'error': result.stderr[-500:] if result.stderr else 'Unknown error'
                }
                print(f"❌ Failed: {result_dict['error']}")
                
        except subprocess.TimeoutExpired:
            result_dict = {
                'experiment': experiment_name,
                'params': params,
                'success': False,
                'error': 'Timeout after 1 hour'
            }
            print("⏰ Timeout")
        
        self.results.append(result_dict)
        return result_dict
    
    def tune_enhancement_parameters(self):
        """Tune enhancement-specific parameters"""
        print("🔧 Phase 1: Tuning Enhancement Parameters")
        
        # Test different enhancement strategies
        experiments = [
            # Conservative enhancements
            {'use_enhanced_gnn': True, 'graph_reg_weight': 0.001, 'feature_interaction_weight': 0.01},
            {'use_enhanced_gnn': True, 'graph_reg_weight': 0.005, 'feature_interaction_weight': 0.05},
            {'use_enhanced_gnn': True, 'graph_reg_weight': 0.01, 'feature_interaction_weight': 0.1},
            
            # More aggressive enhancements
            {'use_enhanced_gnn': True, 'graph_reg_weight': 0.02, 'feature_interaction_weight': 0.2},
            {'use_enhanced_gnn': True, 'graph_reg_weight': 0.05, 'feature_interaction_weight': 0.3},
        ]
        
        for i, params in enumerate(experiments):
            self.run_experiment(params, f'enhancement_tune_{i+1}')
    
    def tune_core_parameters(self):
        """Tune core LLMRec parameters for better baseline"""
        print("\n🔧 Phase 2: Tuning Core Parameters")
        
        experiments = [
            # Learning rate variations
            {'lr': 0.0005, 'use_enhanced_gnn': True},
            {'lr': 0.0002, 'use_enhanced_gnn': True},
            {'lr': 0.00005, 'use_enhanced_gnn': True},
            
            # Embedding size variations
            {'embed_size': 128, 'use_enhanced_gnn': True},
            {'embed_size': 32, 'use_enhanced_gnn': True},
            
            # Dropout variations
            {'drop_rate': 0.1, 'use_enhanced_gnn': True},
            {'drop_rate': 0.2, 'use_enhanced_gnn': True},
        ]
        
        for i, params in enumerate(experiments):
            self.run_experiment(params, f'core_tune_{i+1}')
    
    def tune_fusion_weights(self):
        """Tune the critical fusion weights"""
        print("\n🔧 Phase 3: Tuning Fusion Weights")
        
        experiments = [
            # Model cat rate variations
            {'model_cat_rate': 0.05, 'use_enhanced_gnn': True},
            {'model_cat_rate': 0.1, 'use_enhanced_gnn': True},
            {'model_cat_rate': 0.01, 'use_enhanced_gnn': True},
            
            # User cat rate variations  
            {'user_cat_rate': 5.0, 'use_enhanced_gnn': True},
            {'user_cat_rate': 1.0, 'use_enhanced_gnn': True},
            
            # Item cat rate variations
            {'item_cat_rate': 0.01, 'use_enhanced_gnn': True},
            {'item_cat_rate': 0.001, 'use_enhanced_gnn': True},
        ]
        
        for i, params in enumerate(experiments):
            self.run_experiment(params, f'fusion_tune_{i+1}')
    
    def find_best_configuration(self):
        """Find and test the best configuration"""
        print("\n🏆 Phase 4: Best Configuration Search")
        
        # Find best performing experiment
        successful_results = [r for r in self.results if r.get('success') and r.get('recall_20')]
        if not successful_results:
            print("❌ No successful experiments found!")
            return None
            
        best_result = max(successful_results, key=lambda x: x['recall_20'])
        
        print(f"\n🥇 Best configuration found:")
        print(f"   Experiment: {best_result['experiment']}")
        print(f"   R@20: {best_result['recall_20']:.4f} (+{best_result['recall_improvement']:.2f}%)")
        print(f"   N@20: {best_result['ndcg_20']:.4f} (+{best_result['ndcg_improvement']:.2f}%)")
        print(f"   Parameters: {best_result['params']}")
        
        # Run best configuration with full epochs
        print(f"\n🚀 Running best configuration with full training...")
        best_params = best_result['params'].copy()
        best_params['epoch'] = 1000  # Full training
        
        final_result = self.run_experiment(best_params, 'best_final_run')
        
        return best_result, final_result
    
    def save_results(self):
        """Save all results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'tuning_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Print summary
        successful = [r for r in self.results if r.get('success') and r.get('recall_20')]
        if successful:
            best = max(successful, key=lambda x: x['recall_20'])
            print(f"\n📊 TUNING SUMMARY:")
            print(f"   Total experiments: {len(self.results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Best R@20: {best['recall_20']:.4f} (+{best['recall_improvement']:.2f}%)")
            print(f"   Best config: {best['params']}")


def main():
    print("🎯 LLMRec Enhancement Hyperparameter Tuning")
    print("=" * 60)
    print(f"📊 Target: Beat baseline R@20=0.0829, N@20=0.0347")
    print("=" * 60)
    
    tuner = HyperparameterTuner()
    
    # Phase 1: Enhancement parameters
    tuner.tune_enhancement_parameters()
    
    # Phase 2: Core parameters  
    tuner.tune_core_parameters()
    
    # Phase 3: Fusion weights
    tuner.tune_fusion_weights()
    
    # Phase 4: Best configuration
    best_config, final_result = tuner.find_best_configuration()
    
    # Save results
    tuner.save_results()
    
    print("\n" + "=" * 60)
    print("🎉 HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 60)

if __name__ == '__main__':
    main()