#!/usr/bin/env python3
"""
Parameter tuning with fixed learning rate (0.0002)
Goal: Improve weak metrics while maintaining the successful configuration base
"""

import os
import subprocess
import time
import re
import json
from datetime import datetime

class FixedLRTuner:
    def __init__(self):
        self.base_cmd = [
            'python3', 'main.py', 
            '--dataset', 'netflix', 
            '--epoch', '150',  # Shorter for quicker iteration
            '--use_enhanced_gnn', 'True',
            '--layers', '2',
            '--lr', '0.0002'  # FIXED learning rate
        ]
        self.results = []
        
        # Current best results as baseline
        self.current_best = {
            'R@10': 0.047,
            'N@10': 0.024, 
            'R@20': 0.08618,
            'N@20': 0.034,
            'R@50': 0.14797,
            'N@50': 0.04634,
            'P@20': 0.00431
        }

    def extract_metrics(self, output_text):
        """Extract performance metrics from output"""
        lines = output_text.split('\n')
        
        for line in lines:
            if 'recall=[' in line and 'precision=[' in line and 'ndcg=[' in line:
                try:
                    # Extract recall values
                    recall_match = re.search(r'recall=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                    # Extract ndcg values  
                    ndcg_match = re.search(r'ndcg=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                    # Extract precision values
                    precision_match = re.search(r'precision=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                    
                    if recall_match and ndcg_match and precision_match:
                        return {
                            'R@10': float(recall_match.group(1)),
                            'R@20': float(recall_match.group(2)),
                            'R@50': float(recall_match.group(3)),
                            'N@10': float(ndcg_match.group(1)),
                            'N@20': float(ndcg_match.group(2)),
                            'N@50': float(ndcg_match.group(3)),
                            'P@10': float(precision_match.group(1)),
                            'P@20': float(precision_match.group(2)),
                            'P@50': float(precision_match.group(3))
                        }
                except Exception as e:
                    continue
        
        return None

    def run_experiment(self, params, exp_name):
        """Run single experiment with given parameters"""
        print(f"\n🧪 实验: {exp_name}")
        print("=" * 50)
        
        # Build command
        cmd = self.base_cmd.copy()
        
        # Add specific parameters
        for key, value in params.items():
            cmd.extend([f'--{key}', str(value)])
        
        cmd.extend(['--title', exp_name.replace(' ', '_').replace(':', '')])
        
        print(f"🔧 参数: {params}")
        print(f"📋 命令: {' '.join(cmd[-10:])}")  # Show last 10 args
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4800)  # 80 min timeout
            end_time = time.time()
            
            if result.returncode == 0:
                metrics = self.extract_metrics(result.stdout)
                
                if metrics:
                    # Calculate improvements vs current best
                    improvements = {}
                    for key in ['R@10', 'R@20', 'R@50', 'N@10', 'N@20', 'N@50', 'P@20']:
                        if key in self.current_best:
                            improvements[key] = (metrics[key] - self.current_best[key]) / self.current_best[key] * 100
                    
                    # Focus on weak metrics
                    weak_metrics_avg = (improvements.get('R@10', -100) + improvements.get('N@10', -100) + improvements.get('N@20', -100)) / 3
                    strong_metrics_avg = (improvements.get('R@20', -100) + improvements.get('R@50', -100)) / 2
                    
                    print(f"📊 结果:")
                    print(f"   R@10: {metrics['R@10']:.5f} ({improvements.get('R@10', 0):+.2f}%)")
                    print(f"   N@10: {metrics['N@10']:.5f} ({improvements.get('N@10', 0):+.2f}%)")
                    print(f"   R@20: {metrics['R@20']:.5f} ({improvements.get('R@20', 0):+.2f}%)")
                    print(f"   N@20: {metrics['N@20']:.5f} ({improvements.get('N@20', 0):+.2f}%)")
                    print(f"   R@50: {metrics['R@50']:.5f} ({improvements.get('R@50', 0):+.2f}%)")
                    print(f"   弱指标平均: {weak_metrics_avg:+.2f}%")
                    print(f"   强指标平均: {strong_metrics_avg:+.2f}%")
                    print(f"⏱️  运行时间: {end_time - start_time:.1f}秒")
                    
                    status = "🚀 优秀" if weak_metrics_avg > 0 else "📈 改进" if weak_metrics_avg > -5 else "📉 下降"
                    print(f"🎯 评估: {status}")
                    
                    return {
                        'success': True,
                        'metrics': metrics,
                        'improvements': improvements,
                        'weak_metrics_avg': weak_metrics_avg,
                        'strong_metrics_avg': strong_metrics_avg,
                        'params': params,
                        'runtime': end_time - start_time
                    }
                else:
                    print("❌ 无法解析性能指标")
                    return {'success': False, 'error': '解析失败'}
            else:
                print(f"❌ 实验失败: {result.stderr[-300:]}")
                return {'success': False, 'error': result.stderr[-300:]}
                
        except subprocess.TimeoutExpired:
            print("⏰ 实验超时")
            return {'success': False, 'error': '超时'}
        except Exception as e:
            print(f"❌ 异常: {e}")
            return {'success': False, 'error': str(e)}

    def run_tuning_experiments(self):
        """Run systematic parameter tuning experiments"""
        print("🎯 固定学习率参数调优 (lr=0.0002)")
        print("=" * 60)
        print("📊 目标: 改进 R@10, N@10, N@20 同时保持其他指标优势")
        print("=" * 60)
        
        # Experiment configurations
        experiments = [
            # Experiment 1: 调整嵌入维度
            {
                'params': {'embed_size': 96},
                'name': '实验1: 嵌入维度96'
            },
            {
                'params': {'embed_size': 160},
                'name': '实验2: 嵌入维度160'
            },
            
            # Experiment 2: 调整模型融合权重 (关键参数)
            {
                'params': {'model_cat_rate': 0.03},
                'name': '实验3: 模型融合权重0.03'
            },
            {
                'params': {'model_cat_rate': 0.06},
                'name': '实验4: 模型融合权重0.06'
            },
            {
                'params': {'model_cat_rate': 0.04, 'user_cat_rate': 3.5},
                'name': '实验5: 平衡融合权重'
            },
            
            # Experiment 3: 调整dropout和正则化
            {
                'params': {'drop_rate': 0.05, 'weight_decay': 0.0005},
                'name': '实验6: 轻度正则化'
            },
            {
                'params': {'drop_rate': 0.1, 'feat_reg_decay': 5e-5},
                'name': '实验7: 特征正则化'
            },
            
            # Experiment 4: 调整批次大小
            {
                'params': {'batch_size': 512},
                'name': '实验8: 小批次512'
            },
            {
                'params': {'batch_size': 2048},
                'name': '实验9: 大批次2048'
            },
            
            # Experiment 5: 调整增强相关参数
            {
                'params': {'aug_sample_rate': 0.15, 'aug_mf_rate': 0.02},
                'name': '实验10: 增强样本率15%'
            },
            {
                'params': {'mm_mf_rate': 0.0005, 'model_cat_rate': 0.04},
                'name': '实验11: 多模态损失权重'
            },
            
            # Experiment 6: 组合优化 (基于成功配置的微调)
            {
                'params': {
                    'embed_size': 96,
                    'model_cat_rate': 0.04,
                    'user_cat_rate': 3.8,
                    'drop_rate': 0.03
                },
                'name': '实验12: 组合优化A'
            },
            {
                'params': {
                    'embed_size': 128,
                    'model_cat_rate': 0.035,
                    'user_cat_rate': 4.2,
                    'item_cat_rate': 0.008,
                    'batch_size': 768
                },
                'name': '实验13: 组合优化B'
            },
            
            # Experiment 7: 针对Top-K的特殊配置
            {
                'params': {
                    'embed_size': 80,
                    'model_cat_rate': 0.045,
                    'user_cat_rate': 4.5,
                    'drop_rate': 0.02,
                    'prune_loss_drop_rate': 0.65  # 调整剪枝损失
                },
                'name': '实验14: Top-K专用配置'
            }
        ]
        
        # Run all experiments
        for exp in experiments:
            result = self.run_experiment(exp['params'], exp['name'])
            self.results.append((exp['name'], result))
            
            # Brief pause between experiments
            time.sleep(2)
        
        return self.results

    def analyze_results(self):
        """Analyze and report results"""
        print("\n" + "=" * 60)
        print("📋 参数调优结果汇总")
        print("=" * 60)
        
        successful_results = [(name, result) for name, result in self.results if result.get('success')]
        
        if not successful_results:
            print("❌ 所有实验都失败了")
            return None
        
        # Sort by weak metrics improvement
        successful_results.sort(key=lambda x: x[1].get('weak_metrics_avg', -100), reverse=True)
        
        print(f"{'实验名称':25} | {'R@10':8} | {'N@10':8} | {'R@20':8} | {'N@20':8} | {'弱指标平均':10}")
        print("-" * 80)
        
        best_weak_improvement = None
        best_overall = None
        best_weak_score = -100
        best_overall_score = -100
        
        for name, result in successful_results:
            if result.get('success'):
                metrics = result['metrics']
                improvements = result['improvements']
                weak_avg = result.get('weak_metrics_avg', -100)
                
                # Calculate overall score (weighted towards weak metrics)
                overall_score = (weak_avg * 2 + result.get('strong_metrics_avg', 0)) / 3
                
                status = "🚀" if weak_avg > 0 else "📈" if weak_avg > -5 else "📉"
                
                print(f"{name[:24]:25} | {metrics['R@10']:.5f} | {metrics['N@10']:.5f} | {metrics['R@20']:.5f} | {metrics['N@20']:.5f} | {weak_avg:+.2f}% {status}")
                
                # Track best results
                if weak_avg > best_weak_score:
                    best_weak_score = weak_avg
                    best_weak_improvement = (name, result)
                
                if overall_score > best_overall_score:
                    best_overall_score = overall_score
                    best_overall = (name, result)
        
        # Report best configurations
        print("\n" + "=" * 60)
        
        if best_weak_improvement:
            name, result = best_weak_improvement
            print(f"🎯 最佳弱指标改进: {name}")
            print(f"   弱指标平均改进: {result['weak_metrics_avg']:+.2f}%")
            print(f"   参数: {result['params']}")
            
            # Generate command for best weak metrics config
            param_str = ' '.join([f'--{k} {v}' for k, v in result['params'].items()])
            print(f"\n🚀 运行最佳弱指标配置:")
            print(f"   python3 main.py --dataset netflix --use_enhanced_gnn True --layers 2 --lr 0.0002 {param_str} --epoch 300 --title best_weak_metrics")
        
        if best_overall and best_overall != best_weak_improvement:
            name, result = best_overall
            print(f"\n🏆 最佳整体配置: {name}")
            print(f"   整体得分: {best_overall_score:.2f}")
            print(f"   参数: {result['params']}")
            
            # Generate command for best overall config
            param_str = ' '.join([f'--{k} {v}' for k, v in result['params'].items()])
            print(f"\n🚀 运行最佳整体配置:")
            print(f"   python3 main.py --dataset netflix --use_enhanced_gnn True --layers 2 --lr 0.0002 {param_str} --epoch 300 --title best_overall")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'fixed_lr_tuning_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump({
                'base_command': self.base_cmd,
                'current_best': self.current_best,
                'results': [{'name': name, 'result': result} for name, result in self.results]
            }, f, indent=2)
        
        print(f"\n💾 结果已保存到: {results_file}")
        
        return best_weak_improvement, best_overall

def main():
    print("🎯 固定学习率参数调优实验")
    print("🔧 基础配置: --lr 0.0002 --layers 2 --use_enhanced_gnn True")
    print("📊 当前最佳: R@20=0.08618 (+3.9%), R@50=0.14797 (+7.1%)")
    print("🎯 目标: 改进 R@10, N@10, N@20")
    
    tuner = FixedLRTuner()
    
    # Run all tuning experiments
    results = tuner.run_tuning_experiments()
    
    # Analyze and get best configurations
    best_weak, best_overall = tuner.analyze_results()
    
    print("\n" + "=" * 60)
    print("🎉 参数调优完成!")
    
    if best_weak:
        print("✅ 找到了改进弱指标的配置")
        print("💡 建议运行上面显示的最佳配置进行完整训练")
    else:
        print("📈 未找到显著改进，建议:")
        print("   1. 尝试更长的训练 (--epoch 400)")
        print("   2. 调整学习率调度")
        print("   3. 使用更复杂的增强模型")
    
    return best_weak is not None

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)