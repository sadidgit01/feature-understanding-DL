import os
import csv
from linear_probe import run_linear_probe

def main():
    models = ['resnet101', 'googlenet', 'zfnet']
    results = []
    
    print("-" * 40)
    print(f"{'Model':<15} | {'Linear Probe Accuracy'}")
    print("-" * 40)
    
    for model_name in models:
        try:
            name, accuracy = run_linear_probe(model_name)
            results.append((name, accuracy))
            print(f"{name:<15} | {accuracy:.4f}")
        except Exception as e:
            print(f"{model_name:<15} | Error: {e}")
            
    print("-" * 40)
    
    # Save results to csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'linear_probe_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Linear Probe Accuracy'])
        for row in results:
            writer.writerow(row)
            
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
