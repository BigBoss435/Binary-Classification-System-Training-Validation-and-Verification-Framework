"""
Adversarial Robustness Testing Module
======================================
This module implements security testing using the Adversarial Robustness Toolbox (ART)
to evaluate model resilience against adversarial attacks.
"""

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt
import os

class ARTModelWrapper(nn.Module):
    """Wrapper to make model compatible with ART."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits = self.model(x)
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)
        return probs


def evaluate_adversarial_robustness(model, test_loader, device, results_dir, logger, 
                                    num_samples=500):
    """
    Comprehensive adversarial robustness evaluation.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        results_dir: Directory to save results
        logger: Logger instance
        num_samples: Number of samples to test (for efficiency)
        
    Returns:
        dict: Adversarial robustness metrics
    """
    
    adv_dir = os.path.join(results_dir, "adversarial_analysis")
    os.makedirs(adv_dir, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("ADVERSARIAL ROBUSTNESS EVALUATION")
    logger.info("="*60)
    
    # Prepare model for ART
    wrapped_model = ARTModelWrapper(model)
    wrapped_model.eval()
    
    # Create ART classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
    
    art_classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(0, 1),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )
    
    # Collect test samples
    X_test, y_test = _collect_test_samples(test_loader, num_samples, device)
    
    # Get clean predictions
    clean_predictions = art_classifier.predict(X_test)
    clean_accuracy = np.mean(np.argmax(clean_predictions, axis=1) == y_test)
    
    logger.info(f"Clean Accuracy: {clean_accuracy:.4f}")
    
    # Define attacks
    attacks = {
        'FGSM_eps0.03': FastGradientMethod(estimator=art_classifier, eps=0.03),
        'FGSM_eps0.1': FastGradientMethod(estimator=art_classifier, eps=0.1),
        'PGD_eps0.03': ProjectedGradientDescent(
            estimator=art_classifier, 
            eps=0.03, 
            eps_step=0.01, 
            max_iter=10
        )
    }
    
    results = {
        'clean_accuracy': float(clean_accuracy),
        'attacks': {}
    }
    
    # Evaluate each attack
    for attack_name, attack in attacks.items():
        logger.info(f"\nEvaluating {attack_name}...")
        
        attack_results = _evaluate_attack(
            attack, attack_name, art_classifier, X_test, y_test, 
            clean_predictions, adv_dir, logger
        )
        
        results['attacks'][attack_name] = attack_results
    
    # Generate summary report
    _generate_adversarial_report(results, adv_dir, logger)
    
    return results


def _collect_test_samples(test_loader, num_samples, device):
    """Collect a subset of test samples for efficiency."""
    
    X_list = []
    y_list = []
    
    for images, labels in test_loader:
        X_list.append(images.cpu().numpy())
        y_list.append(labels.cpu().numpy())
        
        if len(X_list) * images.shape[0] >= num_samples:
            break
    
    X = np.concatenate(X_list, axis=0)[:num_samples]
    y = np.concatenate(y_list, axis=0)[:num_samples]
    
    return X, y


def _evaluate_attack(attack, attack_name, classifier, X_test, y_test, 
                    clean_predictions, output_dir, logger):
    """Evaluate a specific adversarial attack."""
    
    # Generate adversarial examples
    logger.info(f"  Generating adversarial examples...")
    X_adv = attack.generate(x=X_test)
    
    # Get adversarial predictions
    adv_predictions = classifier.predict(X_adv)
    adv_accuracy = np.mean(np.argmax(adv_predictions, axis=1) == y_test)
    
    # Calculate perturbation statistics
    perturbation = X_adv - X_test
    l2_perturbation = np.mean(np.linalg.norm(perturbation.reshape(len(X_test), -1), axis=1))
    linf_perturbation = np.mean(np.max(np.abs(perturbation.reshape(len(X_test), -1)), axis=1))
    
    # Calculate attack success rate (fooling rate)
    clean_correct = np.argmax(clean_predictions, axis=1) == y_test
    adv_correct = np.argmax(adv_predictions, axis=1) == y_test
    attack_success_rate = np.mean(clean_correct & ~adv_correct)
    
    logger.info(f"  Adversarial Accuracy: {adv_accuracy:.4f}")
    logger.info(f"  Attack Success Rate: {attack_success_rate:.4f}")
    logger.info(f"  L2 Perturbation: {l2_perturbation:.6f}")
    logger.info(f"  L infinity Perturbation: {linf_perturbation:.6f}")
    
    # Visualize examples
    _visualize_adversarial_examples(
        X_test, X_adv, y_test, clean_predictions, adv_predictions,
        attack_name, output_dir
    )
    
    return {
        'adversarial_accuracy': float(adv_accuracy),
        'attack_success_rate': float(attack_success_rate),
        'l2_perturbation': float(l2_perturbation),
        'linf_perturbation': float(linf_perturbation),
        'robustness_score': float(adv_accuracy)
    }


def _visualize_adversarial_examples(X_clean, X_adv, y_true, pred_clean, pred_adv,
                                    attack_name, output_dir, num_examples=5):
    """Visualize adversarial examples."""
    
    # Find successful attacks
    clean_correct = np.argmax(pred_clean, axis=1) == y_true
    adv_incorrect = np.argmax(pred_adv, axis=1) != y_true
    successful_attacks = np.where(clean_correct & adv_incorrect)[0]
    
    if len(successful_attacks) == 0:
        return
    
    # Select random examples
    indices = np.random.choice(successful_attacks, 
                              min(num_examples, len(successful_attacks)), 
                              replace=False)
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original image
        img_clean = X_clean[idx].transpose(1, 2, 0)
        img_clean = (img_clean - img_clean.min()) / (img_clean.max() - img_clean.min())
        axes[i, 0].imshow(img_clean)
        axes[i, 0].set_title(f'Original\nPred: {np.argmax(pred_clean[idx])}\nConf: {pred_clean[idx].max():.3f}')
        axes[i, 0].axis('off')
        
        # Adversarial image
        img_adv = X_adv[idx].transpose(1, 2, 0)
        img_adv = (img_adv - img_adv.min()) / (img_adv.max() - img_adv.min())
        axes[i, 1].imshow(img_adv)
        axes[i, 1].set_title(f'Adversarial\nPred: {np.argmax(pred_adv[idx])}\nConf: {pred_adv[idx].max():.3f}')
        axes[i, 1].axis('off')
        
        # Perturbation
        perturbation = np.abs(X_adv[idx] - X_clean[idx]).transpose(1, 2, 0)
        perturbation = perturbation / perturbation.max()  # Normalize for visualization
        axes[i, 2].imshow(perturbation)
        axes[i, 2].set_title('Perturbation\n(amplified)')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Adversarial Examples: {attack_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'adversarial_examples_{attack_name}.png'), dpi=300)
    plt.close()


def _generate_adversarial_report(results, output_dir, logger):
    """Generate comprehensive adversarial robustness report."""
    
    report_path = os.path.join(output_dir, 'adversarial_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADVERSARIAL ROBUSTNESS ASSESSMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Clean Model Accuracy: {results['clean_accuracy']:.4f}\n\n")
        
        f.write("Attack Results:\n")
        f.write("-"*80 + "\n\n")
        
        for attack_name, attack_results in results['attacks'].items():
            f.write(f"{attack_name}:\n")
            f.write(f"  Adversarial Accuracy: {attack_results['adversarial_accuracy']:.4f}\n")
            f.write(f"  Attack Success Rate: {attack_results['attack_success_rate']:.4f}\n")
            f.write(f"  L2 Perturbation: {attack_results['l2_perturbation']:.6f}\n")
            f.write(f"  L infinity Perturbation: {attack_results['linf_perturbation']:.6f}\n")
            f.write(f"  Robustness Score: {attack_results['robustness_score']:.4f}\n")
            f.write("\n")
        
        # Overall assessment
        avg_robustness = np.mean([r['robustness_score'] for r in results['attacks'].values()])
        f.write("\nOverall Assessment:\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Robustness Score: {avg_robustness:.4f}\n")
        
        if avg_robustness > 0.8:
            f.write("Model shows STRONG robustness against adversarial attacks\n")
        elif avg_robustness > 0.6:
            f.write("Model shows MODERATE robustness against adversarial attacks\n")
        else:
            f.write("Model shows WEAK robustness - consider adversarial training\n")
    
    logger.info(f"Adversarial report saved to: {report_path}")
