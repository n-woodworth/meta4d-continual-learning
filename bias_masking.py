"""
Bias Masking Consistency - Dead Neurons Don't Get Biases

Problem: If all weights for a neuron are masked (dead neuron), its bias is still active.
Result: Meaningless bias adds noise to the network.

Solution: If row i in weight mask is all zeros, mask bias[i] too.

Logic:
    For each Linear layer:
        If weight_mask[row_i, :].sum() == 0:  # Dead neuron
            bias_mask[i] = 0  # Mask its bias

Result: Clean, consistent masking. Dead neurons are truly dead.
"""

import torch
import torch.nn as nn


def enforce_bias_consistency(model, masks):
    """
    Enforce bias masking consistency with weight masking.
    
    Rule: If a neuron has no incoming/outgoing weights, its bias should be masked.
    
    Args:
        model: Neural network
        masks: Dictionary of masks {param_name: mask_tensor}
    
    Implementation:
        For each Linear layer with weight W and bias b:
            For each output neuron i:
                If W[i, :].sum() == 0:  # All outgoing weights masked
                    b[i] = 0  # Mask the bias too
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            
            # Check if both weight and bias exist and are masked
            if weight_name in masks and bias_name in masks and module.bias is not None:
                weight_mask = masks[weight_name]
                bias_mask = masks[bias_name]
                
                # Check each output neuron (row in weight matrix)
                for row_idx in range(weight_mask.shape[0]):
                    # If entire row is masked (dead output neuron)
                    if weight_mask[row_idx, :].sum() == 0:
                        # Mask the corresponding bias
                        bias_mask[row_idx] = 0.0
                        # Also zero the bias value itself
                        module.bias.data[row_idx] = 0.0


def initialize_bias_masks(model, masks):
    """
    Initialize bias masks for all Linear layers.
    
    By default, all biases start active (mask=1).
    They'll be masked automatically if their neurons become dead.
    
    Args:
        model: Neural network
        masks: Dictionary to add bias masks to
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            bias_name = f"{name}.bias"
            
            # Initialize with all active
            if bias_name not in masks:
                masks[bias_name] = torch.ones_like(module.bias)


class BiasAwareMasking:
    """
    Mixin to add bias masking consistency to any masking system.
    
    Usage:
        class MyMasking(BiasAwareMasking, BlockHebbianMasking):
            pass
    
    Now enforce() will automatically handle bias consistency.
    """
    
    def enforce(self, model):
        """
        Enhanced enforce that handles bias consistency.
        
        1. Enforce weight masks (standard)
        2. Enforce bias masks
        3. Ensure consistency (dead neurons have masked biases)
        """
        # Standard weight masking
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
        
        # Enforce bias consistency
        enforce_bias_consistency(model, self.masks)
    
    def _initialize_masks(self, model):
        """
        Enhanced initialization that includes bias masks.
        
        Call this from __init__ after creating weight masks.
        """
        initialize_bias_masks(model, self.masks)


class ConsistentHebbianMasking(BiasAwareMasking):
    """
    Complete masking system with bias consistency.
    
    Features:
    - Block-wise weight masking
    - Automatic bias masking for dead neurons
    - Spawning/pruning neurons as complete units (weights + bias)
    """
    
    def __init__(self, model, block_size=16, dormant_ratio=0.7):
        # Initialize weight masks (simplified for demo)
        self.masks = {}
        self.block_size = block_size
        self.dormant_ratio = dormant_ratio
        self.recording = False
        
        # Create masks for all parameters
        for name, param in model.named_parameters():
            if param.dim() == 2:
                # Weight matrix - block masking
                mask = torch.ones_like(param)
                
                # Make some blocks dormant initially
                rows, cols = param.shape
                n_rows_b = rows // block_size
                n_cols_b = cols // block_size
                
                for r in range(n_rows_b):
                    for c in range(n_cols_b):
                        if torch.rand(1).item() < dormant_ratio:
                            r_start = r * block_size
                            r_end = min(r_start + block_size, rows)
                            c_start = c * block_size
                            c_end = min(c_start + block_size, cols)
                            mask[r_start:r_end, c_start:c_end] = 0.0
                
                self.masks[name] = mask
        
        # Initialize bias masks
        initialize_bias_masks(model, self.masks)
        
        # Initial enforcement
        self.enforce(model)
    
    def spawn_neuron(self, model, layer_name, neuron_idx):
        """
        Spawn a complete neuron (weights + bias).
        
        Args:
            model: Neural network
            layer_name: Name of layer (e.g., 'enc1')
            neuron_idx: Which output neuron to spawn
        """
        weight_name = f"{layer_name}.weight"
        bias_name = f"{layer_name}.bias"
        
        # Activate all incoming weights for this neuron
        if weight_name in self.masks:
            self.masks[weight_name][neuron_idx, :] = 1.0
        
        # Activate its bias
        if bias_name in self.masks:
            self.masks[bias_name][neuron_idx] = 1.0
        
        print(f"✓ Spawned neuron {neuron_idx} in {layer_name} (weights + bias)")
    
    def prune_neuron(self, model, layer_name, neuron_idx):
        """
        Prune a complete neuron (weights + bias).
        
        Args:
            model: Neural network
            layer_name: Name of layer
            neuron_idx: Which output neuron to prune
        """
        weight_name = f"{layer_name}.weight"
        bias_name = f"{layer_name}.bias"
        
        # Deactivate all weights for this neuron
        if weight_name in self.masks:
            self.masks[weight_name][neuron_idx, :] = 0.0
        
        # Deactivate its bias
        if bias_name in self.masks:
            self.masks[bias_name][neuron_idx] = 0.0
        
        print(f"✗ Pruned neuron {neuron_idx} in {layer_name} (weights + bias)")


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == '__main__':
    """
    Demonstrate bias masking consistency
    """
    from meta4d_continual_learning import WorkspaceNet, set_global_seed
    
    set_global_seed(42)
    device = torch.device("cpu")
    
    print("="*70)
    print("BIAS MASKING CONSISTENCY DEMO")
    print("="*70)
    
    # Create model
    model = WorkspaceNet().to(device)
    
    print("\nCreating masking system with bias consistency...")
    masker = ConsistentHebbianMasking(model, block_size=16, dormant_ratio=0.3)
    
    # Check initial state
    print("\nInitial State:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            
            if weight_name in masker.masks:
                weight_mask = masker.masks[weight_name]
                bias_mask = masker.masks.get(bias_name)
                
                # Count dead neurons
                dead_neurons = 0
                for row_idx in range(weight_mask.shape[0]):
                    if weight_mask[row_idx, :].sum() == 0:
                        dead_neurons += 1
                        
                        # Verify bias is masked
                        if bias_mask is not None:
                            assert bias_mask[row_idx] == 0, f"Bias {row_idx} should be masked!"
                
                total_neurons = weight_mask.shape[0]
                active_neurons = total_neurons - dead_neurons
                
                print(f"\n  {name}:")
                print(f"    Total neurons: {total_neurons}")
                print(f"    Active neurons: {active_neurons}")
                print(f"    Dead neurons: {dead_neurons}")
                
                if bias_mask is not None:
                    active_biases = bias_mask.sum().item()
                    print(f"    Active biases: {int(active_biases)}")
                    print(f"    ✓ Consistency: {active_biases == active_neurons}")
    
    # Demonstrate spawning/pruning
    print("\n" + "="*70)
    print("SPAWNING AND PRUNING NEURONS")
    print("="*70)
    
    # Get first layer  
    first_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            first_layer_name = name
            break
    
    if first_layer_name:
        weight_name = f"{first_layer_name}.weight"
        bias_name = f"{first_layer_name}.bias"
        
        # Find a dead neuron
        weight_mask = masker.masks[weight_name]
        dead_idx = None
        for row_idx in range(weight_mask.shape[0]):
            if weight_mask[row_idx, :].sum() == 0:
                dead_idx = row_idx
                break
        
        if dead_idx is not None:
            print(f"\nFound dead neuron {dead_idx} in {first_layer_name}")
            print(f"  Weight mask sum: {weight_mask[dead_idx, :].sum().item()}")
            print(f"  Bias mask value: {masker.masks[bias_name][dead_idx].item()}")
            
            # Spawn it
            print(f"\nSpawning neuron {dead_idx}...")
            masker.spawn_neuron(model, first_layer_name, dead_idx)
            masker.enforce(model)
            
            print(f"  Weight mask sum: {masker.masks[weight_name][dead_idx, :].sum().item()}")
            print(f"  Bias mask value: {masker.masks[bias_name][dead_idx].item()}")
            
            # Prune it again
            print(f"\nPruning neuron {dead_idx}...")
            masker.prune_neuron(model, first_layer_name, dead_idx)
            masker.enforce(model)
            
            print(f"  Weight mask sum: {masker.masks[weight_name][dead_idx, :].sum().item()}")
            print(f"  Bias mask value: {masker.masks[bias_name][dead_idx].item()}")
    
    print(f"\n{'RESULTS':^70}")
    print("="*70)
    
    print("\n✓ Bias masking consistency enforced")
    print("✓ Dead neurons have masked biases")
    print("✓ Spawning activates weights + bias together")
    print("✓ Pruning deactivates weights + bias together")
    print("✓ No meaningless biases polluting the network")
    
    print("\n💡 Key Rules:")
    print("  1. If weight_mask[i, :].sum() == 0 → bias_mask[i] = 0")
    print("  2. Spawn neuron → Activate weights AND bias")
    print("  3. Prune neuron → Deactivate weights AND bias")
    print("  4. enforce() automatically maintains consistency")
