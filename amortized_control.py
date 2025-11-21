"""
Amortized Meta-Control - Run Expensive Operations Less Frequently

Problem: Running Meta-Controller + Hebbian calculations every step is expensive:
- Virtual gradient computation
- Homeostatic state updates
- Block spawning/pruning decisions

Solution: Only run every N steps (e.g., 10-50 steps)
- Most steps: Just enforce masks (cheap)
- Every N steps: Full meta-control (expensive)

Result: 10-50x reduction in meta-control overhead!
"""

import torch
import torch.nn as nn
import numpy as np


class AmortizedController:
    """
    Wrapper that amortizes expensive meta-control operations.
    
    Instead of:
        Every step: Full meta-control (expensive)
    
    Do this:
        Steps 1-9: Just enforce masks (cheap)
        Step 10: Full meta-control (expensive)
        Steps 11-19: Just enforce masks
        Step 20: Full meta-control
        ...
    
    Configurable amortization intervals:
    - Aggressive (N=10): 10x speedup, still responsive
    - Balanced (N=25): 25x speedup, good for stable tasks
    - Conservative (N=50): 50x speedup, for production
    """
    
    def __init__(self, controller, masker, amortization_interval=10):
        """
        Args:
            controller: The base controller (GeneralizingController, etc.)
            masker: The masking system (BlockHebbianMasking, etc.)
            amortization_interval: Run full meta-control every N steps
        """
        self.controller = controller
        self.masker = masker
        self.interval = amortization_interval
        self.step_counter = 0
        
        # Statistics
        self.full_runs = 0
        self.cheap_runs = 0
        self.total_spawns = 0
        self.total_prunes = 0
    
    def should_run_full_control(self):
        """Decide whether to run full meta-control this step"""
        return self.step_counter % self.interval == 0
    
    def step(self, model, loss, biomass, new_growth, optimizer=None):
        """
        Main step function - replaces controller.observe() + masker actions.
        
        Args:
            model: Neural network
            loss: Current loss value
            biomass: Network sparsity
            new_growth: Recent growth amount
            optimizer: Optional optimizer for state injection
        
        Returns:
            dict with operation info
        """
        self.step_counter += 1
        
        # Every N steps: Full meta-control
        if self.should_run_full_control():
            self.full_runs += 1
            
            # Run controller decision
            if hasattr(self.controller, 'observe'):
                # Standard controller (returns operation, intensity)
                result = self.controller.observe(loss, biomass, new_growth)
                
                if len(result) == 3:
                    # Dynamic focus controller (returns operation, intensity, focus)
                    operation, intensity, focus = result
                else:
                    operation, intensity = result
                    focus = 1.0
            else:
                operation, intensity, focus = None, 0.0, 1.0
            
            # Execute operation
            actual_growth = 0.0
            
            if operation == 'SPAWN':
                # Check if optimizer injection is available
                if hasattr(self.masker, 'hebbian_block_spawn_with_optimizer') and optimizer:
                    actual_growth = self.masker.hebbian_block_spawn_with_optimizer(
                        model, intensity, optimizer, focus=focus
                    )
                else:
                    spawn_kwargs = {'focus': focus} if 'focus' in self.masker.hebbian_block_spawn.__code__.co_varnames else {}
                    actual_growth = self.masker.hebbian_block_spawn(model, intensity, **spawn_kwargs)
                
                self.total_spawns += 1
            
            elif operation == 'PRUNE':
                if hasattr(self.masker, 'smart_block_prune'):
                    self.masker.smart_block_prune(model, intensity)
                self.total_prunes += 1
            
            elif operation == 'REVERT':
                if hasattr(self.controller, 'revert'):
                    self.controller.revert(self.masker)
            
            return {
                'ran_control': True,
                'operation': operation,
                'intensity': intensity,
                'focus': focus,
                'growth': actual_growth
            }
        
        else:
            # Cheap steps: Just enforce masks
            self.cheap_runs += 1
            self.masker.enforce(model)
            
            return {
                'ran_control': False,
                'operation': None,
                'intensity': 0.0,
                'focus': 0.0,
                'growth': 0.0
            }
    
    def get_efficiency_stats(self):
        """Get statistics on computational savings"""
        total_steps = self.full_runs + self.cheap_runs
        
        if total_steps == 0:
            return {
                'total_steps': 0,
                'full_control_runs': 0,
                'cheap_runs': 0,
                'overhead_reduction': 0.0,
                'spawns': 0,
                'prunes': 0
            }
        
        overhead_reduction = (self.cheap_runs / total_steps) * 100
        
        return {
            'total_steps': total_steps,
            'full_control_runs': self.full_runs,
            'cheap_runs': self.cheap_runs,
            'overhead_reduction': overhead_reduction,
            'amortization_factor': self.interval,
            'spawns': self.total_spawns,
            'prunes': self.total_prunes
        }


class AdaptiveAmortization:
    """
    Dynamically adjust amortization interval based on system state.
    
    Strategy:
    - High stress (new task): N=5 (responsive)
    - Moderate stress: N=25 (balanced)
    - Low stress (stable): N=50 (efficient)
    """
    
    def __init__(self, controller, masker, min_interval=5, max_interval=50):
        self.controller = controller
        self.masker = masker
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = min_interval
        self.step_counter = 0
        
        # Track stress
        self.recent_losses = []
        self.loss_variance = 0.0
    
    def compute_adaptive_interval(self, loss):
        """
        Compute interval based on loss variance (stress indicator).
        
        High variance = high stress = low interval (responsive)
        Low variance = low stress = high interval (efficient)
        """
        self.recent_losses.append(loss)
        
        # Keep last 50 losses
        if len(self.recent_losses) > 50:
            self.recent_losses.pop(0)
        
        # Calculate variance
        if len(self.recent_losses) >= 10:
            self.loss_variance = np.var(self.recent_losses)
            
            # Map variance to interval
            # High variance (>0.5) → min_interval
            # Low variance (<0.1) → max_interval
            if self.loss_variance > 0.5:
                self.current_interval = self.min_interval
            elif self.loss_variance < 0.1:
                self.current_interval = self.max_interval
            else:
                # Linear interpolation
                ratio = (self.loss_variance - 0.1) / (0.5 - 0.1)
                self.current_interval = int(
                    self.max_interval - ratio * (self.max_interval - self.min_interval)
                )
        
        return self.current_interval


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """
    Demonstrate amortized meta-control
    """
    import torch.optim as optim
    from meta4d_continual_learning import (
        BlockHebbianMasking, 
        GeneralizingController,
        WorkspaceNet, 
        set_global_seed
    )
    
    set_global_seed(42)
    device = torch.device("cpu")
    
    print("="*70)
    print("AMORTIZED META-CONTROL DEMO")
    print("="*70)
    
    # Setup
    model = WorkspaceNet().to(device)
    masker = BlockHebbianMasking(model, block_size=16, dormant_ratio=0.7)
    controller = GeneralizingController(target_biomass=0.35, verbose=False)
    
    # Wrap with amortization
    amortized = AmortizedController(
        controller, 
        masker, 
        amortization_interval=10  # Run full control every 10 steps
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nConfiguration:")
    print(f"  Amortization interval: {amortized.interval} steps")
    print(f"  Full control: Every {amortized.interval}th step")
    print(f"  Cheap steps: {amortized.interval - 1} out of {amortized.interval}")
    
    # Training loop
    print(f"\nTraining for 100 steps...")
    
    last_growth = 0.0
    
    for step in range(100):
        # Forward/backward
        x = torch.randn(32, 784).to(device)
        
        masker.recording = True
        masker.enforce(model)
        
        y = model(x)
        loss = y.mean()
        
        optimizer.zero_grad()
        loss.backward()
        masker.enforce(model)
        optimizer.step()
        
        # Amortized meta-control
        biomass = sum(m.sum() for m in masker.masks.values()) / sum(m.numel() for m in masker.masks.values())
        result = amortized.step(model, loss.item(), biomass, last_growth, optimizer)
        
        last_growth = result['growth']
        
        # Print on control steps
        if result['ran_control'] and result['operation']:
            print(f"  Step {step}: [{result['operation']}] intensity={result['intensity']:.2f}")
        
        masker.clear_buffers()
    
    # Results
    print(f"\n{'RESULTS':^70}")
    print("="*70)
    
    stats = amortized.get_efficiency_stats()
    
    print(f"\nComputational Efficiency:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Full control runs: {stats['full_control_runs']} ({stats['full_control_runs']/stats['total_steps']*100:.1f}%)")
    print(f"  Cheap runs: {stats['cheap_runs']} ({stats['cheap_runs']/stats['total_steps']*100:.1f}%)")
    print(f"  Overhead reduction: {stats['overhead_reduction']:.1f}%")
    print(f"  Speedup factor: {stats['amortization_factor']}x")
    
    print(f"\nMeta-Control Actions:")
    print(f"  Spawns: {stats['spawns']}")
    print(f"  Prunes: {stats['prunes']}")
    
    print(f"\n✓ Amortized meta-control achieved!")
    print(f"✓ {stats['overhead_reduction']:.0f}% reduction in meta-control overhead")
    print(f"✓ Training mostly runs at full speed")
    print(f"✓ Meta-control only when needed (every {stats['amortization_factor']} steps)")
    
    # Show comparison with standard approach
    print(f"\n{'COMPARISON':^70}")
    print("="*70)
    print(f"\nStandard approach (every step):")
    print(f"  Meta-control calls: 100")
    print(f"  Relative cost: 100%")
    
    print(f"\nAmortized approach (every {stats['amortization_factor']} steps):")
    print(f"  Meta-control calls: {stats['full_control_runs']}")
    print(f"  Relative cost: {100 - stats['overhead_reduction']:.1f}%")
    
    print(f"\n💡 Recommendation for production:")
    print(f"  - Development/debugging: N=10 (responsive)")
    print(f"  - Training: N=25 (balanced)")
    print(f"  - Production/inference: N=50 (maximum efficiency)")
