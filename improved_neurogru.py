
# Try to import cupy, fall back to numpy if not available
try:
    import cupy as cp
    use_cupy = True
except ImportError:
    import numpy as cp
    use_cupy = False
    print("CuPy not available, using NumPy instead.")
import numpy as np
from collections import deque
import scipy.stats as stats
import math
import gc

class ImprovedNeuroGRU:
    """
    Biologically-Enhanced Neuromorphic GRU Implementation
    
    Key Improvements:
    1. Task-driven neurogenesis and competitive survival mechanisms
    2. Neuronal autonomy with local learning objectives and specialization
    3. Advanced Hebbian learning with true associative learning and structural plasticity
    4. Hierarchical processing and dynamic neuronal ensembles
    
    This model more closely mimics biological neural principles to improve
    trading performance, adaptability, and efficiency.
    """
    
    def __init__(self, input_dim, hidden_dim, max_neurons=256, min_neurons=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.structural_plasticity_enabled = False
        
        # --- Neuron differentiation parameters ---
        self._initialize_neuron_types()
        
        # --- Multi-scale weight matrices ---
        self._initialize_weight_matrices()
        
        # --- Neuromodulatory systems ---
        self._initialize_neuromodulation()
        
        # --- Activity tracking ---
        self._initialize_activity_tracking()
        
        # --- Energy and health systems ---
        self._initialize_energy_systems()
        
        # --- Plasticity parameters ---
        self._initialize_plasticity()
        
        # --- Structural plasticity ---
        self._initialize_structural_plasticity()
        
        # --- Performance metrics ---
        self._initialize_performance_tracking()
        
        # --- NEW: Neuronal autonomy ---
        self._initialize_neuronal_autonomy()
        
        # --- NEW: Hierarchical organization ---
        self._initialize_hierarchical_organization()
        
        # --- NEW: Ensemble formation ---
        self._initialize_ensemble_formation()
        
        # --- Metadata consistency check ---
        self._validate_dimensions()
    
    def _initialize_neuron_types(self):
        """Initialize diverse neuron types mimicking cortical diversity"""
        # --- Dale's principle - E/I balance ---
        # ~75-80% excitatory, ~20-25% inhibitory based on cortical ratios
        self.neuron_type = cp.ones(self.hidden_dim)  # 1 for excitatory, -1 for inhibitory
        inhibitory_mask = cp.random.choice(self.hidden_dim, size=int(self.hidden_dim * 0.2), replace=False)
        self.neuron_type[inhibitory_mask] = -1
        
        # --- Heterogeneous activation functions ---
        # 0: tanh-like, 1: sigmoid-like, 2: ReLU-like, 3: adaptive (NEW)
        self.activation_types = cp.zeros(self.hidden_dim, dtype=np.int32)
        
        # Distribute neurons across activation types
        sigmoid_mask = cp.random.choice(self.hidden_dim, size=int(self.hidden_dim * 0.2), replace=False)
        # Handle both CuPy and NumPy cases
        if use_cupy:
            relu_mask_candidates = np.setdiff1d(cp.asnumpy(cp.arange(self.hidden_dim)), cp.asnumpy(sigmoid_mask))
        else:
            relu_mask_candidates = np.setdiff1d(cp.arange(self.hidden_dim), sigmoid_mask)
        relu_mask = cp.asarray(np.random.choice(relu_mask_candidates, size=int(self.hidden_dim * 0.1), replace=False))
        
        # NEW: Add adaptive neurons (type 3)
        # Handle both CuPy and NumPy cases
        if use_cupy:
            adaptive_mask_candidates = np.setdiff1d(cp.asnumpy(cp.arange(self.hidden_dim)), 
                                                  np.union1d(cp.asnumpy(sigmoid_mask), cp.asnumpy(relu_mask)))
        else:
            adaptive_mask_candidates = np.setdiff1d(cp.arange(self.hidden_dim), 
                                                  np.union1d(sigmoid_mask, relu_mask))
        adaptive_mask = cp.asarray(np.random.choice(adaptive_mask_candidates, 
                                                  size=int(self.hidden_dim * 0.1), replace=False))
        
        self.activation_types[sigmoid_mask] = 1
        self.activation_types[relu_mask] = 2
        self.activation_types[adaptive_mask] = 3
        
        # Adaptive activation parameters
        self.activation_gain = cp.ones(self.hidden_dim)
        self.activation_threshold = cp.zeros(self.hidden_dim)
        
        # NEW: Task-specific activation parameters
        self.activation_params = cp.zeros((self.hidden_dim, 3))  # 3 learnable parameters per neuron
        
        # Timescale diversity - neurons operate at different speeds
        # Based on biological diversity of neural integration timescales
        self.tau = 10.0 + cp.random.randn(self.hidden_dim) * 5.0
        self.tau = cp.maximum(self.tau, 1.0)  # Ensure positive timescales
        
        # NEW: Neuron specialization - what patterns each neuron responds to
        self.specialization = cp.zeros((self.hidden_dim, self.input_dim))
        # Initialize with random preferences (will be refined through learning)
        self.specialization = cp.random.randn(self.hidden_dim, self.input_dim) * 0.1
    
    def _initialize_weight_matrices(self):
        """Initialize multi-scale weight matrices with balanced dynamics"""
        scale = 0.1 / cp.sqrt(self.input_dim)  # He initialization
        
        # Standard GRU weights
        self.Wz = cp.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Uz = self._create_balanced_recurrent_weights(self.hidden_dim)
        self.bz = cp.zeros((self.hidden_dim,))
        
        self.Wr = cp.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Ur = self._create_balanced_recurrent_weights(self.hidden_dim)
        self.br = cp.zeros((self.hidden_dim,))
        
        self.Wh = cp.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Uh = self._create_balanced_recurrent_weights(self.hidden_dim)
        self.bh = cp.zeros((self.hidden_dim,))
        
        # Multi-timescale weight components
        # Fast weights - changes on short timescales
        self.fast_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
        
        # Slow weights - changes on long timescales, captures recurring patterns
        self.slow_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
        
        # NEW: Structural connectivity matrix (binary) - represents physical connections
        self.connectivity = cp.ones((self.hidden_dim, self.hidden_dim), dtype=bool)
        
        # State vectors
        self.h = cp.zeros((self.hidden_dim,))
        self.h_prev = cp.zeros((self.hidden_dim,))
        self.dh = cp.zeros((self.hidden_dim,))  # Rate of change for criticality assessment
        
        # Spike-Timing Dependent Plasticity (STDP) components
        self.pre_trace = cp.zeros((self.hidden_dim,))
        self.post_trace = cp.zeros((self.hidden_dim,))
        
        # Adaptive threshold for homeostatic regulation
        self.firing_threshold = cp.ones((self.hidden_dim,)) * 0.1
        
        # NEW: Dendritic compartments (mini-computers within neurons)
        self.num_dendrites = 3  # Each neuron has multiple dendritic compartments
        self.dendritic_weights = cp.random.randn(self.hidden_dim, self.num_dendrites, self.input_dim) * 0.1
        self.dendritic_activations = cp.zeros((self.hidden_dim, self.num_dendrites))
        
        # Store current input for dendritic weight updates
        self.current_input = None
    
    def _create_balanced_recurrent_weights(self, size):
        """
        Create a balanced recurrent weight matrix following Dale's principle
        with criticality optimization (spectral radius near 1)
        """
        W = cp.random.randn(size, size) * 0.1 / cp.sqrt(size)
        
        # Apply Dale's principle - neurons maintain excitatory/inhibitory nature
        for i in range(size):
            if self.neuron_type[i] < 0:  # Inhibitory neuron
                W[i, :] = -cp.abs(W[i, :])  # Always inhibitory output
            else:  # Excitatory neuron
                W[i, :] = cp.abs(W[i, :])   # Always excitatory output
        
        # Scale to bring spectral radius near 1 for criticality
        try:
            eigenvalues = cp.linalg.eigvals(W)
            spectral_radius = cp.max(cp.abs(eigenvalues))
            if spectral_radius > 0:
                W *= 0.95 / spectral_radius  # Target slightly below 1 for stability
        except:
            # If eigenvalue computation fails, use a conservative scaling
            W *= 0.5 / cp.mean(cp.abs(W))
            
        return W
    
    def _initialize_neuromodulation(self):
        """Initialize neuromodulatory systems with adaptive regulation"""
        # Primary neuromodulators with dynamic ranges
        self.modulation = {
            'glutamate': 1.0,       # Fast excitation
            'gaba': 0.3,            # Fast inhibition
            'dopamine': 0.5,        # Reward/reinforcement learning
            'acetylcholine': 0.7,   # Attention/learning rate
            'norepinephrine': 0.5,  # Exploration/arousal
            'serotonin': 0.8,       # Mood/temporal discounting
        }
        
        # Target ranges for each modulator
        self.modulation_targets = {
            'glutamate': (0.7, 1.0),
            'gaba': (0.2, 0.5),
            'dopamine': (0.3, 0.9),
            'acetylcholine': (0.5, 0.9),
            'norepinephrine': (0.2, 0.8),
            'serotonin': (0.6, 0.9)
        }
        
        # Adaptation rates for neuromodulators
        self.modulation_adaptation_rate = 0.01
        
        # Neuromodulator release triggered by activity patterns
        self.activity_triggered_modulation = {
            'dopamine_release': 0.0,    # Reward signal
            'ach_release': 0.0,         # Novelty signal
            'ne_release': 0.0           # Prediction error signal
        }
        
        # NEW: Neuron-specific neuromodulator sensitivity
        self.neuromod_sensitivity = cp.ones((self.hidden_dim, len(self.modulation)))
        # Add some variability to sensitivity
        self.neuromod_sensitivity += cp.random.randn(self.hidden_dim, len(self.modulation)) * 0.2
        self.neuromod_sensitivity = cp.clip(self.neuromod_sensitivity, 0.5, 1.5)
    
    def _initialize_activity_tracking(self):
        """Initialize multi-timescale activity tracking"""
        # Multiple timescales for different mechanisms
        self.time_constants = [0.99, 0.9, 0.5]  # Fast, medium, slow
        
        self.activity_history = {
            'fast': cp.zeros((self.hidden_dim,)),   # For intrinsic plasticity
            'medium': cp.zeros((self.hidden_dim,)), # For structural plasticity
            'slow': cp.zeros((self.hidden_dim,)),   # For metaplasticity
        }
        
        # Eligibility traces for different plasticity mechanisms
        self.eligibility_traces = cp.zeros((self.hidden_dim, self.hidden_dim))
        self.e_trace_decay = 0.9
        
        # Recent activation patterns for stability analysis
        self.recent_activations = deque(maxlen=100)
        
        # STDP traces
        self.pre_synaptic_trace = cp.zeros((self.hidden_dim,))
        self.post_synaptic_trace = cp.zeros((self.hidden_dim,))
        
        # Information-theoretic metrics
        self.mutual_information = cp.zeros((self.hidden_dim, self.hidden_dim))
        self.information_content = cp.zeros((self.hidden_dim,))
        
        # NEW: Prediction error tracking for each neuron
        self.prediction_errors = cp.zeros((self.hidden_dim,))
        self.prediction_history = deque(maxlen=100)  # Store recent predictions
        
        # NEW: Causal association tracking
        self.causal_matrix = cp.zeros((self.hidden_dim, self.hidden_dim))
        self.temporal_memory = deque(maxlen=10)  # Store recent activations for temporal associations
    
    def _initialize_energy_systems(self):
        """Initialize energy and resource allocation systems"""
        # Energy resources with recovery dynamics
        self.energy = cp.ones((self.hidden_dim,))
        self.energy_recovery_rate = 0.01
        self.energy_consumption_rate = 0.1
        
        # Neuron health metrics
        self.health = cp.ones((self.hidden_dim,))
        self.recovery_rate = 0.005
        
        # Metabolic efficiency tracking
        self.metabolic_cost = 0.0
        self.information_gain = 0.0
        self.efficiency_ratio = 1.0  # Information gain / metabolic cost
        
        # NEW: Priority-based energy allocation
        self.energy_priority = cp.ones((self.hidden_dim,))  # Higher values get more energy
    
    def _initialize_plasticity(self):
        """Initialize multi-scale plasticity mechanisms"""
        # Metaplasticity (plasticity of plasticity)
        self.plasticity_rates = cp.ones((self.hidden_dim,)) * 0.001
        self.metaplasticity_rate = 0.0001
        
        # Homeostatic synaptic scaling parameters
        self.target_activity = 0.1
        self.synaptic_scaling_rate = 0.01
        
        # Consolidation parameters - transfer from fast to slow weights
        self.consolidation_threshold = 0.5
        self.consolidation_rate = 0.001
        
        # STDP parameters
        self.stdp_params = {
            'A_plus': 0.01,    # LTP (Long-Term Potentiation) amplitude
            'A_minus': 0.0105, # LTD (Long-Term Depression) amplitude (slightly larger for stability)
            'tau_plus': 20,    # LTP time constant
            'tau_minus': 20    # LTD time constant
        }
        
        # NEW: Multi-factor plasticity parameters
        self.plasticity_factors = {
            'reward': 1.0,      # Dopamine-modulated reward signal
            'surprise': 1.0,    # Norepinephrine-modulated surprise signal
            'attention': 1.0,   # Acetylcholine-modulated attention signal
            'novelty': 1.0      # Novelty detection signal
        }
        
        # NEW: Structural plasticity parameters for synapse formation/elimination
        self.synapse_formation_threshold = 0.7  # Correlation threshold for new synapse formation
        self.synapse_elimination_threshold = 0.1  # Activity threshold below which synapses are pruned
    
    def _initialize_structural_plasticity(self):
        """Initialize structural plasticity with criticality regulation"""
        # Tracking counters
        self.step_counter = 0
        self.total_grown = 0
        self.total_pruned = 0
        self.total_replaced = 0
        
        # Structural plasticity parameters
        self.structural_plasticity_period = 50
        self.enable_growth = True
        self.enable_pruning = True
        
        # Neuron utility tracking
        self.neuron_utility = cp.ones((self.hidden_dim,)) * 0.1
        
        # Growth regulation parameters
        self.last_growth_step = 0
        self.growth_cooldown = 200
        self.max_growth_per_phase = 8
        self.growth_rate_decay = 0.9
        self.current_growth_rate = 1.0
        
        # Criticality regulation
        self.target_spectral_radius = 0.95
        self.current_growth_threshold = 0.75
        self.growth_threshold_increase = 0.05
        self.criticality_metrics = {
            'spectral_radius': 0.0,
            'eigenvalue_gap': 0.0,
            'participation_ratio': 0.0,
            'edge_of_chaos_distance': 1.0
        }
        
        # Information-theoretic connectivity optimization
        self.connectivity_complexity = cp.zeros((self.hidden_dim,))
        self.subnetwork_modularity = cp.zeros((self.hidden_dim,))
        
        # Targeted growth location tracking
        self.growth_potential = cp.zeros((self.hidden_dim,))
        
        # NEW: Task-driven neurogenesis parameters
        self.error_gradient = cp.zeros((self.hidden_dim,))  # Gradient of error w.r.t. each neuron
        self.error_history = deque(maxlen=100)  # Track recent errors for neurogenesis decisions
        
        # NEW: Competitive survival parameters
        self.survival_score = cp.ones((self.hidden_dim,))  # Higher values = higher survival probability
        self.competition_strength = 0.1  # How strongly neurons compete with each other
        
        # NEW: Developmental staging
        self.developmental_stage = 'critical_period'  # 'critical_period', 'refinement', 'mature'
        self.critical_period_end = 1000  # Steps until critical period ends
        self.maturation_end = 5000  # Steps until full maturation
    
    def _initialize_performance_tracking(self):
        """Initialize performance metrics tracking"""
        # Recent performance history
        self.recent_loss_history = deque(maxlen=100)
        self.recent_accuracy_history = deque(maxlen=100)
        
        # Performance gradient tracking
        self.performance_gradient = 0.0
        self.performance_acceleration = 0.0
        
        # Learning phase identification
        self.learning_phases = {
            'exploration': False,
            'exploitation': False,
            'refinement': False,
            'convergence': False
        }
        
        # Sliding window statistics
        self.sliding_window_size = 20
        self.sliding_stats = {
            'loss_mean': deque(maxlen=self.sliding_window_size),
            'loss_std': deque(maxlen=self.sliding_window_size),
            'accuracy_mean': deque(maxlen=self.sliding_window_size),
            'accuracy_std': deque(maxlen=self.sliding_window_size)
        }
        
        # NEW: Class-specific performance tracking
        self.class_performance = {
            'buy': deque(maxlen=50),   # Track accuracy on buy signals
            'sell': deque(maxlen=50),  # Track accuracy on sell signals
            'hold': deque(maxlen=50)   # Track accuracy on hold signals
        }
        
        # NEW: Misclassification tracking
        self.misclassification_counts = cp.zeros((3, 3))  # Confusion matrix-like structure
    
    def _initialize_neuronal_autonomy(self):
        """NEW: Initialize parameters for neuronal autonomy"""
        # Local learning objectives for each neuron
        self.local_objectives = cp.zeros((self.hidden_dim,))
        
        # Intrinsic motivation (curiosity) for each neuron
        self.intrinsic_motivation = cp.ones((self.hidden_dim,)) * 0.5
        
        # Neuron-specific learning rates
        self.neuron_learning_rates = cp.ones((self.hidden_dim,)) * 0.01
        
        # Predictive coding parameters
        self.prediction_targets = cp.zeros((self.hidden_dim,))
        self.prediction_weights = cp.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
        
        # Self-organizing map parameters
        self.som_neighborhood_size = 5.0  # Initial neighborhood size
        self.som_learning_rate = 0.1      # Initial SOM learning rate
        
        # Lateral inhibition strengths
        self.lateral_inhibition = cp.zeros((self.hidden_dim, self.hidden_dim))
        
        # Initialize lateral inhibition with distance-based falloff
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                if i != j:
                    # Stronger inhibition between nearby neurons (assuming 1D topology)
                    distance = min(abs(i - j), self.hidden_dim - abs(i - j))
                    if distance < self.hidden_dim // 4:  # Local inhibition
                        self.lateral_inhibition[i, j] = 0.2 * np.exp(-distance / 10)
    
    def _initialize_hierarchical_organization(self):
        """NEW: Initialize hierarchical processing organization"""
        # Define hierarchical layers (neurons are organized into layers)
        self.num_layers = 3  # Number of hierarchical layers
        neurons_per_layer = self.hidden_dim // self.num_layers
        
        # Assign neurons to layers
        self.neuron_layer = cp.zeros(self.hidden_dim, dtype=np.int32)
        for i in range(self.num_layers):
            start_idx = i * neurons_per_layer
            end_idx = (i + 1) * neurons_per_layer if i < self.num_layers - 1 else self.hidden_dim
            self.neuron_layer[start_idx:end_idx] = i
        
        # Different timescales for different layers
        # Lower layers (0) process faster, higher layers process slower
        for i in range(self.hidden_dim):
            layer = self.neuron_layer[i]
            # Adjust tau based on layer (higher layers = slower timescales)
            self.tau[i] *= (1.0 + 0.5 * layer)
        
        # Layer-specific neuromodulation
        self.layer_modulation = cp.ones((self.num_layers, len(self.modulation)))
        
        # Feedback connections (top-down)
        self.feedback_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
        
        # Initialize feedback from higher to lower layers
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                if self.neuron_layer[i] > self.neuron_layer[j]:  # Higher to lower
                    self.feedback_weights[i, j] = cp.random.randn() * 0.05
    
    def _initialize_ensemble_formation(self):
        """NEW: Initialize dynamic ensemble formation mechanisms"""
        # Ensemble membership tracking
        self.max_ensembles = 10
        self.ensemble_membership = cp.zeros((self.hidden_dim, self.max_ensembles))
        
        # Ensemble activity tracking
        self.ensemble_activity = cp.zeros(self.max_ensembles)
        
        # Ensemble specialization (what each ensemble responds to)
        self.ensemble_specialization = cp.zeros((self.max_ensembles, 3))  # 3 classes (buy, sell, hold)
        
        # Synchronization parameters
        self.phase = cp.random.uniform(0, 2*np.pi, size=self.hidden_dim)  # Oscillatory phase
        self.frequency = cp.ones(self.hidden_dim) * 0.1  # Base oscillation frequency
        
        # Coherence tracking between neurons
        self.coherence = cp.zeros((self.hidden_dim, self.hidden_dim))
        
        # Assembly vectors (prototype patterns for each ensemble)
        self.assembly_vectors = cp.zeros((self.max_ensembles, self.hidden_dim))
    
    def _validate_dimensions(self):
        """Verify dimensional consistency across all neuron metadata arrays"""
        try:
            inconsistent = False
            # Principal dimension to check against
            dim = self.hidden_dim
            
            # Check all neuron-specific arrays
            if self.neuron_type.shape[0] != dim:
                print(f"Dimension mismatch: neuron_type has shape {self.neuron_type.shape[0]}, should be {dim}")
                self.neuron_type = self._resize_array(self.neuron_type, dim)
                inconsistent = True
                
            if self.activation_types.shape[0] != dim:
                print(f"Dimension mismatch: activation_types has shape {self.activation_types.shape[0]}, should be {dim}")
                self.activation_types = self._resize_array(self.activation_types, dim, dtype=np.int32)
                inconsistent = True
            
            if self.activation_gain.shape[0] != dim:
                print(f"Dimension mismatch: activation_gain has shape {self.activation_gain.shape[0]}, should be {dim}")
                self.activation_gain = self._resize_array(self.activation_gain, dim)
                inconsistent = True
                
            if self.activation_threshold.shape[0] != dim:
                print(f"Dimension mismatch: activation_threshold has shape {self.activation_threshold.shape[0]}, should be {dim}")
                self.activation_threshold = self._resize_array(self.activation_threshold, dim)
                inconsistent = True
            
            # Check weight matrices
            for attr in ['Wz', 'Wr', 'Wh']:
                w = getattr(self, attr)
                if w.shape[0] != dim:
                    print(f"Dimension mismatch: {attr} has shape {w.shape}, should be ({dim}, {self.input_dim})")
                    setattr(self, attr, cp.random.randn(dim, self.input_dim) * 0.1)
                    inconsistent = True
            
            for attr in ['Uz', 'Ur', 'Uh']:
                w = getattr(self, attr)
                if w.shape[0] != dim or w.shape[1] != dim:
                    print(f"Dimension mismatch: {attr} has shape {w.shape}, should be ({dim}, {dim})")
                    setattr(self, attr, cp.random.randn(dim, dim) * 0.1)
                    inconsistent = True
            
            # Check bias vectors and state vectors
            for attr in ['bz', 'br', 'bh', 'h', 'h_prev', 'dh']:
                v = getattr(self, attr)
                if v.shape[0] != dim:
                    print(f"Dimension mismatch: {attr} has shape {v.shape[0]}, should be {dim}")
                    setattr(self, attr, cp.zeros(dim))
                    inconsistent = True
                    
            # Check activity history
            for key in self.activity_history:
                if self.activity_history[key].shape[0] != dim:
                    print(f"Dimension mismatch: activity_history[{key}] has shape {self.activity_history[key].shape[0]}, should be {dim}")
                    self.activity_history[key] = cp.zeros(dim)
                    inconsistent = True
            
            # NEW: Check neuronal autonomy arrays
            for attr in ['local_objectives', 'intrinsic_motivation', 'neuron_learning_rates']:
                v = getattr(self, attr)
                if v.shape[0] != dim:
                    print(f"Dimension mismatch: {attr} has shape {v.shape[0]}, should be {dim}")
                    setattr(self, attr, cp.ones(dim) * 0.5)  # Default initialization
                    inconsistent = True
            
            # NEW: Check hierarchical organization
            if self.neuron_layer.shape[0] != dim:
                print(f"Dimension mismatch: neuron_layer has shape {self.neuron_layer.shape[0]}, should be {dim}")
                self.neuron_layer = cp.zeros(dim, dtype=np.int32)
                inconsistent = True
            
            # NEW: Check ensemble formation
            if self.ensemble_membership.shape[0] != dim:
                print(f"Dimension mismatch: ensemble_membership has shape {self.ensemble_membership.shape[0]}, should be {dim}")
                self.ensemble_membership = cp.zeros((dim, self.max_ensembles))
                inconsistent = True
            
            if inconsistent:
                print("Fixed dimensional inconsistencies in neuronal metadata")
            
            return not inconsistent
            
        except Exception as e:
            print(f"Error during dimension validation: {e}")
            return False
    
    def _resize_array(self, arr, new_size, dtype=None):
        """Resize an array to the new size, preserving values where possible"""
        if dtype is None:
            new_arr = cp.zeros(new_size, dtype=arr.dtype)
        else:
            new_arr = cp.zeros(new_size, dtype=dtype)
            
        # Copy values from old array, up to min size
        min_size = min(arr.shape[0], new_size)
        new_arr[:min_size] = arr[:min_size]
        
        return new_arr
    
    def adaptive_activation(self, x, neuron_indices=None):
        """
        Apply heterogeneous activation functions based on neuron type
        
        Parameters:
        x: Input tensor
        neuron_indices: Optional indices to apply activation to specific neurons
        
        Returns:
        Activated outputs
        """
        # Ensure activation_types has consistent dimensions
        if self.activation_types.shape[0] != self.hidden_dim:
            print(f"Fixing activation_types dimension mismatch: {self.activation_types.shape[0]} vs {self.hidden_dim}")
            # Re-initialize activation types
            old_types = self.activation_types
            self.activation_types = cp.zeros(self.hidden_dim, dtype=np.int32)
            
            # Copy old values where possible
            min_size = min(old_types.shape[0], self.hidden_dim)
            self.activation_types[:min_size] = old_types[:min_size]
            
            # Assign random types to new neurons
            if self.hidden_dim > old_types.shape[0]:
                new_neurons = self.hidden_dim - old_types.shape[0]
                # Distribute new neurons: 60% tanh, 20% sigmoid, 10% ReLU, 10% adaptive
                remaining = cp.arange(old_types.shape[0], self.hidden_dim)
                n_sigmoid = int(new_neurons * 0.2)
                n_relu = int(new_neurons * 0.1)
                n_adaptive = int(new_neurons * 0.1)
                
                sigmoid_mask = cp.random.choice(remaining, size=n_sigmoid, replace=False)
                remaining_after_sigmoid = np.setdiff1d(cp.asnumpy(remaining), cp.asnumpy(sigmoid_mask))
                
                relu_mask = cp.asarray(np.random.choice(remaining_after_sigmoid, size=n_relu, replace=False))
                remaining_after_relu = np.setdiff1d(cp.asnumpy(remaining_after_sigmoid), cp.asnumpy(relu_mask))
                
                adaptive_mask = cp.asarray(np.random.choice(remaining_after_relu, size=n_adaptive, replace=False))
                
                self.activation_types[sigmoid_mask] = 1
                self.activation_types[relu_mask] = 2
                self.activation_types[adaptive_mask] = 3
        
        # Ensure x has the correct shape
        if len(x.shape) == 1:
            x = x.reshape(-1)
        elif len(x.shape) > 1:
            # Handle multi-dimensional input by flattening
            x = x.flatten()
        
        # Initialize output array with input (will be overwritten)
        y = x.copy()
        
        if neuron_indices is None:
            indices = cp.arange(len(x))
        else:
            indices = neuron_indices
        
        # Validate indices before proceeding
        if indices.size > 0 and indices.max() >= self.hidden_dim:
            print(f"Warning: Indices out of bounds - max index {indices.max()} exceeds hidden_dim {self.hidden_dim}")
            indices = indices[indices < self.hidden_dim]
            
        if indices.size == 0:
            return y
            
        # Apply different activation functions based on neuron type
        for i in range(4):  # For each activation type (including new adaptive type)
            mask = self.activation_types[indices] == i
            if not cp.any(mask):
                continue
                
            subset_idx = indices[mask]
            
            if i == 0:  # tanh-like
                # Adaptive tanh with gain and threshold
                y[mask] = cp.tanh((x[mask] - self.activation_threshold[subset_idx]) 
                                * self.activation_gain[subset_idx])
            elif i == 1:  # sigmoid-like
                # Adaptive sigmoid with gain and threshold
                y[mask] = 1.0 / (1.0 + cp.exp(-(x[mask] - self.activation_threshold[subset_idx]) 
                                           * self.activation_gain[subset_idx]))
            elif i == 2:  # ReLU-like with soft saturation
                # Adaptive ReLU with leak and saturation
                shifted_x = x[mask] - self.activation_threshold[subset_idx]
                y[mask] = cp.minimum(
                    cp.maximum(shifted_x * self.activation_gain[subset_idx], 0.01 * shifted_x),
                    1.0  # Saturation at 1.0
                )
            elif i == 3:  # NEW: Adaptive activation function
                # Parameterized activation function with learnable parameters
                # p0: controls shape, p1: controls threshold, p2: controls saturation
                for j, idx in enumerate(subset_idx):
                    p0 = self.activation_params[idx, 0]
                    p1 = self.activation_params[idx, 1]
                    p2 = self.activation_params[idx, 2]
                    
                    # Swish-like adaptive activation: x * sigmoid(p0*x + p1) + p2
                    input_val = x[mask][j]
                    sigmoid_val = 1.0 / (1.0 + cp.exp(-(p0 * input_val + p1)))
                    y[mask][j] = input_val * sigmoid_val + p2
        
        return y
    
    def softmax(self, x):
        """Manual implementation of softmax for CuPy"""
        exp_x = cp.exp(x - cp.max(x))  # Subtract max for numerical stability
        return exp_x / cp.sum(exp_x)
    
    def _update_neuromodulation(self, input_data=None, prediction_error=None):
        """
        Updates neuromodulatory signals that regulate network plasticity and dynamics.
        Called during forward pass to adjust neuron excitability and synaptic efficacy.
        
        NEW: Incorporates prediction error and layer-specific modulation
        """
        # Target-based adaptation of neuromodulators
        for key in self.modulation:
            if key in self.modulation_targets:
                lower, upper = self.modulation_targets[key]
                target = (lower + upper) / 2
                
                # Move slowly toward target with random fluctuations
                self.modulation[key] += self.modulation_adaptation_rate * (
                    target - self.modulation[key] + cp.random.normal(0, 0.05)
                )
                # Clip to valid range
                self.modulation[key] = max(lower, min(upper, self.modulation[key]))
        
        # Calculate modulatory signals based on network state
        if hasattr(self, 'h') and self.h is not None:
            # Simple implementation - dynamics based on current activation
            activity_level = float(cp.mean(cp.abs(self.h)))
            
            # Activity-based neuromodulation
            if activity_level > 0.6:  # High activity
                self.activity_triggered_modulation['ach_release'] = 0.2  # Attention boost
            elif activity_level < 0.2:  # Low activity
                self.activity_triggered_modulation['ne_release'] = 0.3   # Exploration boost
        
        # NEW: Prediction error-based neuromodulation
        if prediction_error is not None:
            # Dopamine release based on prediction error (reward prediction)
            if prediction_error < -0.1:  # Better than expected (negative error)
                self.activity_triggered_modulation['dopamine_release'] = 0.3
            elif prediction_error > 0.1:  # Worse than expected (positive error)
                self.activity_triggered_modulation['ne_release'] = 0.2  # Increase exploration
        
        # Incorporate activity-triggered modulation into core neuromodulators
        self.modulation['acetylcholine'] += self.activity_triggered_modulation['ach_release']
        self.modulation['norepinephrine'] += self.activity_triggered_modulation['ne_release']
        self.modulation['dopamine'] += self.activity_triggered_modulation['dopamine_release']
        
        # Decay activity-triggered releases
        for key in self.activity_triggered_modulation:
            self.activity_triggered_modulation[key] *= 0.9  # 10% decay per step
        
        # NEW: Apply layer-specific neuromodulation
        for i in range(self.num_layers):
            layer_mask = self.neuron_layer == i
            if cp.any(layer_mask):
                # Apply layer-specific modulation to neuron sensitivities
                for j, key in enumerate(self.modulation):
                    # Adjust sensitivity based on layer
                    mod_factor = self.layer_modulation[i, j]
                    # Apply to neurons in this layer through their sensitivity
                    self.neuromod_sensitivity[layer_mask, j] *= mod_factor
        
        return self.modulation
    
    def update_performance(self, batch_loss, batch_accuracy=None, class_accuracies=None, prediction_error=None):
        """
        Tracks performance metrics during training to adjust learning parameters.
        Called after each batch to update internal performance tracking.
        
        NEW: Incorporates class-specific performance tracking and prediction error
        """
        try:
            # Ensure metrics are stored as native Python floats
            # This prevents type inconsistencies with CuPy/NumPy arrays
            try:
                # Handle different possible types
                if isinstance(batch_loss, (cp.ndarray, np.ndarray)):
                    if batch_loss.size == 1:
                        self.recent_loss_history.append(float(batch_loss))
                    else:
                        self.recent_loss_history.append(float(batch_loss.mean()))
                else:
                    self.recent_loss_history.append(float(batch_loss))
                
                if batch_accuracy is not None:
                    if isinstance(batch_accuracy, (cp.ndarray, np.ndarray)):
                        if batch_accuracy.size == 1:
                            self.recent_accuracy_history.append(float(batch_accuracy))
                        else:
                            self.recent_accuracy_history.append(float(batch_accuracy.mean()))
                    else:
                        self.recent_accuracy_history.append(float(batch_accuracy))
            except Exception as e:
                print(f"Error converting performance metrics: {e}")
                # Use default values if conversion fails
                self.recent_loss_history.append(1.0)
                if batch_accuracy is not None:
                    self.recent_accuracy_history.append(0.0)
            
            # NEW: Update class-specific performance if provided
            if class_accuracies is not None:
                if 'buy' in class_accuracies:
                    self.class_performance['buy'].append(float(class_accuracies['buy']))
                if 'sell' in class_accuracies:
                    self.class_performance['sell'].append(float(class_accuracies['sell']))
                if 'hold' in class_accuracies:
                    self.class_performance['hold'].append(float(class_accuracies['hold']))
            
            # NEW: Store prediction error for neurogenesis decisions
            if prediction_error is not None:
                self.error_history.append(float(prediction_error))
            
            # Calculate performance gradients (1st derivative)
            if len(self.recent_loss_history) >= 2:
                # Smoothed gradient over last few batches
                recent_losses = list(self.recent_loss_history)[-5:]
                if len(recent_losses) >= 2:
                    self.performance_gradient = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            
            # Calculate acceleration (2nd derivative)
            if len(self.recent_loss_history) >= 3:
                old_gradient = 0
                if hasattr(self, 'performance_gradient_prev'):
                    old_gradient = self.performance_gradient_prev
                
                self.performance_acceleration = self.performance_gradient - old_gradient
                self.performance_gradient_prev = self.performance_gradient
            
            # Update sliding window statistics - CRITICAL: Proper array conversion
            if len(self.recent_loss_history) >= self.sliding_window_size:
                # Get window as a list first
                window_list = list(self.recent_loss_history)[-self.sliding_window_size:]
                
                # Convert list to NumPy array first (safer than direct CuPy conversion)
                window_array = np.array(window_list, dtype=np.float32)
                
                # Calculate statistics using NumPy (more reliable for mixed Python/CuPy environments)
                mean_val = float(np.mean(window_array))
                std_val = float(np.std(window_array))
                
                # Store as Python floats
                self.sliding_stats['loss_mean'].append(mean_val)
                self.sliding_stats['loss_std'].append(std_val)
            
            if len(self.recent_accuracy_history) >= self.sliding_window_size and batch_accuracy is not None:
                # Same safe conversion process for accuracy
                acc_window_list = list(self.recent_accuracy_history)[-self.sliding_window_size:]
                acc_window_array = np.array(acc_window_list, dtype=np.float32)
                
                # Calculate using NumPy
                acc_mean = float(np.mean(acc_window_array))
                acc_std = float(np.std(acc_window_array))
                
                # Store as Python floats
                self.sliding_stats['accuracy_mean'].append(acc_mean)
                self.sliding_stats['accuracy_std'].append(acc_std)
            
            # Adaptive learning phase detection
            self._detect_learning_phase()
            
            # NEW: Update developmental stage based on step counter
            self._update_developmental_stage()
            
            return True
            
        except Exception as e:
            # Comprehensive error handling to prevent training failure
            print(f"Performance tracking error: {e}")
            import traceback
            traceback.print_exc()
            
            # Create default values if calculations failed
            if not hasattr(self, 'performance_gradient'):
                self.performance_gradient = 0.0
            if not hasattr(self, 'performance_acceleration'):
                self.performance_acceleration = 0.0
                
            return False
    
    def _detect_learning_phase(self):
        """Detect optimal learning phase based on performance metrics"""
        try:
            # First phase - high variance, high loss
            if (len(self.sliding_stats['loss_std']) > 0 and 
                self.sliding_stats['loss_std'][-1] > 0.1 and
                self.performance_gradient < 0):
                self.set_learning_phase('exploration')
            
            # Second phase - consistently decreasing loss
            elif (self.performance_gradient < -0.01 and
                 self.performance_acceleration > -0.005):
                self.set_learning_phase('exploitation')
            
            # Third phase - small improvements, low variance
            elif (len(self.sliding_stats['loss_std']) > 0 and
                 self.sliding_stats['loss_std'][-1] < 0.05 and
                 -0.01 < self.performance_gradient < 0):
                self.set_learning_phase('refinement')
            
            # Final phase - plateauing performance
            elif (self.performance_gradient > -0.001 and
                 len(self.sliding_stats['loss_mean']) > 5 and
                 abs(self.sliding_stats['loss_mean'][-1] - self.sliding_stats['loss_mean'][-5]) < 0.001):
                self.set_learning_phase('convergence')
        except Exception as e:
            print(f"Learning phase detection error: {e}")
            # Default to current phase if detection fails
    
    def _update_developmental_stage(self):
        """NEW: Update developmental stage based on step counter and performance"""
        try:
            # Transition from critical period to refinement
            if self.developmental_stage == 'critical_period' and self.step_counter > self.critical_period_end:
                self.developmental_stage = 'refinement'
                print(f"Step {self.step_counter}: Transitioning from critical period to refinement stage")
                # Reduce plasticity rates as we exit critical period
                self.plasticity_rates *= 0.7
                # Increase stability of existing connections
                self.synapse_elimination_threshold *= 0.8
            
            # Transition from refinement to mature
            elif self.developmental_stage == 'refinement' and self.step_counter > self.maturation_end:
                self.developmental_stage = 'mature'
                print(f"Step {self.step_counter}: Transitioning to mature developmental stage")
                # Further reduce plasticity in mature stage
                self.plasticity_rates *= 0.5
                # Make pruning more selective
                self.synapse_elimination_threshold *= 0.5
                # Reduce neurogenesis
                self.max_growth_per_phase = max(2, self.max_growth_per_phase // 2)
            
            # Adjust parameters based on current stage
            if self.developmental_stage == 'critical_period':
                # High plasticity, rapid growth and pruning
                self.structural_plasticity_period = 30  # More frequent structural changes
            elif self.developmental_stage == 'refinement':
                # Moderate plasticity, more selective changes
                self.structural_plasticity_period = 50
            else:  # mature
                # Low baseline plasticity, very selective changes
                self.structural_plasticity_period = 100  # Less frequent structural changes
        
        except Exception as e:
            print(f"Error updating developmental stage: {e}")
    
    def _process_dendritic_inputs(self, x):
        """
        NEW: Process inputs through dendritic compartments (mini-computers)
        Each dendrite specializes in processing specific input patterns
        """
        try:
            # Store current input for dendritic weight updates
            self.current_input = x.copy()
            
            # Process input through each dendritic compartment
            for i in range(self.hidden_dim):
                for d in range(self.num_dendrites):
                    # Each dendrite has its own weights and processes inputs independently
                    dendritic_input = self.dendritic_weights[i, d] @ x
                    
                    # Apply nonlinear activation within dendrite
                    if dendritic_input > 0:
                        # Supralinear integration for strong inputs (dendrite spike)
                        self.dendritic_activations[i, d] = dendritic_input ** 1.5
                    else:
                        # Sublinear integration for weak inputs
                        self.dendritic_activations[i, d] = 0.1 * dendritic_input
            
            # Return the sum of dendritic activations for each neuron
            return cp.sum(self.dendritic_activations, axis=1)
            
        except Exception as e:
            print(f"Error in dendritic processing: {e}")
            return cp.zeros(self.hidden_dim)

    def _update_neuronal_autonomy(self, x, prediction_error=None):
        """
        NEW: Update neuronal autonomy parameters
        Each neuron acts as a mini-computer with its own objectives
        """
        try:
            # Update intrinsic motivation based on novelty and prediction error
            if prediction_error is not None:
                # Neurons that contribute to error reduction get higher motivation
                error_contribution = cp.abs(self.h * prediction_error)
                # Normalize to [0, 1] range
                if cp.max(error_contribution) > 0:
                    error_contribution = error_contribution / cp.max(error_contribution)
                
                # Update intrinsic motivation (curiosity)
                # High error = high curiosity for neurons that can help
                self.intrinsic_motivation = 0.95 * self.intrinsic_motivation + 0.05 * error_contribution
            
            # Update local learning objectives
            # Each neuron tries to maximize its contribution while minimizing energy
            contribution = cp.abs(self.h) / (cp.sum(cp.abs(self.h)) + 1e-10)
            energy_efficiency = 1.0 - self.energy_consumption_rate * cp.abs(self.h)
            
            # Local objective combines contribution and energy efficiency
            self.local_objectives = 0.7 * contribution + 0.3 * energy_efficiency
            
            # Update neuron-specific learning rates based on local objectives
            # Neurons with higher objectives get higher learning rates
            self.neuron_learning_rates = 0.001 + 0.01 * self.local_objectives
            
            # Update predictive coding
            # Each neuron tries to predict its next state
            self.prediction_targets = self.h.copy()  # Current state becomes target for next step
            
            # Update lateral inhibition based on activity
            # Neurons that fire together strengthen their lateral connections
            active_neurons = cp.abs(self.h) > 0.2
            if cp.sum(active_neurons) > 1:  # Need at least 2 active neurons
                active_idx = cp.where(active_neurons)[0]
                
                # Convert array indices to Python integers
                # Handle both CuPy and NumPy cases correctly
                if use_cupy:
                    # Convert CuPy array to NumPy then extract Python scalars
                    active_idx_list = [int(idx) for idx in cp.asnumpy(active_idx)]
                else:
                    # Convert NumPy array elements to Python integers
                    active_idx_list = [int(idx) for idx in active_idx]
                
                for i in active_idx_list:
                    for j in active_idx_list:
                        if i != j:
                            # Strengthen inhibition between co-active neurons
                            # Now using Python integers for indexing
                            self.lateral_inhibition[i, j] += 0.01
            
            # Decay lateral inhibition
            self.lateral_inhibition *= 0.99
            
            # Apply lateral inhibition
            inhibition = self.lateral_inhibition @ cp.abs(self.h)
            return inhibition
            
        except Exception as e:
            print(f"Error updating neuronal autonomy: {e}")
            return cp.zeros(self.hidden_dim)
    
    def _update_ensembles(self):
        """
        NEW: Update dynamic neuronal ensembles
        Neurons form functional groups that process specific patterns
        """
        try:
            # Identify active neurons
            active_neurons = cp.abs(self.h) > 0.2
            active_idx = cp.where(active_neurons)[0]
            
            if len(active_idx) < 2:  # Need at least 2 neurons for an ensemble
                return
            
            # Calculate pairwise coherence between active neurons
            for i in active_idx:
                for j in active_idx:
                    if i != j:
                        # Coherence based on phase similarity and activity correlation
                        phase_coherence = cp.cos(self.phase[i] - self.phase[j])
                        # Update coherence matrix
                        self.coherence[i, j] = 0.9 * self.coherence[i, j] + 0.1 * phase_coherence
            
            # Find strongly coherent groups (potential ensembles)
            threshold = 0.7  # Coherence threshold for ensemble membership
            
            # Reset ensemble membership
            self.ensemble_membership *= 0.9  # Decay old memberships
            
            # Assign neurons to ensembles based on coherence
            for e in range(self.max_ensembles):
                # Find neurons that are coherent with existing ensemble members
                ensemble_neurons = cp.where(self.ensemble_membership[:, e] > 0.5)[0]
                
                if len(ensemble_neurons) > 0:
                    # Existing ensemble - add neurons that are coherent with current members
                    for i in active_idx:
                        if i not in ensemble_neurons:
                            # Check coherence with existing members
                            mean_coherence = cp.mean(self.coherence[i, ensemble_neurons])
                            if mean_coherence > threshold:
                                self.ensemble_membership[i, e] += 0.3
                else:
                    # Try to form a new ensemble from highly coherent neurons
                    # Find the most coherent pair of neurons not in an ensemble
                    max_coherence = 0
                    max_pair = None
                    
                    for i in active_idx:
                        if cp.sum(self.ensemble_membership[i]) < 0.5:  # Not strongly in any ensemble
                            for j in active_idx:
                                if i != j and cp.sum(self.ensemble_membership[j]) < 0.5:
                                    if self.coherence[i, j] > max_coherence:
                                        max_coherence = self.coherence[i, j]
                                        max_pair = (i, j)
                    
                    if max_pair is not None and max_coherence > threshold:
                        # Create new ensemble with this pair
                        i, j = max_pair
                        self.ensemble_membership[i, e] = 1.0
                        self.ensemble_membership[j, e] = 1.0
                        break  # Only create one new ensemble per update
            
            # Normalize ensemble membership (soft assignment)
            row_sums = cp.sum(self.ensemble_membership, axis=1, keepdims=True)
            row_sums = cp.clip(row_sums, 1e-10, None)  # Avoid division by zero
            self.ensemble_membership = self.ensemble_membership / row_sums
            
            # Update ensemble activity
            for e in range(self.max_ensembles):
                # Weighted sum of member neuron activations
                self.ensemble_activity[e] = cp.sum(cp.abs(self.h) * self.ensemble_membership[:, e])
            
            # Update assembly vectors (prototype patterns for each ensemble)
            for e in range(self.max_ensembles):
                if self.ensemble_activity[e] > 0.1:  # Active ensemble
                    # Update assembly vector as weighted average of member activations
                    weights = self.ensemble_membership[:, e]
                    self.assembly_vectors[e] = (0.9 * self.assembly_vectors[e] + 
                                              0.1 * (weights * self.h) / (cp.sum(weights) + 1e-10))
            
        except Exception as e:
            print(f"Error updating ensembles: {e}")
    
    def _apply_hierarchical_processing(self, x):
        """
        NEW: Apply hierarchical processing across layers
        Lower layers process faster, higher layers integrate information
        """
        try:
            # Process through hierarchical layers
            layer_outputs = [cp.zeros(self.hidden_dim) for _ in range(self.num_layers)]
            
            # Forward pass through layers
            for layer in range(self.num_layers):
                # Get neurons in this layer
                layer_mask = self.neuron_layer == layer
                
                if layer == 0:
                    # First layer processes raw input
                    layer_inputs = x
                else:
                    # Higher layers get input from lower layers
                    prev_layer_mask = self.neuron_layer == (layer - 1)
                    # Convert boolean mask to indices for safer selection
                    prev_layer_indices = cp.where(prev_layer_mask)[0]
                    if len(prev_layer_indices) > 0:
                        layer_inputs = self.h[prev_layer_mask]  # Boolean indexing is safe here
                    else:
                        continue  # Skip if no neurons in previous layer
                
                # Process inputs for this layer's neurons
                if cp.any(layer_mask):
                    # Get indices for this layer's neurons for safer operations
                    layer_indices = cp.where(layer_mask)[0]
                    
                    if layer == 0:
                        # Fast, detailed processing in lower layers
                        # Use advanced indexing for the weights
                        weighted_input = cp.matmul(self.Wh[layer_mask], layer_inputs)
                        # Assign results using boolean mask (safe for assignment)
                        layer_outputs[layer][layer_mask] = weighted_input
                    else:
                        # More integrative processing in higher layers
                        layer_count = int(cp.sum(layer_mask))  # Get count as Python int
                        input_size = len(layer_inputs)
                        
                        # Create weight attribute name
                        weight_attr = f'layer_{layer-1}_to_{layer}_weights'
                        
                        # Initialize if needed with explicit dimensions
                        if not hasattr(self, weight_attr):
                            setattr(self, weight_attr, 
                                cp.random.randn(layer_count, input_size) * 0.1)
                        
                        # Get projection weights and verify dimensions
                        projection_weights = getattr(self, weight_attr)
                        if projection_weights.shape != (layer_count, input_size):
                            # Resize if dimensions don't match
                            new_weights = cp.random.randn(layer_count, input_size) * 0.1
                            setattr(self, weight_attr, new_weights)
                            projection_weights = new_weights
                        
                        # Apply projection and store result
                        projected = cp.matmul(projection_weights, layer_inputs)
                        # Use boolean mask for assignment (safe)
                        layer_outputs[layer][layer_mask] = projected
            
            # Feedback from higher to lower layers (top-down modulation)
            feedback = cp.zeros(self.hidden_dim)
            
            for layer in range(self.num_layers - 1, 0, -1):  # From highest to second layer
                higher_layer_mask = self.neuron_layer == layer
                lower_layer_mask = self.neuron_layer == (layer - 1)
                
                if cp.any(higher_layer_mask) and cp.any(lower_layer_mask):
                    # Get explicit indices for both layers
                    higher_indices = cp.where(higher_layer_mask)[0] 
                    lower_indices = cp.where(lower_layer_mask)[0]
                    
                    # Only proceed if both layers have neurons
                    if len(higher_indices) > 0 and len(lower_indices) > 0:
                        # Apply feedback connections using boolean masks where safe
                        higher_activations = self.h[higher_layer_mask]
                        
                        # Extract the relevant submatrix using boolean masks
                        feedback_weights_subset = self.feedback_weights[higher_layer_mask][:, lower_layer_mask]
                        
                        # Compute feedback using matrix multiplication
                        layer_feedback = cp.matmul(feedback_weights_subset.T, higher_activations)
                        
                        # Apply feedback to lower layer neurons
                        feedback[lower_layer_mask] += layer_feedback
            
            return layer_outputs, feedback
            
        except Exception as e:
            print(f"Error in hierarchical processing: {e}")
            import traceback
            traceback.print_exc()
            return [cp.zeros(self.hidden_dim) for _ in range(self.num_layers)], cp.zeros(self.hidden_dim)
    
    def forward(self, x, target=None, neuromodulators=None, inference_mode=False):
        """
        Forward pass with biologically plausible dynamics
        
        Parameters:
        x: Input tensor
        target: Optional target for supervised learning and error calculation
        neuromodulators: Optional dict to override neuromodulator values
        inference_mode: If True, disables plasticity for stable inference
        
        Returns:
        Hidden state vector
        """
        try:
            # Verify dimensional consistency before proceeding
            self._validate_dimensions()
            
            # Ensure input is properly formatted
            try:
                if not isinstance(x, cp.ndarray):
                    x = cp.asarray(x)
            except Exception as e:
                print(f"Warning: CuPy conversion failed: {e}. Attempting NumPy fallback.")
                if not isinstance(x, np.ndarray):
                    x = np.array(x)
                # Try again with NumPy array
                try:
                    x = cp.asarray(x)
                except:
                    raise ValueError(f"Failed to convert input to CuPy array: {x}")
            
            # Handle different input shapes
            if len(x.shape) == 0:  # Scalar input
                x = cp.array([float(x)])
            elif len(x.shape) == 1:  # Vector input
                x = x.reshape(-1)
            elif len(x.shape) > 1:  # Multi-dimensional input
                # Flatten or reshape as needed
                x = x.reshape(x.shape[0], -1).flatten()
                
            # Verify dimensions
            if x.shape[0] != self.input_dim:
                raise ValueError(f"Input dimension mismatch: got {x.shape[0]}, expected {self.input_dim}")
            
            # Store previous state for plasticity and rate calculation
            self.h_prev = self.h.copy()
            
            # Calculate prediction error if target is provided
            prediction_error = None
            if target is not None and not inference_mode:
                # Simple prediction error calculation
                if hasattr(self, 'output_weights') and hasattr(self, 'output_bias'):
                    # Generate prediction
                    logits = self.output_weights @ self.h + self.output_bias
                    probs = self.softmax(logits)
                    pred = cp.argmax(probs)
                    label = cp.argmax(target)
                    
                    # Calculate error
                    prediction_error = float(pred != label)
                    
                    # Store for neurogenesis decisions
                    self.error_history.append(prediction_error)
            
            # Apply neuromodulation
            if neuromodulators is not None:
                for key, value in neuromodulators.items():
                    if key in self.modulation:
                        self.modulation[key] = value
            
            # Update neuromodulation based on internal dynamics if not overridden
            else:
                self._update_neuromodulation(x, prediction_error)
            
            # Access key neuromodulators
            glutamate = self.modulation['glutamate']  # Excitation
            gaba = self.modulation['gaba']            # Inhibition
            ach = self.modulation['acetylcholine']    # Input sensitivity/attention
            ne = self.modulation['norepinephrine']    # Exploration/noise
            
            # NEW: Process inputs through dendritic compartments
            dendritic_input = self._process_dendritic_inputs(x)
            
            # NEW: Apply hierarchical processing
            layer_outputs, feedback = self._apply_hierarchical_processing(x)
            
            # GRU gating operations modulated by neuromodulators
            z_input = ach * (self.Wz @ x + 0.2 * dendritic_input)
            z_recurrent = self.Uz @ self.h
            
            # Apply sigmoid activation to all neurons for the gates
            # No need to use adaptive activation for gates
            z = 1.0 / (1.0 + cp.exp(-(z_input + z_recurrent + self.bz)))
            r = 1.0 / (1.0 + cp.exp(-(ach * (self.Wr @ x) + self.Ur @ self.h + self.br)))
            
            # Candidate activation - influenced by glutamate (excitation)
            h_input = glutamate * (self.Wh @ x + 0.3 * dendritic_input)
            h_recurrent = self.Uh @ (r * self.h)
            
            # NEW: Add feedback from higher layers
            h_input += 0.2 * feedback
            
            # Use adaptive activation for the candidate
            h_tilde = self.adaptive_activation(h_input + h_recurrent + self.bh)
            
            # Calculate rate of change for criticality assessment
            self.dh = h_tilde - self.h
            
            # Update hidden state with GRU dynamics
            self.h = (1 - z) * self.h + z * h_tilde
            
            # NEW: Apply neuronal autonomy - lateral inhibition
            inhibition = self._update_neuronal_autonomy(x, prediction_error)
            self.h -= gaba * inhibition  # Inhibitory effect
            
            # Apply Dale's principle - preserve E/I nature of neurons
            # Safe implementation with explicit broadcasting
            excitatory_mask = self.neuron_type > 0
            inhibitory_mask = self.neuron_type < 0
            
            # Process excitatory neurons (preserve positive values)
            self.h[excitatory_mask] = cp.maximum(0, self.h[excitatory_mask])
            
            # Process inhibitory neurons (preserve negative values)
            self.h[inhibitory_mask] = cp.minimum(0, self.h[inhibitory_mask])
            
            # NEW: Apply priority-based energy allocation
            # Neurons with higher priority get more energy
            energy_allocation = self.energy * self.energy_priority
            energy_allocation = energy_allocation / (cp.sum(energy_allocation) + 1e-10)
            
            # Gating by available energy
            energy_gate = (1.0 - cp.exp(-5.0 * energy_allocation))
            self.h = self.h * energy_gate
            
            # Homeostatic regulation via adaptive threshold
            over_active = self.h > self.firing_threshold
            self.firing_threshold += 0.001 * over_active - 0.0001 * (~over_active)
            self.firing_threshold = cp.clip(self.firing_threshold, 0.01, 0.5)
            
            # Apply stochastic noise modulated by norepinephrine (exploration)
            if ne > 0.3 and not inference_mode:
                noise_scale = (ne - 0.3) * 0.2
                self.h += cp.random.randn(self.hidden_dim) * noise_scale
            
            # Track neuron utility
            h_sum = cp.sum(cp.abs(self.h))
            if h_sum > 1e-10:
                active_contribution = cp.abs(self.h) / h_sum
            else:
                active_contribution = cp.zeros_like(self.h)
                
            self.neuron_utility = 0.95 * self.neuron_utility + 0.05 * active_contribution
            
            # Update activity tracking
            self._update_activity_tracking()
            
            # Update energy metabolism
            self._update_energy_metabolism()
            
            # NEW: Update ensemble formation
            self._update_ensembles()
            
            # Apply homeostatic plasticity periodically
            if self.step_counter % 10 == 0 and not inference_mode:
                self._apply_homeostatic_plasticity()
            
            # Apply multi-timescale synaptic plasticity if in training mode
            if not inference_mode and self.modulation['dopamine'] > 0.3:
                # Apply hybrid Hebbian-STDP plasticity
                self._apply_synaptic_plasticity(target)
                
                # NEW: Apply structural synaptic plasticity
                if self.step_counter % 20 == 0:
                    self._apply_structural_synaptic_plasticity()
                
                # Consolidate fast to slow weights periodically
                if self.step_counter % 20 == 0:
                    self._consolidate_memory()
            
            # Apply structural plasticity periodically
            if (self.step_counter % self.structural_plasticity_period == 0 and 
                self.step_counter > 0 and not inference_mode):
                try:
                    # NEW: Apply task-driven neurogenesis
                    self._apply_task_driven_neurogenesis(prediction_error)
                    # Verify dimensions after structural changes
                    self._validate_dimensions()
                except Exception as e:
                    print(f"Skipping structural plasticity: {e}")
            
            # Update information-theoretic metrics periodically
            if self.step_counter % 50 == 0 and not inference_mode:
                self._update_information_metrics()
            
            # Increment step counter
            self.step_counter += 1
            
            # Store recent activation for stability analysis
            self.recent_activations.append(float(cp.mean(cp.abs(self.h))))
            
            return self.h
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return previous state as fallback
            return self.h_prev
    
    def _update_activity_tracking(self):
        """Update multi-timescale activity tracking"""
        try:
            # Ensure dimensions match
            for key in self.activity_history:
                if self.activity_history[key].shape[0] != self.hidden_dim:
                    self.activity_history[key] = cp.zeros((self.hidden_dim,))
            
            # Current activation energy (squared for energy calculation)
            activity = self.h ** 2
            
            # Update history at different timescales
            self.activity_history['fast'] = (self.activity_history['fast'] * self.time_constants[0] + 
                                           activity * (1 - self.time_constants[0]))
            self.activity_history['medium'] = (self.activity_history['medium'] * self.time_constants[1] + 
                                             activity * (1 - self.time_constants[1]))
            self.activity_history['slow'] = (self.activity_history['slow'] * self.time_constants[2] + 
                                           activity * (1 - self.time_constants[2]))
            
            # Update STDP traces - ensure dimensions match
            if self.pre_synaptic_trace.shape[0] != self.hidden_dim:
                self.pre_synaptic_trace = cp.zeros((self.hidden_dim,))
                
            if self.post_synaptic_trace.shape[0] != self.hidden_dim:
                self.post_synaptic_trace = cp.zeros((self.hidden_dim,))
            
            # Update traces
            self.pre_synaptic_trace = (self.pre_synaptic_trace * 
                                     (1 - 1/self.stdp_params['tau_plus']) + 
                                     self.h_prev)
            self.post_synaptic_trace = (self.post_synaptic_trace * 
                                      (1 - 1/self.stdp_params['tau_minus']) + 
                                      self.h)
            
            # NEW: Update temporal memory for causal associations
            self.temporal_memory.append(self.h.copy())
            
            # NEW: Update causal association matrix
            if len(self.temporal_memory) >= 2:
                # Get current and previous activations
                current = self.temporal_memory[-1]
                previous = self.temporal_memory[-2]
                
                # Update causal matrix - neurons that activate in sequence
                for i in range(self.hidden_dim):
                    if abs(previous[i]) > 0.2:  # Previously active
                        for j in range(self.hidden_dim):
                            if abs(current[j]) > 0.2:  # Currently active
                                # Strengthen causal connection from i to j
                                self.causal_matrix[i, j] += 0.01
                
                # Decay causal connections
                self.causal_matrix *= 0.999
            
        except Exception as e:
            print(f"Error updating activity tracking: {e}")
    
    def _update_energy_metabolism(self):
        """Update energy resources based on activity"""
        try:
            # Ensure dimensions match
            if self.energy.shape[0] != self.hidden_dim:
                self.energy = cp.ones((self.hidden_dim,))
                
            # Energy consumption proportional to activity
            energy_consumption = self.energy_consumption_rate * cp.abs(self.h)
            
            # Ensure energy doesn't go negative
            energy_consumption = cp.minimum(energy_consumption, self.energy)
            
            # Compute metabolic cost
            self.metabolic_cost = float(cp.sum(energy_consumption))
            
            # Consume energy
            self.energy -= energy_consumption
            
            # NEW: Priority-based energy recovery
            # Neurons with higher utility get faster energy recovery
            recovery_priority = 0.5 + 0.5 * self.neuron_utility
            recovery = self.energy_recovery_rate * (1.0 - self.energy) * recovery_priority
            self.energy += recovery
            self.energy = cp.clip(self.energy, 0.0, 1.0)
            
            # Update neuron health
            if self.health.shape[0] != self.hidden_dim:
                self.health = cp.ones((self.hidden_dim,))
                
            energy_deficit = energy_consumption > 0.8
            self.health -= 0.01 * energy_deficit
            
            # Health recovery for less active neurons
            self.health += self.recovery_rate * (1.0 - self.health) * (energy_consumption < 0.3)
            self.health = cp.clip(self.health, 0.0, 1.0)
            
            # Calculate efficiency ratio if information gain is measured
            if hasattr(self, 'information_gain') and self.information_gain > 0:
                self.efficiency_ratio = self.information_gain / (self.metabolic_cost + 1e-10)
            
            # NEW: Update energy priority based on utility and information content
            if hasattr(self, 'information_content') and self.information_content.shape[0] == self.hidden_dim:
                # Neurons with higher information content get higher priority
                info_priority = self.information_content / (cp.max(self.information_content) + 1e-10)
                # Combine with utility
                self.energy_priority = 0.7 * self.neuron_utility + 0.3 * info_priority
            else:
                self.energy_priority = self.neuron_utility.copy()
            
        except Exception as e:
            print(f"Error updating energy metabolism: {e}")
    
    def _apply_homeostatic_plasticity(self):
        """
        Apply homeostatic plasticity mechanisms to maintain stable network activity
        This includes synaptic scaling, intrinsic plasticity adjustments, and metaplasticity
        """
        try:
            # 1. Synaptic scaling - scale synaptic weights based on activity history
            # Get medium-term activity history for each neuron
            medium_activity = self.activity_history['medium']
            
            # Calculate deviation from target activity
            target = self.target_activity * cp.ones_like(medium_activity)
            deviation = medium_activity - target
            
            # Apply scaling to recurrent weights proportional to deviation
            # Neurons that are too active have their incoming weights decreased
            # Neurons that are too inactive have their incoming weights increased
            scale_factor = 1.0 - self.synaptic_scaling_rate * deviation
            scale_factor = cp.clip(scale_factor, 0.8, 1.2)  # Limit scaling range
            
            # Apply scaling differently to excitatory and inhibitory connections
            # Scale rows of weight matrices (incoming weights)
            self.Uz = cp.diag(scale_factor) @ self.Uz
            self.Ur = cp.diag(scale_factor) @ self.Ur
            self.Uh = cp.diag(scale_factor) @ self.Uh
            
            # 2. Intrinsic plasticity - adjust activation function parameters
            # Adjust activation gain based on activity
            adjustment = self.metaplasticity_rate * deviation
            self.activation_gain -= adjustment
            self.activation_gain = cp.clip(self.activation_gain, 0.5, 2.0)
            
            # Adjust activation threshold based on activity
            self.activation_threshold += 0.5 * adjustment
            self.activation_threshold = cp.clip(self.activation_threshold, -0.5, 0.5)
            
            # NEW: Adjust adaptive activation parameters for type 3 neurons
            adaptive_mask = self.activation_types == 3
            if cp.any(adaptive_mask):
                # Adjust parameters based on activity
                self.activation_params[adaptive_mask, 0] += 0.01 * deviation[adaptive_mask]  # Shape parameter
                self.activation_params[adaptive_mask, 1] -= 0.01 * deviation[adaptive_mask]  # Threshold parameter
                # Clip to reasonable ranges
                self.activation_params = cp.clip(self.activation_params, -2.0, 2.0)
            
            # 3. Metaplasticity - adjust plasticity parameters based on learning phase
            # Determine homeostatic needs based on learning phase
            if self.learning_phases['exploration']:
                # In exploration, allow higher plasticity variability
                self.plasticity_rates += cp.random.normal(0, 0.0005, size=self.hidden_dim)
            elif self.learning_phases['convergence']:
                # In convergence, reduce plasticity
                self.plasticity_rates *= 0.99
            
            # NEW: Adjust plasticity based on developmental stage
            if self.developmental_stage == 'critical_period':
                # Higher plasticity during critical period
                self.plasticity_rates = cp.clip(self.plasticity_rates, 0.0005, 0.02)
            elif self.developmental_stage == 'refinement':
                # Moderate plasticity during refinement
                self.plasticity_rates = cp.clip(self.plasticity_rates, 0.0002, 0.01)
            else:  # mature
                # Lower plasticity in mature stage
                self.plasticity_rates = cp.clip(self.plasticity_rates, 0.0001, 0.005)
            
            return True
            
        except Exception as e:
            print(f"Error in homeostatic plasticity: {e}")
            return False
    
    def _apply_synaptic_plasticity(self, target=None):
        """
        Apply activity-dependent plasticity to synaptic weights with dimensional safeguards
        
        NEW: Incorporates true associative learning and multi-factor plasticity
        """
        try:
            # 1. CRITICAL: Verify and synchronize all tensor dimensions first
            self._synchronize_tensor_dimensions()
            
            # 2. Create outer products with dimensional safety checks
            if self.h.shape[0] != self.hidden_dim or self.h_prev.shape[0] != self.hidden_dim:
                # Force resize state vectors to current network size
                self.h = self._resize_array(self.h, self.hidden_dim)
                self.h_prev = self._resize_array(self.h_prev, self.hidden_dim)
            
            # 3. Hebbian component based on pre/post correlation with size verification
            pre_post_activity = cp.zeros((self.hidden_dim, self.hidden_dim))
            valid_h_size = min(self.h.shape[0], self.hidden_dim)
            valid_h_prev_size = min(self.h_prev.shape[0], self.hidden_dim)
            # Use valid subsets of each vector for the outer product
            valid_h = self.h[:valid_h_size]
            valid_h_prev = self.h_prev[:valid_h_prev_size]
            # Create partial pre_post_activity matrix with valid dimensions
            partial_size = min(valid_h_size, valid_h_prev_size)
            if partial_size > 0:
                pre_post_partial = cp.outer(valid_h[:partial_size], valid_h_prev[:partial_size])
                # Safely copy into the correctly sized matrix
                pre_post_activity[:partial_size, :partial_size] = pre_post_partial
            else:
                # Handle empty case
                print("Warning: Empty partial size in synaptic plasticity calculation")
            
            # 4. STDP component with similar dimensional safeguards
            # Ensure STDP traces have proper dimensions
            if self.pre_synaptic_trace.shape[0] != self.hidden_dim:
                self.pre_synaptic_trace = self._resize_array(self.pre_synaptic_trace, self.hidden_dim)
            if self.post_synaptic_trace.shape[0] != self.hidden_dim:
                self.post_synaptic_trace = self._resize_array(self.post_synaptic_trace, self.hidden_dim)
                
            # Create STDP matrices with verified dimensions
            stdp_component = cp.zeros((self.hidden_dim, self.hidden_dim))
            valid_post_size = min(self.post_synaptic_trace.shape[0], self.hidden_dim)
            valid_pre_size = min(self.pre_synaptic_trace.shape[0], self.hidden_dim)
            partial_size = min(valid_post_size, valid_pre_size)
            # Create partial components with valid dimensions
            if partial_size > 0:
                stdp_partial = cp.outer(
                    self.post_synaptic_trace[:partial_size],
                    self.pre_synaptic_trace[:partial_size]
                )
                stdp_component[:partial_size, :partial_size] = stdp_partial
                
            stdp_potentiation = self.stdp_params['A_plus'] * stdp_component
            
            # Reversed timing for depression
            stdp_depression = cp.zeros((self.hidden_dim, self.hidden_dim))
            if partial_size > 0:
                stdp_depression_partial = cp.outer(
                    self.pre_synaptic_trace[:partial_size],
                    self.post_synaptic_trace[:partial_size]
                )
                stdp_depression[:partial_size, :partial_size] = stdp_depression_partial
                
            stdp_depression = -self.stdp_params['A_minus'] * stdp_depression
            
            # NEW: True associative learning component
            associative_component = cp.zeros((self.hidden_dim, self.hidden_dim))
            
            # Use causal matrix to strengthen connections between causally related neurons
            associative_component = 0.01 * self.causal_matrix
            
            # NEW: Multi-factor plasticity modulation
            # Calculate modulation factors
            reward_factor = self.plasticity_factors['reward']
            surprise_factor = self.plasticity_factors['surprise']
            attention_factor = self.plasticity_factors['attention']
            
            # Modulate based on neuromodulators
            reward_factor *= self.modulation['dopamine']
            surprise_factor *= self.modulation['norepinephrine']
            attention_factor *= self.modulation['acetylcholine']
            
            # Apply target-based modulation if available
            if target is not None:
                # Simple reward signal based on target
                label = cp.argmax(target)
                if hasattr(self, 'output_weights') and hasattr(self, 'output_bias'):
                    logits = self.output_weights @ self.h + self.output_bias
                    probs = self.softmax(logits)
                    pred = cp.argmax(probs)
                    
                    # Increase reward factor for correct predictions
                    if pred == label:
                        reward_factor *= 1.5
                    else:
                        # Increase surprise factor for incorrect predictions
                        surprise_factor *= 1.5
            
            # 5. Verify plasticity_rates dimensions
            if self.plasticity_rates.shape[0] != self.hidden_dim:
                self.plasticity_rates = self._resize_array(self.plasticity_rates, self.hidden_dim)
            
            # 6. Compute plasticity matrix with safeguards and multi-factor modulation
            # Use diagonal matrix multiplication for proper broadcasting
            plasticity_diag = cp.diag(self.plasticity_rates)
            plasticity_matrix = plasticity_diag @ (
                0.3 * pre_post_activity * reward_factor +
                0.3 * (stdp_potentiation + stdp_depression) * attention_factor +
                0.3 * associative_component * surprise_factor
            )
            
            # 7. Verify and initialize fast_weights if needed
            if self.fast_weights.shape != (self.hidden_dim, self.hidden_dim):
                old_weights = self.fast_weights
                self.fast_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
                
                # Copy existing weights if possible
                min_rows = min(old_weights.shape[0], self.hidden_dim)
                min_cols = min(old_weights.shape[1], self.hidden_dim)
                self.fast_weights[:min_rows, :min_cols] = old_weights[:min_rows, :min_cols]
                print(f"Resized fast_weights from {old_weights.shape} to {self.fast_weights.shape}")
            
            # NEW: Apply structural connectivity constraints
            # Only update weights where there are physical connections
            plasticity_matrix = plasticity_matrix * self.connectivity
            
            # 8. Apply changes to fast weights
            self.fast_weights += plasticity_matrix
            
            # 9. Apply Dale's principle with dimensional safety
            # Verify neuron_type dimensions
            if self.neuron_type.shape[0] != self.hidden_dim:
                self.neuron_type = self._resize_array(self.neuron_type, self.hidden_dim)
                
            # Apply Dale's principle safely with proper indexes
            for i in range(self.hidden_dim):
                if i < self.neuron_type.shape[0]:  # Safety check
                    if self.neuron_type[i] < 0:  # Inhibitory neuron
                        self.fast_weights[i, :] = -cp.abs(self.fast_weights[i, :])
                    else:  # Excitatory neuron
                        self.fast_weights[i, :] = cp.abs(self.fast_weights[i, :])
            
            # 10. Weight normalization for stability
            # Row-wise normalization with safety checks
            norm = cp.sqrt(cp.sum(self.fast_weights**2, axis=1, keepdims=True))
            norm = cp.clip(norm, 1e-10, None)  # Avoid division by zero
            self.fast_weights = self.fast_weights / norm
            
            # 11. Update eligibility traces with safety checks
            if not hasattr(self, 'eligibility_traces') or self.eligibility_traces.shape != (self.hidden_dim, self.hidden_dim):
                self.eligibility_traces = cp.zeros((self.hidden_dim, self.hidden_dim))
                
                # Copy existing traces if possible
                if hasattr(self, 'eligibility_traces'):
                    old_traces = self.eligibility_traces
                    min_rows = min(old_traces.shape[0], self.hidden_dim)
                    min_cols = min(old_traces.shape[1], self.hidden_dim)
                    self.eligibility_traces[:min_rows, :min_cols] = old_traces[:min_rows, :min_cols]
            
            self.eligibility_traces = self.e_trace_decay * self.eligibility_traces + plasticity_matrix
            
            # NEW: Update dendritic weights based on input correlations
            self._update_dendritic_weights(target)
            
            return True
            
        except Exception as e:
            print(f"Error in synaptic plasticity: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_dendritic_weights(self, target=None):
        """
        NEW: Update dendritic weights based on input correlations
        Each dendrite specializes in processing specific input patterns
        """
        try:
            # Skip if no recent inputs
            if not hasattr(self, 'recent_inputs'):
                self.recent_inputs = deque(maxlen=10)
                return
            
            # Store current input
            if hasattr(self, 'current_input'):
                self.recent_inputs.append(self.current_input)
            
            if len(self.recent_inputs) < 2:
                return
            
            # Calculate learning rate based on performance
            if target is not None and hasattr(self, 'output_weights') and hasattr(self, 'output_bias'):
                # Generate prediction
                logits = self.output_weights @ self.h + self.output_bias
                probs = self.softmax(logits)
                pred = cp.argmax(probs)
                label = cp.argmax(target)
                
                # Higher learning rate for incorrect predictions
                dendrite_lr = 0.01 * (1.0 + 2.0 * (pred != label))
            else:
                dendrite_lr = 0.01
            
            # Update each neuron's dendritic weights
            for i in range(self.hidden_dim):
                # Skip inactive neurons
                if abs(self.h[i]) < 0.1:
                    continue
                
                # Find the most active dendrite for this neuron
                dendrite_activations = self.dendritic_activations[i]
                most_active_dendrite = cp.argmax(dendrite_activations)
                
                # Update the weights of the most active dendrite
                # This implements competitive learning among dendrites
                current_input = self.recent_inputs[-1]
                
                # Hebbian update: strengthen weights for inputs that activated this dendrite
                self.dendritic_weights[i, most_active_dendrite] += dendrite_lr * (
                    current_input * self.h[i] - 0.01 * self.dendritic_weights[i, most_active_dendrite]
                )
                
                # Occasionally update a random dendrite to encourage exploration
                if cp.random.random() < 0.1:
                    random_dendrite = cp.random.randint(0, self.num_dendrites)
                    if random_dendrite != most_active_dendrite:
                        # Smaller update to random dendrite
                        self.dendritic_weights[i, random_dendrite] += 0.2 * dendrite_lr * (
                            current_input * self.h[i] - 0.01 * self.dendritic_weights[i, random_dendrite]
                        )
            
            # Normalize dendritic weights for stability
            for i in range(self.hidden_dim):
                for d in range(self.num_dendrites):
                    norm = cp.linalg.norm(self.dendritic_weights[i, d])
                    if norm > 1e-10:
                        self.dendritic_weights[i, d] /= norm
            
        except Exception as e:
            print(f"Error updating dendritic weights: {e}")
    
    def _apply_structural_synaptic_plasticity(self):
        """
        NEW: Apply structural synaptic plasticity
        Form and eliminate synaptic connections based on activity patterns
        """
        try:
            # 1. Synapse formation - create new connections between correlated neurons
            # Find neurons with correlated activity
            for i in range(self.hidden_dim):
                for j in range(self.hidden_dim):
                    if i != j and not self.connectivity[i, j]:
                        # Check if these neurons should form a connection
                        # Based on causal relationship or correlation
                        if self.causal_matrix[i, j] > self.synapse_formation_threshold:
                            # Form new connection
                            self.connectivity[i, j] = True
                            # Initialize with small weight
                            self.fast_weights[i, j] = 0.01 * cp.random.randn()
                            # Apply Dale's principle
                            if self.neuron_type[i] < 0:  # Inhibitory neuron
                                self.fast_weights[i, j] = -abs(self.fast_weights[i, j])
                            else:  # Excitatory neuron
                                self.fast_weights[i, j] = abs(self.fast_weights[i, j])
            
            # 2. Synapse elimination - remove unused connections
            # Find connections with consistently low activity
            for i in range(self.hidden_dim):
                for j in range(self.hidden_dim):
                    if self.connectivity[i, j]:
                        # Check if this connection is consistently unused
                        if abs(self.fast_weights[i, j]) < self.synapse_elimination_threshold:
                            # Probabilistic pruning - higher chance to prune weak connections
                            if cp.random.random() < 0.2:
                                # Eliminate connection
                                self.connectivity[i, j] = False
                                # Zero out weight
                                self.fast_weights[i, j] = 0.0
            
            # 3. Adjust elimination threshold based on developmental stage
            if self.developmental_stage == 'critical_period':
                # More lenient pruning during critical period
                self.synapse_elimination_threshold = 0.05
            elif self.developmental_stage == 'refinement':
                # More selective pruning during refinement
                self.synapse_elimination_threshold = 0.08
            else:  # mature
                # Very selective pruning in mature stage
                self.synapse_elimination_threshold = 0.1
            
            return True
            
        except Exception as e:
            print(f"Error in structural synaptic plasticity: {e}")
            return False
    
    def _synchronize_tensor_dimensions(self):
        """Ensure all tensors have consistent dimensions based on current network size"""
        try:
            # Check key tensors and resize if necessary
            tensors_to_check = [
                # State vectors
                ('h', 0), ('h_prev', 0), ('dh', 0),
                # Traces
                ('pre_synaptic_trace', 0), ('post_synaptic_trace', 0),
                # Neuron properties
                ('neuron_type', 0), ('activation_types', 0), 
                ('activation_gain', 0), ('activation_threshold', 0),
                ('energy', 0), ('health', 0), ('plasticity_rates', 0),
                ('neuron_utility', 0), ('firing_threshold', 0),
                # NEW: Neuronal autonomy
                ('local_objectives', 0), ('intrinsic_motivation', 0),
                ('neuron_learning_rates', 0), ('prediction_targets', 0),
                # NEW: Hierarchical organization
                ('neuron_layer', 0), ('phase', 0), ('frequency', 0)
            ]
            
            changed = False
            
            # Check and resize 1D arrays
            for name, dim in tensors_to_check:
                if hasattr(self, name):
                    tensor = getattr(self, name)
                    if tensor.shape[dim] != self.hidden_dim:
                        print(f"Resynchronizing {name}: {tensor.shape} to match hidden_dim={self.hidden_dim}")
                        resized = self._resize_array(tensor, self.hidden_dim)
                        setattr(self, name, resized)
                        changed = True
            
            # Check and resize 2D weight matrices
            matrices_to_check = [
                # Recurrent weights
                'Uz', 'Ur', 'Uh',
                # Fast and slow weights
                'fast_weights', 'slow_weights',
                # Eligibility traces
                'eligibility_traces',
                # Information metrics
                'mutual_information',
                # NEW: Structural connectivity
                'connectivity',
                # NEW: Lateral inhibition
                'lateral_inhibition',
                # NEW: Coherence
                'coherence',
                # NEW: Causal matrix
                'causal_matrix',
                # NEW: Prediction weights
                'prediction_weights',
                # NEW: Feedback weights
                'feedback_weights'
            ]
            
            for name in matrices_to_check:
                if hasattr(self, name):
                    matrix = getattr(self, name)
                    if matrix.shape != (self.hidden_dim, self.hidden_dim):
                        print(f"Resynchronizing matrix {name}: {matrix.shape} to {(self.hidden_dim, self.hidden_dim)}")
                        new_matrix = cp.zeros((self.hidden_dim, self.hidden_dim))
                        # Copy existing values up to min dimensions
                        min_rows = min(matrix.shape[0], self.hidden_dim)
                        min_cols = min(matrix.shape[1], self.hidden_dim)
                        new_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
                        setattr(self, name, new_matrix)
                        changed = True
            
            # Verify input weight matrices (hidden_dim x input_dim)
            input_matrices = ['Wz', 'Wr', 'Wh']
            for name in input_matrices:
                if hasattr(self, name):
                    matrix = getattr(self, name)
                    if matrix.shape[0] != self.hidden_dim:
                        print(f"Resynchronizing input matrix {name}: {matrix.shape} to match hidden_dim={self.hidden_dim}")
                        new_matrix = cp.zeros((self.hidden_dim, self.input_dim))
                        # Copy existing values
                        min_rows = min(matrix.shape[0], self.hidden_dim)
                        min_cols = min(matrix.shape[1], self.input_dim)
                        new_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
                        setattr(self, name, new_matrix)
                        changed = True
            
            # Update activity histories (dicts containing arrays)
            if hasattr(self, 'activity_history'):
                for key in self.activity_history:
                    history = self.activity_history[key]
                    if history.shape[0] != self.hidden_dim:
                        print(f"Resynchronizing activity_history[{key}]: {history.shape} to match hidden_dim={self.hidden_dim}")
                        self.activity_history[key] = self._resize_array(history, self.hidden_dim)
                        changed = True
            
            # NEW: Check and resize ensemble membership
            if hasattr(self, 'ensemble_membership'):
                if self.ensemble_membership.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing ensemble_membership: {self.ensemble_membership.shape} to match hidden_dim={self.hidden_dim}")
                    new_matrix = cp.zeros((self.hidden_dim, self.max_ensembles))
                    # Copy existing values
                    min_rows = min(self.ensemble_membership.shape[0], self.hidden_dim)
                    min_cols = min(self.ensemble_membership.shape[1], self.max_ensembles)
                    new_matrix[:min_rows, :min_cols] = self.ensemble_membership[:min_rows, :min_cols]
                    self.ensemble_membership = new_matrix
                    changed = True
            
            # NEW: Check and resize dendritic weights
            if hasattr(self, 'dendritic_weights'):
                if self.dendritic_weights.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing dendritic_weights: {self.dendritic_weights.shape} to match hidden_dim={self.hidden_dim}")
                    new_tensor = cp.zeros((self.hidden_dim, self.num_dendrites, self.input_dim))
                    # Copy existing values
                    min_dim0 = min(self.dendritic_weights.shape[0], self.hidden_dim)
                    min_dim1 = min(self.dendritic_weights.shape[1], self.num_dendrites)
                    min_dim2 = min(self.dendritic_weights.shape[2], self.input_dim)
                    new_tensor[:min_dim0, :min_dim1, :min_dim2] = self.dendritic_weights[:min_dim0, :min_dim1, :min_dim2]
                    self.dendritic_weights = new_tensor
                    changed = True
            
            # NEW: Check and resize dendritic activations
            if hasattr(self, 'dendritic_activations'):
                if self.dendritic_activations.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing dendritic_activations: {self.dendritic_activations.shape} to match hidden_dim={self.hidden_dim}")
                    new_matrix = cp.zeros((self.hidden_dim, self.num_dendrites))
                    # Copy existing values
                    min_rows = min(self.dendritic_activations.shape[0], self.hidden_dim)
                    min_cols = min(self.dendritic_activations.shape[1], self.num_dendrites)
                    new_matrix[:min_rows, :min_cols] = self.dendritic_activations[:min_rows, :min_cols]
                    self.dendritic_activations = new_matrix
                    changed = True
            
            # NEW: Check and resize neuromodulator sensitivity
            if hasattr(self, 'neuromod_sensitivity'):
                if self.neuromod_sensitivity.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing neuromod_sensitivity: {self.neuromod_sensitivity.shape} to match hidden_dim={self.hidden_dim}")
                    new_matrix = cp.ones((self.hidden_dim, len(self.modulation)))
                    # Copy existing values
                    min_rows = min(self.neuromod_sensitivity.shape[0], self.hidden_dim)
                    min_cols = min(self.neuromod_sensitivity.shape[1], len(self.modulation))
                    new_matrix[:min_rows, :min_cols] = self.neuromod_sensitivity[:min_rows, :min_cols]
                    self.neuromod_sensitivity = new_matrix
                    changed = True
            
            # NEW: Check and resize activation parameters
            if hasattr(self, 'activation_params'):
                if self.activation_params.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing activation_params: {self.activation_params.shape} to match hidden_dim={self.hidden_dim}")
                    new_matrix = cp.zeros((self.hidden_dim, 3))  # 3 parameters per neuron
                    # Copy existing values
                    min_rows = min(self.activation_params.shape[0], self.hidden_dim)
                    min_cols = min(self.activation_params.shape[1], 3)
                    new_matrix[:min_rows, :min_cols] = self.activation_params[:min_rows, :min_cols]
                    self.activation_params = new_matrix
                    changed = True
            
            # NEW: Check and resize specialization
            if hasattr(self, 'specialization'):
                if self.specialization.shape[0] != self.hidden_dim:
                    print(f"Resynchronizing specialization: {self.specialization.shape} to match hidden_dim={self.hidden_dim}")
                    new_matrix = cp.zeros((self.hidden_dim, self.input_dim))
                    # Copy existing values
                    min_rows = min(self.specialization.shape[0], self.hidden_dim)
                    min_cols = min(self.specialization.shape[1], self.input_dim)
                    new_matrix[:min_rows, :min_cols] = self.specialization[:min_rows, :min_cols]
                    self.specialization = new_matrix
                    changed = True
            
            if changed:
                print(f"Network tensor dimensions resynchronized to match current hidden_dim={self.hidden_dim}")
            
            return changed
        
        except Exception as e:
            print(f"Error during tensor resynchronization: {e}")
            return False
    
    def _consolidate_memory(self):
        """Transfer patterns from fast to slow weights based on repeated activation with dimension safeguards"""
        try:
            # Ensure tensor dimensions are synchronized
            self._synchronize_tensor_dimensions()
            
            # Verify specific matrix dimensions for this operation
            if self.fast_weights.shape != (self.hidden_dim, self.hidden_dim):
                self.fast_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
                
            if self.slow_weights.shape != (self.hidden_dim, self.hidden_dim):
                self.slow_weights = cp.zeros((self.hidden_dim, self.hidden_dim))
                
            if self.eligibility_traces.shape != (self.hidden_dim, self.hidden_dim):
                self.eligibility_traces = cp.zeros((self.hidden_dim, self.hidden_dim))
            
            # NEW: Apply structural connectivity constraints
            # Only consolidate existing connections
            consolidation_mask = cp.abs(self.eligibility_traces) > self.consolidation_threshold
            consolidation_mask = consolidation_mask * self.connectivity
            
            # Transfer a small portion of fast weights to slow weights at masked positions
            transfer = self.consolidation_rate * self.fast_weights * consolidation_mask
            self.slow_weights += transfer
            
            # Decay fast weights where consolidation occurred to prevent redundancy
            self.fast_weights -= transfer
            
            # Apply weight normalization to slow weights for stability
            norm = cp.sqrt(cp.sum(self.slow_weights**2, axis=1, keepdims=True))
            norm = cp.clip(norm, 1e-10, None)  # Avoid division by zero
            self.slow_weights = self.slow_weights / norm
            
            # NEW: Adjust consolidation threshold based on developmental stage
            if self.developmental_stage == 'critical_period':
                # Higher threshold during critical period (less consolidation)
                self.consolidation_threshold = 0.6
            elif self.developmental_stage == 'refinement':
                # Moderate threshold during refinement
                self.consolidation_threshold = 0.5
            else:  # mature
                # Lower threshold in mature stage (more consolidation)
                self.consolidation_threshold = 0.4
            
            return True
            
        except Exception as e:
            print(f"Error in memory consolidation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_task_driven_neurogenesis(self, prediction_error=None):
        """
        NEW: Apply task-driven neurogenesis and competitive survival
        Add neurons based on task performance and prune underperforming ones
        """
        # Skip if disabled
        if not self.enable_growth and not self.enable_pruning:
            return False
            
        try:
            # Calculate current spectral radius
            try:
                # Combine weights for analysis
                combined_weights = self.Uh + self.slow_weights
                eigenvalues = cp.linalg.eigvals(combined_weights)
                self.criticality_metrics['spectral_radius'] = float(cp.max(cp.abs(eigenvalues)))
                
                # Calculate eigenvalue gap (related to mixing time)
                sorted_eigs = cp.sort(cp.abs(eigenvalues))[::-1]
                if len(sorted_eigs) >= 2:
                    self.criticality_metrics['eigenvalue_gap'] = float(sorted_eigs[0] - sorted_eigs[1])
            except:
                # Fallback estimation
                self.criticality_metrics['spectral_radius'] = float(cp.mean(cp.abs(combined_weights)))
                self.criticality_metrics['eigenvalue_gap'] = 0.1
            
            # Distance from edge of chaos
            eoc_distance = abs(self.criticality_metrics['spectral_radius'] - self.target_spectral_radius)
            self.criticality_metrics['edge_of_chaos_distance'] = eoc_distance
            
            # NEW: Task-driven growth determination
            growth_needed = False
            
            # Check if we have error history
            if len(self.error_history) > 10:
                # Calculate recent error trend
                recent_errors = list(self.error_history)[-10:]
                error_mean = np.mean(recent_errors)
                
                # Growth is needed if:
                # 1. Error is consistently high
                # 2. We're not in a cooling down period
                # 3. Network size is below maximum
                if (error_mean > 0.3 and 
                    self.step_counter - self.last_growth_step >= self.growth_cooldown and
                    self.hidden_dim < self.max_neurons):
                    growth_needed = True
                    
                    # Determine growth amount based on error
                    growth_amount = int(min(
                        self.max_growth_per_phase * self.current_growth_rate,
                        max(1, int(error_mean * 10))  # Scale with error
                    ))
                    
                    # Adjust for developmental stage
                    if self.developmental_stage == 'critical_period':
                        # More growth during critical period
                        growth_amount = int(growth_amount * 1.5)
                    elif self.developmental_stage == 'mature':
                        # Less growth in mature stage
                        growth_amount = max(1, growth_amount // 2)
            
            # NEW: Competitive survival determination
            # Calculate survival scores based on utility and error contribution
            self.survival_score = self.neuron_utility.copy()
            
            # Adjust survival score based on error contribution if available
            if prediction_error is not None and prediction_error > 0:
                # Neurons with higher activity get lower survival score when error is high
                # This encourages pruning neurons that contribute to errors
                self.survival_score -= self.competition_strength * cp.abs(self.h) * prediction_error
            
            # Identify neurons for pruning based on survival score
            prune_candidates = cp.where(self.survival_score < 0.05)[0]
            pruning_needed = len(prune_candidates) > 0 and self.hidden_dim > self.min_neurons
            
            # Apply growth if needed and enabled
            if growth_needed and self.enable_growth:
                result = self._add_neurons(growth_amount)
                
                # Memory cleanup
                try:
                    # Force garbage collection to clean up temporary arrays
                    gc.collect()
                    
                    # Clear CuPy memory cache if available
                    if hasattr(cp, 'get_default_memory_pool'):
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                except Exception as e:
                    print(f"Warning: Memory cleanup failed: {e}")
                
                return result
                
            # Apply pruning if needed and enabled
            elif pruning_needed and self.enable_pruning:
                result = self._prune_neurons(prune_candidates)
                
                # Memory cleanup
                try:
                    # Force garbage collection to clean up temporary arrays
                    gc.collect()
                    
                    # Clear CuPy memory cache if available
                    if hasattr(cp, 'get_default_memory_pool'):
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                except Exception as e:
                    print(f"Warning: Memory cleanup failed: {e}")
                
                return result
                
            return False
                
        except Exception as e:
            print(f"Error in task-driven neurogenesis: {e}")
            return False
    
    def _add_neurons(self, num_to_add):
        """
        Add new neurons to the network with initialization
        
        NEW: Implements more sophisticated neuron initialization based on existing knowledge
        """
        try:
            # Limit growth to maximum size
            num_to_add = min(num_to_add, self.max_neurons - self.hidden_dim)
            if num_to_add <= 0:
                return False
                
            # Track growth
            self.last_growth_step = self.step_counter
            self.total_grown += num_to_add
            
            # Store old dimensions for reference
            old_dim = self.hidden_dim
            new_dim = old_dim + num_to_add
            self.hidden_dim = new_dim
            
            # Initialize expanded arrays
            # Neuron type (E/I balance)
            new_neuron_type = cp.ones(num_to_add)  # Default to excitatory
            inhibitory_count = int(0.2 * num_to_add)  # Maintain 80/20 E/I ratio
            if inhibitory_count > 0:
                inhibitory_idx = cp.random.choice(num_to_add, size=inhibitory_count, replace=False)
                new_neuron_type[inhibitory_idx] = -1
                
            # Activation diversity
            new_activation_types = cp.zeros(num_to_add, dtype=np.int32)
            sigmoid_count = int(0.2 * num_to_add)
            relu_count = int(0.1 * num_to_add)
            adaptive_count = int(0.2 * num_to_add)  # More adaptive neurons in new growth
            
            if sigmoid_count > 0:
                sigmoid_idx = cp.random.choice(num_to_add, size=sigmoid_count, replace=False)
                new_activation_types[sigmoid_idx] = 1
                
                remaining = np.setdiff1d(cp.asnumpy(cp.arange(num_to_add)), cp.asnumpy(sigmoid_idx))
                
                if relu_count > 0 and len(remaining) > 0:
                    # Ensure no overlap between neuron types
                    relu_idx = cp.asarray(np.random.choice(remaining, 
                                                         size=min(relu_count, len(remaining)), 
                                                         replace=False))
                    new_activation_types[relu_idx] = 2
                    
                    remaining = np.setdiff1d(cp.asnumpy(remaining), cp.asnumpy(relu_idx))
                
                if adaptive_count > 0 and len(remaining) > 0:
                    adaptive_idx = cp.asarray(np.random.choice(remaining,
                                                             size=min(adaptive_count, len(remaining)),
                                                             replace=False))
                    new_activation_types[adaptive_idx] = 3
            
            # Resize all neuron-specific arrays
            self.neuron_type = cp.concatenate([self.neuron_type, new_neuron_type])
            self.activation_types = cp.concatenate([self.activation_types, new_activation_types])
            self.activation_gain = cp.concatenate([self.activation_gain, cp.ones(num_to_add)])
            self.activation_threshold = cp.concatenate([self.activation_threshold, cp.zeros(num_to_add)])
            
            # NEW: Initialize activation parameters for adaptive neurons
            new_activation_params = cp.zeros((num_to_add, 3))
            # Initialize with small random values
            new_activation_params[:, 0] = cp.random.randn(num_to_add) * 0.1  # Shape parameter
            new_activation_params[:, 1] = cp.random.randn(num_to_add) * 0.1  # Threshold parameter
            new_activation_params[:, 2] = cp.random.randn(num_to_add) * 0.1  # Saturation parameter
            self.activation_params = cp.concatenate([self.activation_params, new_activation_params])
            
            # Timescale diversity
            new_tau = 10.0 + cp.random.randn(num_to_add) * 5.0
            new_tau = cp.maximum(new_tau, 1.0)  # Ensure positive timescales
            self.tau = cp.concatenate([self.tau, new_tau])
            
            # NEW: Initialize specialization for new neurons
            # Some neurons specialize in patterns similar to existing ones
            new_specialization = cp.zeros((num_to_add, self.input_dim))
            
            # For half the neurons, initialize with random specialization
            random_half = num_to_add // 2
            new_specialization[:random_half] = cp.random.randn(random_half, self.input_dim) * 0.1
            
            # For the other half, copy specialization from successful neurons with small variations
            if old_dim > 0:
                # Find successful neurons (high utility)
                success_threshold = cp.percentile(self.neuron_utility, 80)
                successful_neurons = cp.where(self.neuron_utility > success_threshold)[0]
                
                if len(successful_neurons) > 0:
                    # Sample from successful neurons with replacement
                    for i in range(random_half, num_to_add):
                        # Pick a random successful neuron
                        template_idx = cp.random.choice(successful_neurons)
                        # Copy its specialization with small variations
                        new_specialization[i] = self.specialization[template_idx] + cp.random.randn(self.input_dim) * 0.05
            
            self.specialization = cp.concatenate([self.specialization, new_specialization])
            
            # Resize weight matrices (with initialization)
            scale_i = 0.1 / cp.sqrt(self.input_dim)  # For input weights
            scale_r = 0.1 / cp.sqrt(new_dim)        # For recurrent weights
            
            # NEW: Initialize input weights based on specialization
            new_Wz = cp.zeros((num_to_add, self.input_dim))
            new_Wr = cp.zeros((num_to_add, self.input_dim))
            new_Wh = cp.zeros((num_to_add, self.input_dim))
            
            for i in range(num_to_add):
                # Use specialization to guide weight initialization
                # Neurons respond more strongly to their specialized patterns
                new_Wz[i] = new_specialization[i] + cp.random.randn(self.input_dim) * scale_i * 0.5
                new_Wr[i] = new_specialization[i] + cp.random.randn(self.input_dim) * scale_i * 0.5
                new_Wh[i] = new_specialization[i] + cp.random.randn(self.input_dim) * scale_i * 0.5
            
            self.Wz = cp.concatenate([self.Wz, new_Wz], axis=0)
            self.Wr = cp.concatenate([self.Wr, new_Wr], axis=0)
            self.Wh = cp.concatenate([self.Wh, new_Wh], axis=0)
            
            # Recurrent weights - more complex due to both dimensions
            # Extend existing weights with zeros first
            Uz_extended = cp.zeros((new_dim, old_dim))
            Uz_extended[:old_dim, :] = self.Uz
            
            Ur_extended = cp.zeros((new_dim, old_dim))
            Ur_extended[:old_dim, :] = self.Ur
            
            Uh_extended = cp.zeros((new_dim, old_dim))
            Uh_extended[:old_dim, :] = self.Uh
            
            # Create new columns for all neurons
            new_cols_Uz = cp.random.randn(new_dim, num_to_add) * scale_r
            new_cols_Ur = cp.random.randn(new_dim, num_to_add) * scale_r
            new_cols_Uh = cp.random.randn(new_dim, num_to_add) * scale_r
            
            # Apply Dale's principle to new neurons
            for i in range(old_dim, new_dim):
                if self.neuron_type[i] < 0:  # Inhibitory neuron
                    new_cols_Uz[i, :] = -cp.abs(new_cols_Uz[i, :])
                    new_cols_Ur[i, :] = -cp.abs(new_cols_Ur[i, :])
                    new_cols_Uh[i, :] = -cp.abs(new_cols_Uh[i, :])
                else:  # Excitatory neuron
                    new_cols_Uz[i, :] = cp.abs(new_cols_Uz[i, :])
                    new_cols_Ur[i, :] = cp.abs(new_cols_Ur[i, :])
                    new_cols_Uh[i, :] = cp.abs(new_cols_Uh[i, :])
            
            # Combine to form new weight matrices
            self.Uz = cp.concatenate([Uz_extended, new_cols_Uz], axis=1)
            self.Ur = cp.concatenate([Ur_extended, new_cols_Ur], axis=1)
            self.Uh = cp.concatenate([Uh_extended, new_cols_Uh], axis=1)
            
            # Extend bias vectors
            self.bz = cp.concatenate([self.bz, cp.zeros(num_to_add)])
            self.br = cp.concatenate([self.br, cp.zeros(num_to_add)])
            self.bh = cp.concatenate([self.bh, cp.zeros(num_to_add)])
            
            # Extend state vectors
            self.h = cp.concatenate([self.h, cp.zeros(num_to_add)])
            self.h_prev = cp.concatenate([self.h_prev, cp.zeros(num_to_add)])
            self.dh = cp.concatenate([self.dh, cp.zeros(num_to_add)])
            
            # Extend activity history
            for key in self.activity_history:
                self.activity_history[key] = cp.concatenate([
                    self.activity_history[key], cp.zeros(num_to_add)
                ])
            
            # Extend fast and slow weights
            fast_ext = cp.zeros((new_dim, new_dim))
            fast_ext[:old_dim, :old_dim] = self.fast_weights
            self.fast_weights = fast_ext
            
            slow_ext = cp.zeros((new_dim, new_dim))
            slow_ext[:old_dim, :old_dim] = self.slow_weights
            self.slow_weights = slow_ext
            
            # NEW: Extend structural connectivity matrix
            connectivity_ext = cp.ones((new_dim, new_dim), dtype=bool)
            connectivity_ext[:old_dim, :old_dim] = self.connectivity
            self.connectivity = connectivity_ext
            
            # NEW: Initialize sparse connectivity for new neurons
            # New neurons start with connections to only a subset of existing neurons
            for i in range(old_dim, new_dim):
                # Connect to ~30% of existing neurons randomly
                connect_count = max(1, int(0.3 * old_dim))
                connect_idx = cp.random.choice(old_dim, size=connect_count, replace=False)
                
                # Initialize all connections to False
                self.connectivity[i, :old_dim] = False
                self.connectivity[:old_dim, i] = False
                
                # Enable only selected connections
                self.connectivity[i, connect_idx] = True
                self.connectivity[connect_idx, i] = True
                
                # Always connect new neurons to each other (forming a new subnetwork)
                self.connectivity[old_dim:new_dim, old_dim:new_dim] = True
            
            # Extend other neuron-specific arrays
            self.pre_synaptic_trace = cp.concatenate([self.pre_synaptic_trace, cp.zeros(num_to_add)])
            self.post_synaptic_trace = cp.concatenate([self.post_synaptic_trace, cp.zeros(num_to_add)])
            self.firing_threshold = cp.concatenate([self.firing_threshold, cp.ones(num_to_add) * 0.1])
            self.energy = cp.concatenate([self.energy, cp.ones(num_to_add)])
            self.health = cp.concatenate([self.health, cp.ones(num_to_add)])
            
            # NEW: Higher initial plasticity for new neurons
            new_plasticity = cp.ones(num_to_add) * 0.002  # Higher than default
            self.plasticity_rates = cp.concatenate([self.plasticity_rates, new_plasticity])
            
            self.neuron_utility = cp.concatenate([self.neuron_utility, cp.ones(num_to_add) * 0.1])
            
            # NEW: Initialize neuronal autonomy parameters
            self.local_objectives = cp.concatenate([self.local_objectives, cp.ones(num_to_add) * 0.5])
            self.intrinsic_motivation = cp.concatenate([self.intrinsic_motivation, cp.ones(num_to_add) * 0.8])  # High initial curiosity
            self.neuron_learning_rates = cp.concatenate([self.neuron_learning_rates, cp.ones(num_to_add) * 0.015])  # Higher initial learning rate
            self.prediction_targets = cp.concatenate([self.prediction_targets, cp.zeros(num_to_add)])
            
            # NEW: Initialize hierarchical organization
            # Assign new neurons to layers
            new_layer_assignments = cp.zeros(num_to_add, dtype=np.int32)
            # Distribute across layers with bias toward lower layers
            layer_probs = [0.5, 0.3, 0.2]  # More neurons in lower layers
            for i in range(num_to_add):
                new_layer_assignments[i] = cp.random.choice(self.num_layers, p=layer_probs)
            
            self.neuron_layer = cp.concatenate([self.neuron_layer, new_layer_assignments])
            
            # NEW: Initialize ensemble parameters
            self.phase = cp.concatenate([self.phase, cp.random.uniform(0, 2*np.pi, size=num_to_add)])
            self.frequency = cp.concatenate([self.frequency, cp.ones(num_to_add) * 0.1])
            
            # NEW: Initialize dendritic compartments
            new_dendritic_weights = cp.random.randn(num_to_add, self.num_dendrites, self.input_dim) * 0.1
            self.dendritic_weights = cp.concatenate([self.dendritic_weights, new_dendritic_weights], axis=0)
            
            new_dendritic_activations = cp.zeros((num_to_add, self.num_dendrites))
            self.dendritic_activations = cp.concatenate([self.dendritic_activations, new_dendritic_activations], axis=0)
            
            # NEW: Initialize neuromodulator sensitivity
            new_neuromod_sensitivity = cp.ones((num_to_add, len(self.modulation)))
            # Add some variability to sensitivity
            new_neuromod_sensitivity += cp.random.randn(num_to_add, len(self.modulation)) * 0.2
            new_neuromod_sensitivity = cp.clip(new_neuromod_sensitivity, 0.5, 1.5)
            self.neuromod_sensitivity = cp.concatenate([self.neuromod_sensitivity, new_neuromod_sensitivity], axis=0)
            
            # Decay growth rate for future growth phases
            self.current_growth_rate *= self.growth_rate_decay
            
            print(f"Added {num_to_add} neurons. Network now has {self.hidden_dim} neurons.")
            return True
            
        except Exception as e:
            print(f"Error adding neurons: {e}")
            return False
    
    def _prune_neurons(self, candidates):
        """
        Remove low utility neurons from the network
        
        NEW: Implements competitive survival based on neuron utility and task performance
        """
        try:
            # Determine how many neurons to prune
            num_to_prune = min(len(candidates), int(0.05 * self.hidden_dim))
            
            # Ensure we don't go below minimum size
            if self.hidden_dim - num_to_prune < self.min_neurons:
                num_to_prune = self.hidden_dim - self.min_neurons
                
            if num_to_prune <= 0:
                return False
            
            # NEW: Select neurons to prune based on survival score
            survival_scores = self.survival_score[candidates]
            ranked_indices = cp.argsort(survival_scores)
            prune_indices = candidates[ranked_indices[:num_to_prune]]
            
            # Create mask for neurons to keep
            keep_mask = cp.ones(self.hidden_dim, dtype=bool)
            keep_mask[prune_indices] = False
            
            # Update dimension
            old_dim = self.hidden_dim
            self.hidden_dim -= num_to_prune
            
            # Prune neuron-specific arrays
            self.neuron_type = self.neuron_type[keep_mask]
            self.activation_types = self.activation_types[keep_mask]
            self.activation_gain = self.activation_gain[keep_mask]
            self.activation_threshold = self.activation_threshold[keep_mask]
            self.tau = self.tau[keep_mask]
            
            # NEW: Prune activation parameters
            self.activation_params = self.activation_params[keep_mask]
            
            # Prune weight matrices
            self.Wz = self.Wz[keep_mask]
            self.Wr = self.Wr[keep_mask]
            self.Wh = self.Wh[keep_mask]
            
            # Recurrent weights require pruning both dims
            self.Uz = self.Uz[keep_mask][:, keep_mask]
            self.Ur = self.Ur[keep_mask][:, keep_mask]
            self.Uh = self.Uh[keep_mask][:, keep_mask]
            
            # Prune bias vectors
            self.bz = self.bz[keep_mask]
            self.br = self.br[keep_mask]
            self.bh = self.bh[keep_mask]
            
            # Prune state vectors
            self.h = self.h[keep_mask]
            self.h_prev = self.h_prev[keep_mask]
            self.dh = self.dh[keep_mask]
            
            # Prune activity history
            for key in self.activity_history:
                self.activity_history[key] = self.activity_history[key][keep_mask]
            
            # Prune fast and slow weights
            self.fast_weights = self.fast_weights[keep_mask][:, keep_mask]
            self.slow_weights = self.slow_weights[keep_mask][:, keep_mask]
            
            # NEW: Prune structural connectivity
            self.connectivity = self.connectivity[keep_mask][:, keep_mask]
            
            # Prune other neuron-specific arrays
            self.pre_synaptic_trace = self.pre_synaptic_trace[keep_mask]
            self.post_synaptic_trace = self.post_synaptic_trace[keep_mask]
            self.firing_threshold = self.firing_threshold[keep_mask]
            self.energy = self.energy[keep_mask]
            self.health = self.health[keep_mask]
            self.plasticity_rates = self.plasticity_rates[keep_mask]
            self.neuron_utility = self.neuron_utility[keep_mask]
            
            # NEW: Prune neuronal autonomy parameters
            self.local_objectives = self.local_objectives[keep_mask]
            self.intrinsic_motivation = self.intrinsic_motivation[keep_mask]
            self.neuron_learning_rates = self.neuron_learning_rates[keep_mask]
            self.prediction_targets = self.prediction_targets[keep_mask]
            
            # NEW: Prune hierarchical organization
            self.neuron_layer = self.neuron_layer[keep_mask]
            
            # NEW: Prune ensemble parameters
            self.phase = self.phase[keep_mask]
            self.frequency = self.frequency[keep_mask]
            
            # NEW: Prune dendritic compartments
            self.dendritic_weights = self.dendritic_weights[keep_mask]
            self.dendritic_activations = self.dendritic_activations[keep_mask]
            
            # NEW: Prune neuromodulator sensitivity
            self.neuromod_sensitivity = self.neuromod_sensitivity[keep_mask]
            
            # NEW: Prune specialization
            self.specialization = self.specialization[keep_mask]
            
            # Track pruning
            self.total_pruned += num_to_prune
            
            # NEW: Redistribute resources to surviving neurons
            # Increase energy and plasticity for survivors
            self.energy *= 1.1
            self.energy = cp.clip(self.energy, 0.0, 1.0)
            
            # Increase plasticity temporarily to allow adaptation
            self.plasticity_rates *= 1.1
            self.plasticity_rates = cp.clip(self.plasticity_rates, 0.0001, 0.02)
            
            print(f"Pruned {num_to_prune} low-utility neurons. Network now has {self.hidden_dim} neurons.")
            return True
            
        except Exception as e:
            print(f"Error pruning neurons: {e}")
            return False
    
    def _update_information_metrics(self):
        """Update information-theoretic metrics for plasticity guidance with dimension safeguards"""
        try:
            # CRITICAL: First synchronize all tensor dimensions
            self._synchronize_tensor_dimensions()
            
            # Initialize information_content array if missing or wrong size
            if not hasattr(self, 'information_content') or self.information_content.shape[0] != self.hidden_dim:
                old_info = getattr(self, 'information_content', None)
                self.information_content = cp.zeros(self.hidden_dim)
                # Copy existing values if possible
                if old_info is not None:
                    copy_size = min(old_info.shape[0], self.hidden_dim)
                    self.information_content[:copy_size] = old_info[:copy_size]
            
            # Initialize mutual_information matrix if missing or wrong size
            if not hasattr(self, 'mutual_information') or self.mutual_information.shape != (self.hidden_dim, self.hidden_dim):
                old_mutual = getattr(self, 'mutual_information', None)
                self.mutual_information = cp.zeros((self.hidden_dim, self.hidden_dim))
                # Copy existing values if possible
                if old_mutual is not None:
                    copy_rows = min(old_mutual.shape[0], self.hidden_dim)
                    copy_cols = min(old_mutual.shape[1], self.hidden_dim)
                    self.mutual_information[:copy_rows, :copy_cols] = old_mutual[:copy_rows, :copy_cols]
            
            # Calculate activity distributions for information analysis
            # Ensure we have sufficient activity history
            if len(self.recent_activations) < 50:
                return False
                
            # Generate activity distribution data from recent activity
            recent_acts = list(self.recent_activations)[-50:]
            
            # Estimate complexity using activity distribution
            # Simplified information content estimation with boundary checks
            try:
                # Access activity history safely
                if not hasattr(self, 'activity_history') or 'medium' not in self.activity_history:
                    return False
                    
                medium_activity = self.activity_history['medium']
                if medium_activity.shape[0] != self.hidden_dim:
                    # Resize to match current dimensions
                    medium_activity = self._resize_array(medium_activity, self.hidden_dim)
                    self.activity_history['medium'] = medium_activity
                
                # Define bins for histogram
                bins = 10
                
                # For each neuron, calculate entropy of its firing pattern with explicit bounds checking
                for i in range(self.hidden_dim):
                    # Safely access activity for this neuron
                    if i >= medium_activity.shape[0]:
                        continue  # Skip if index out of bounds
                        
                    # Get activity samples for this neuron as a numpy array for histogram calculation
                    samples = cp.asnumpy(medium_activity[i])
                    if not isinstance(samples, np.ndarray) or samples.size == 0:
                        continue  # Skip if empty or invalid
                    
                    # Safely calculate histogram
                    try:
                        hist, _ = np.histogram(samples, bins=bins, range=(0, 1), density=True)
                        # Calculate entropy with safety checks
                        if hist.size > 0 and np.sum(hist) > 0:
                            hist = hist / np.sum(hist)
                            entropy = -np.sum(hist * np.log2(hist + 1e-10))
                            # Safely update information content
                            if i < self.information_content.shape[0]:
                                self.information_content[i] = entropy
                    except Exception as e:
                        print(f"Error calculating histogram for neuron {i}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error in information content calculation: {e}")
                
            # Update mutual information between neuron pairs (simplified)
            # This is expensive, so do it less frequently
            if self.step_counter % 200 == 0:
                try:
                    # Use a small random sample of neurons for mutual information calculation
                    # to reduce computational burden
                    sample_size = min(max(5, self.hidden_dim // 4), self.hidden_dim)
                    
                    if sample_size >= 3:  # Need at least 3 for meaningful correlation
                        # Safely sample neurons
                        try:
                            indices = cp.random.choice(self.hidden_dim, size=sample_size, replace=False)
                            indices = indices[indices < self.activity_history['medium'].shape[0]]
                            
                            if len(indices) < 3:  # Too few valid indices
                                return False
                                
                            # Get activities for sampled neurons
                            activities = self.activity_history['medium'][indices]
                            
                            # Validate shape of activities
                            if len(activities.shape) == 1:
                                # Reshape to 2D if needed
                                activities = activities.reshape(len(activities), 1)
                                
                            # Skip if we don't have enough data points
                            if activities.shape[1] < 2:
                                return False
                                
                            # Use numpy for more stable correlation
                            np_activities = cp.asnumpy(activities)
                            
                            # Calculate correlation safely
                            try:
                                corr = np.corrcoef(np_activities)
                                
                                # Handle case where corr is a scalar or 1D
                                if np.isscalar(corr) or len(corr.shape) < 2:
                                    return False
                                    
                                # Update mutual information matrix with explicit bounds checking
                                corr_size = min(corr.shape[0], len(indices))
                                for i in range(corr_size):
                                    if indices[i] >= self.mutual_information.shape[0]:
                                        continue
                                        
                                    for j in range(corr_size):
                                        if indices[j] >= self.mutual_information.shape[1]:
                                            continue
                                            
                                        # Only update if indices are valid
                                        if (not np.isnan(corr[i, j]) and 
                                            indices[i] < self.mutual_information.shape[0] and 
                                            indices[j] < self.mutual_information.shape[1]):
                                            
                                            self.mutual_information[indices[i], indices[j]] = abs(corr[i, j])
                            except Exception as e:
                                print(f"Error in correlation calculation: {e}")
                                return False
                                
                        except Exception as e:
                            print(f"Error sampling neurons for mutual information: {e}")
                            return False
                except Exception as e:
                    print(f"Error in mutual information calculation: {e}")
                
            return True
            
        except Exception as e:
            print(f"Error updating information metrics: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def enable_structural_plasticity(self, enable_flag: bool = True): # Added boolean argument with a default
        """
        Enables or disables structural plasticity (both growth and pruning).
        If enable_flag is True (default), enables plasticity.
        If enable_flag is False, disables plasticity.
        """
        self.enable_growth = enable_flag
        self.enable_pruning = enable_flag # Typically, growth and pruning are controlled together by "structural plasticity"
        
        if enable_flag:
            print("Structural plasticity (growth & pruning) has been ENABLED.")
        else:
            print("Structural plasticity (growth & pruning) has been DISABLED.")

        
    def reset_growth_parameters(self):
        """Reset growth regulation parameters for a fresh start"""
        self.current_growth_rate = 1.0
        self.current_growth_threshold = 0.75
        self.last_growth_step = 0  # Allow immediate growth
        print("Growth parameters reset")
        
    def set_learning_phase(self, phase):
        """
        Configure network dynamics for distinct neurobiological learning phases
        
        The network transitions between neurobiologically-inspired learning regimes:
        - Exploration: High plasticity, broader sampling of solution space (like REM sleep)
        - Exploitation: Targeted learning with high reward sensitivity (like focused learning)
        - Refinement: Fine-tuning with reduced plasticity (like skill consolidation)  
        - Convergence: Minimal exploration, high precision (like expert performance)
        
        Parameters:
        phase: One of 'exploration', 'exploitation', 'refinement', 'convergence'
        """
        try:
            valid_phases = ['exploration', 'exploitation', 'refinement', 'convergence']
            if phase not in valid_phases:
                print(f"Invalid phase '{phase}'. Must be one of {valid_phases}")
                return
                
            # Reset all phases
            for p in valid_phases:
                self.learning_phases[p] = False
                
            # Set requested phase
            self.learning_phases[phase] = True
            
            # Configure neuromodulatory systems for specific learning dynamics
            if phase == 'exploration':
                # Exploration resembles high acetylcholine and norepinephrine states
                # similar to REM sleep or novel environment exploration
                self.modulation['norepinephrine'] = 0.7  # High exploration/arousal
                self.modulation['dopamine'] = 0.4        # Moderate reward sensitivity 
                self.modulation['acetylcholine'] = 0.8   # High attention to input
                self.modulation['serotonin'] = 0.7       # Balanced mood/plasticity
                
                # Adjust plasticity parameters
                self.metaplasticity_rate = 0.0002        # Higher meta-plasticity
                self.synaptic_scaling_rate = 0.015       # Faster homeostatic regulation
                
                # NEW: Adjust neuronal autonomy parameters
                self.intrinsic_motivation += 0.2         # Increase curiosity
                self.intrinsic_motivation = cp.clip(self.intrinsic_motivation, 0.0, 1.0)
                
            elif phase == 'exploitation':
                # Exploitation resembles high dopamine states focused on reward
                # similar to motivated, reward-driven learning
                self.modulation['norepinephrine'] = 0.4  # Moderate exploration
                self.modulation['dopamine'] = 0.7        # High reward sensitivity
                self.modulation['acetylcholine'] = 0.6   # Moderate attention
                self.modulation['serotonin'] = 0.8       # High persistence
                
                # Adjust plasticity parameters
                self.metaplasticity_rate = 0.0001        # Standard meta-plasticity
                self.synaptic_scaling_rate = 0.01        # Standard homeostatic regulation
                
                # NEW: Adjust multi-factor plasticity
                self.plasticity_factors['reward'] = 1.5  # Increase reward-based learning
                
            elif phase == 'refinement':
                # Refinement resembles balanced neuromodulation for detailed tuning
                # similar to deep practice or skill refinement
                self.modulation['norepinephrine'] = 0.2  # Low exploration
                self.modulation['dopamine'] = 0.6        # Moderate reward sensitivity
                self.modulation['acetylcholine'] = 0.5   # Focused attention
                self.modulation['serotonin'] = 0.6       # Moderate persistence
                
                # Adjust plasticity parameters
                self.metaplasticity_rate = 0.00005       # Reduced meta-plasticity
                self.synaptic_scaling_rate = 0.005       # Slower homeostatic regulation
                
                # NEW: Adjust structural plasticity
                self.synapse_formation_threshold = 0.8   # More selective synapse formation
                
            elif phase == 'convergence':
                # Convergence resembles high acetylcholine with low norepinephrine
                # similar to expert performance or memory consolidation
                self.modulation['norepinephrine'] = 0.1  # Minimal exploration
                self.modulation['dopamine'] = 0.3        # Low reward sensitivity
                self.modulation['acetylcholine'] = 0.9   # High precision attention
                self.modulation['serotonin'] = 0.9       # High persistence/low flexibility
                
                # Adjust plasticity parameters
                self.metaplasticity_rate = 0.00001       # Minimal meta-plasticity
                self.synaptic_scaling_rate = 0.002       # Very slow homeostatic changes
                
                # NEW: Increase memory consolidation
                self.consolidation_rate = 0.002          # Faster consolidation
                
            # Adjust STDP parameters based on phase
            if phase in ['exploration', 'exploitation']:
                # Higher LTP/LTD ratio for acquisition phases
                self.stdp_params['A_plus'] = 0.012
                self.stdp_params['A_minus'] = 0.011
            else:
                # Lower and more balanced for refinement phases
                self.stdp_params['A_plus'] = 0.008
                self.stdp_params['A_minus'] = 0.0085
            
            # NEW: Adjust layer-specific modulation based on phase
            if phase == 'exploration':
                # Lower layers more active during exploration
                self.layer_modulation[0, :] = 1.2  # Boost lower layer
                self.layer_modulation[1, :] = 1.0  # Normal middle layer
                self.layer_modulation[2, :] = 0.8  # Reduce higher layer
            elif phase == 'refinement' or phase == 'convergence':
                # Higher layers more active during refinement/convergence
                self.layer_modulation[0, :] = 0.8  # Reduce lower layer
                self.layer_modulation[1, :] = 1.0  # Normal middle layer
                self.layer_modulation[2, :] = 1.2  # Boost higher layer
            else:
                # Balanced during exploitation
                self.layer_modulation[:, :] = 1.0
            
            print(f"Learning phase set to '{phase}' with adjusted neuromodulation profile")
            
            # Return current configuration for monitoring
            return {k: v for k, v in self.modulation.items()}
            
        except Exception as e:
            print(f"Error setting learning phase: {e}")
            # Do not change phase if an error occurs
            return None
    
    def get_network_stats(self):
        """
        Return comprehensive statistics about the neuromorphic network's state and composition
        
        This method provides detailed analytics for monitoring network evolution, 
        criticality dynamics, and biologically-relevant metrics similar to those
        tracked in cortical network analysis.
        
        Returns:
            dict: Comprehensive metrics about network composition and state
        """
        try:
            # Neuron type distribution (excitatory/inhibitory balance)
            excitatory_count = int(cp.sum(self.neuron_type > 0))
            inhibitory_count = int(cp.sum(self.neuron_type < 0))
            
            # Activation function diversity (computational heterogeneity)
            tanh_count = int(cp.sum(self.activation_types == 0))
            sigmoid_count = int(cp.sum(self.activation_types == 1))
            relu_count = int(cp.sum(self.activation_types == 2))
            adaptive_count = int(cp.sum(self.activation_types == 3))
            
            # Activity statistics (firing patterns)
            if len(self.recent_activations) > 0:
                # Use NumPy for safer conversion of Python collections
                recent_acts = np.array(list(self.recent_activations)[-10:])
                mean_activity = float(np.mean(recent_acts))
                activity_std = float(np.std(recent_acts))
            else:
                mean_activity = 0.0
                activity_std = 0.0
                
            # Energy consumption and neuronal health
            mean_energy = float(cp.mean(self.energy))
            mean_health = float(cp.mean(self.health))
            
            # Metabolic efficiency if available
            if hasattr(self, 'efficiency_ratio'):
                metabolic_efficiency = float(self.efficiency_ratio)
            else:
                metabolic_efficiency = 0.0
            
            # Network criticality metrics (edge-of-chaos dynamics)
            if hasattr(self, 'criticality_metrics'):
                criticality = self.criticality_metrics.copy()
            else:
                criticality = {
                    'spectral_radius': 0.0,
                    'eigenvalue_gap': 0.0,
                    'participation_ratio': 0.0,
                    'edge_of_chaos_distance': 1.0
                }
            
            # Information-theoretic metrics if available
            if hasattr(self, 'information_content') and self.information_content.shape[0] > 0:
                mean_info_content = float(cp.mean(self.information_content))
            else:
                mean_info_content = 0.0
                
            # Current learning phase
            active_phase = 'unknown'
            for phase, is_active in self.learning_phases.items():
                if is_active:
                    active_phase = phase
                    break
            
            # Performance metrics if available
            performance_metrics = {}
            if hasattr(self, 'recent_loss_history') and len(self.recent_loss_history) > 0:
                performance_metrics['recent_loss'] = float(self.recent_loss_history[-1])
                
                if hasattr(self, 'performance_gradient'):
                    performance_metrics['learning_gradient'] = float(self.performance_gradient)
                    
                if hasattr(self, 'performance_acceleration'):
                    performance_metrics['learning_acceleration'] = float(self.performance_acceleration)
            
            # Synaptic weight statistics
            if hasattr(self, 'Uh') and self.Uh.shape[0] > 0:
                weight_mean = float(cp.mean(self.Uh))
                weight_std = float(cp.std(self.Uh))
                weight_max = float(cp.max(cp.abs(self.Uh)))
            else:
                weight_mean = 0.0
                weight_std = 0.0
                weight_max = 0.0
                
            # Fast vs slow weight balance (memory consolidation)
            if hasattr(self, 'fast_weights') and hasattr(self, 'slow_weights'):
                fast_weight_norm = float(cp.sum(cp.abs(self.fast_weights)))
                slow_weight_norm = float(cp.sum(cp.abs(self.slow_weights)))
                if fast_weight_norm + slow_weight_norm > 0:
                    memory_consolidation_ratio = slow_weight_norm / (fast_weight_norm + slow_weight_norm)
                else:
                    memory_consolidation_ratio = 0.0
            else:
                memory_consolidation_ratio = 0.0
            
            # NEW: Connectivity statistics
            if hasattr(self, 'connectivity'):
                connectivity_density = float(cp.mean(self.connectivity))
                # Calculate small-world index (approximation)
                # In a small-world network, neurons connect to nearby neurons but also have some long-range connections
                try:
                    # Calculate clustering coefficient (approximation)
                    clustering = 0.0
                    for i in range(min(100, self.hidden_dim)):  # Sample for efficiency
                        neighbors = cp.where(self.connectivity[i])[0]
                        if len(neighbors) > 1:
                            # Count connections between neighbors
                            connections = 0
                            for j in neighbors:
                                for k in neighbors:
                                    if j != k and self.connectivity[j, k]:
                                        connections += 1
                            max_possible = len(neighbors) * (len(neighbors) - 1)
                            if max_possible > 0:
                                clustering += connections / max_possible
                    
                    clustering = clustering / min(100, self.hidden_dim)
                    
                    # Calculate average path length (approximation)
                    # Use connectivity as adjacency matrix
                    small_world_index = clustering / connectivity_density
                except:
                    small_world_index = 1.0
            else:
                connectivity_density = 0.0
                small_world_index = 1.0
            
            # NEW: Ensemble statistics
            if hasattr(self, 'ensemble_activity') and hasattr(self, 'ensemble_membership'):
                active_ensembles = int(cp.sum(self.ensemble_activity > 0.1))
                mean_ensemble_size = float(cp.mean(cp.sum(self.ensemble_membership > 0.5, axis=0)))
                ensemble_metrics = {
                    'active_ensembles': active_ensembles,
                    'mean_ensemble_size': mean_ensemble_size,
                    'ensemble_activity': [float(a) for a in cp.asnumpy(self.ensemble_activity)]
                }
            else:
                ensemble_metrics = {
                    'active_ensembles': 0,
                    'mean_ensemble_size': 0,
                    'ensemble_activity': []
                }
            
            # NEW: Hierarchical organization statistics
            if hasattr(self, 'neuron_layer'):
                layer_counts = [int(cp.sum(self.neuron_layer == i)) for i in range(self.num_layers)]
                layer_activities = []
                for i in range(self.num_layers):
                    layer_mask = self.neuron_layer == i
                    if cp.any(layer_mask):
                        layer_act = float(cp.mean(cp.abs(self.h[layer_mask])))
                    else:
                        layer_act = 0.0
                    layer_activities.append(layer_act)
                
                hierarchy_metrics = {
                    'layer_counts': layer_counts,
                    'layer_activities': layer_activities
                }
            else:
                hierarchy_metrics = {
                    'layer_counts': [],
                    'layer_activities': []
                }
            
            # NEW: Neuronal autonomy metrics
            if hasattr(self, 'intrinsic_motivation') and hasattr(self, 'local_objectives'):
                autonomy_metrics = {
                    'mean_intrinsic_motivation': float(cp.mean(self.intrinsic_motivation)),
                    'mean_local_objective': float(cp.mean(self.local_objectives)),
                    'mean_learning_rate': float(cp.mean(self.neuron_learning_rates))
                }
            else:
                autonomy_metrics = {
                    'mean_intrinsic_motivation': 0.0,
                    'mean_local_objective': 0.0,
                    'mean_learning_rate': 0.0
                }
            
            # NEW: Developmental stage
            developmental_metrics = {
                'stage': self.developmental_stage,
                'step_counter': self.step_counter,
                'critical_period_end': self.critical_period_end,
                'maturation_end': self.maturation_end
            }
            
            # Compile comprehensive statistics
            stats = {
                # Basic network architecture
                'hidden_dim': self.hidden_dim,
                'excitatory_count': excitatory_count,
                'inhibitory_count': inhibitory_count,
                'ei_ratio': excitatory_count / max(1, inhibitory_count),
                
                # Computational diversity metrics
                'activation_diversity': {
                    'tanh_count': tanh_count,
                    'sigmoid_count': sigmoid_count,
                    'relu_count': relu_count,
                    'adaptive_count': adaptive_count
                },
                
                # Neuroplasticity tracking
                'growth_metrics': {
                    'total_grown': self.total_grown,
                    'total_pruned': self.total_pruned,
                    'structural_plasticity_enabled': self.enable_growth and self.enable_pruning,
                    'growth_rate': float(self.current_growth_rate) if hasattr(self, 'current_growth_rate') else 0.0,
                },
                
                # Activity and energy dynamics
                'activity_metrics': {
                    'mean_activity': mean_activity,
                    'activity_std': activity_std,
                    'mean_energy': mean_energy,
                    'mean_health': mean_health,
                    'metabolic_efficiency': metabolic_efficiency,
                },
                
                # Network dynamics and informational complexity
                'network_dynamics': {
                    'spectral_radius': float(criticality['spectral_radius']),
                    'edge_of_chaos_distance': float(criticality['edge_of_chaos_distance']),
                    'mean_information_content': mean_info_content,
                    'memory_consolidation_ratio': memory_consolidation_ratio,
                    'connectivity_density': connectivity_density,
                    'small_world_index': small_world_index
                },
                
                # Synaptic weight statistics
                'weight_statistics': {
                    'mean': weight_mean,
                    'std': weight_std,
                    'max_abs': weight_max,
                },
                
                # Current learning state
                'learning_state': {
                    'phase': active_phase,
                    'step_counter': self.step_counter,
                    **performance_metrics
                },
                
                # Neuromodulation state
                'neuromodulation': dict(self.modulation),
                
                # NEW: Ensemble metrics
                'ensemble_metrics': ensemble_metrics,
                
                # NEW: Hierarchical organization
                'hierarchy_metrics': hierarchy_metrics,
                
                # NEW: Neuronal autonomy
                'autonomy_metrics': autonomy_metrics,
                
                # NEW: Developmental stage
                'developmental_metrics': developmental_metrics
            }
            
            return stats
            
        except Exception as e:
            print(f"Error generating network statistics: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal fail-safe statistics to prevent training disruption
            return {
                'hidden_dim': self.hidden_dim,
                'excitatory_count': int(0.8 * self.hidden_dim),
                'inhibitory_count': int(0.2 * self.hidden_dim),
                'learning_phase': 'unknown',
                'step_counter': self.step_counter if hasattr(self, 'step_counter') else 0
            }
    
    def reset_state(self):
        """Reset the network state (but not weights)"""
        self.h = cp.zeros((self.hidden_dim,))
        self.h_prev = cp.zeros((self.hidden_dim,))
        self.dh = cp.zeros((self.hidden_dim,))
        self.pre_synaptic_trace = cp.zeros((self.hidden_dim,))
        self.post_synaptic_trace = cp.zeros((self.hidden_dim,))
        self.dendritic_activations = cp.zeros((self.hidden_dim, self.num_dendrites))
        
        # Clear temporal memory
        self.temporal_memory.clear()
        
        # Reset phase for oscillations
        self.phase = cp.random.uniform(0, 2*np.pi, size=self.hidden_dim)

