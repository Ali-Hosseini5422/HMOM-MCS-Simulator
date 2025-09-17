"""
HMOM-MCS Project Structure
=========================

Project Directory:
hmom_mcs_project/
│
├── config/
│   └── parameters.py           # System parameters configuration
│
├── models/
│   ├── __init__.py
│   ├── user.py                # User class
│   ├── task.py                # Task class
│   └── blockchain.py          # Mock PoA Blockchain
│
├── algorithms/
│   ├── __init__.py
│   ├── federated_learning.py  # FL module
│   ├── coalition.py           # Coalition formation
│   └── optimizer.py           # BES-GO optimizer
│
├── visualization/
│   ├── __init__.py
│   ├── metrics_plot.py        # Main metrics visualization
│   ├── network_plot.py        # Social network visualization
│   ├── distribution_plot.py   # Task/User distribution
│   ├── coalition_plot.py      # Coalition formation visualization
│   └── blockchain_plot.py     # Blockchain visualization
│
├── core/
│   ├── __init__.py
│   └── hmom_mcs.py           # Main system class
│
├── utils/
│   ├── __init__.py
│   └── network_generator.py   # Social network generation
│
└── main.py                     # Main execution script

"""