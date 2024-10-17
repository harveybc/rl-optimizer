import importlib.metadata
from importlib import import_module
import sys

# Print sys.path to confirm visibility
print("Current sys.path:", sys.path)

# List registered plugins in 'rl_optimizer.optimizers'
print("Registered plugins:")
for entry_point in importlib.metadata.entry_points()['rl_optimizer.optimizers']:
    print(f"- {entry_point.name}: {entry_point.value}")

# Manually attempt to load the 'neat_a_nomc' plugin
try:
    entry_point = next(ep for ep in importlib.metadata.entry_points()['rl_optimizer.optimizers'] if ep.name == 'neat_a_nomc')
    print(f"Attempting to load plugin {entry_point.name}")
    plugin_class = entry_point.load()
    print(f"Loaded plugin: {plugin_class}")
except Exception as e:
    print(f"Failed to load plugin: {e}")
