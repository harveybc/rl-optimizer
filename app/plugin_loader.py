from importlib.metadata import entry_points

def load_plugin(plugin_group, plugin_name):
    print(f"Attempting to load plugin: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {plugin_class.plugin_params}")
        return plugin_class, required_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name} from group {plugin_group}, Error: {e}")
        raise

def load_environment_plugin(env_name):
    return load_plugin('rl_optimizer.environments', env_name)

def load_agent_plugin(agent_name):
    return load_plugin('rl_optimizer.agents', agent_name)

def load_optimizer_plugin(optimizer_name):
    return load_plugin('rl_optimizer.optimizers', optimizer_name)

def get_plugin_params(plugin_group, plugin_name):
    print(f"Getting plugin parameters for: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        print(f"Retrieved plugin params: {plugin_class.plugin_params}")
        return plugin_class.plugin_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
        raise ImportError(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
