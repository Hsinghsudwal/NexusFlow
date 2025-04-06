def load_config(mode):
    import yaml
    with open(f"config/{mode}.yml") as f:
        return yaml.safe_load(f)
