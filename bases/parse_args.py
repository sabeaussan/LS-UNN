import yaml


def parse_yaml(file_name):
    args = []
    with open("config_files/"+file_name, 'r') as stream:
        documents = yaml.safe_load(stream)
        documents = documents["common"]
        for items in documents :
            args.append(list(items.values())[0])
    return args,documents


def save_yaml(dir_path,run_id,config):
    with open(dir_path+f"{run_id}.yaml", 'w') as file:
        documents = yaml.dump(config, file)
