import os

def parse_metafile(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()[1:]
    celeb_ids = {}
    for line in lines:
        ID, name, _, _, _ = line.rstrip().split('\t')
        celeb_ids[ID] = name
    return celeb_ids

def get_labels(voice_list, face_list):
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names

    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = sorted(list(names))
    label_dict = dict(zip(names, range(len(names))))
    for item in voice_list+face_list:
        item['label_id'] = label_dict[item['name']]
    return voice_list, face_list, len(names)
    

def get_dataset_files(data_dir, data_ext, celeb_ids, split):
    data_list = []
    # read data directory
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                # so hacky, be careful! 
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder)
                if celeb_name.startswith(tuple(split)):
                    data_list.append({'filepath': filepath, 'name': celeb_name})
    return data_list

def get_dataset(data_params):
    celeb_ids = parse_metafile(data_params['meta_file'])
    
    voice_list = get_dataset_files(data_params['voice_dir'],
                                   data_params['voice_ext'],
                                   celeb_ids,
                                   data_params['split'])
    face_list = get_dataset_files(data_params['face_dir'],
                                  data_params['face_ext'],
                                  celeb_ids,
                                  data_params['split'])
    return get_labels(voice_list, face_list)

