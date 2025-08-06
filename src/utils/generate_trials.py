import os
import random
import json
import pandas as pd
from src.utils.dataset_loader import DATASET_ROOTS
from src.utils.utils import set_seed
from src.utils.neuron_map import read_dissect_csv

#TODO: not hard code
full_class = ['dumbbell', 'cupsaucer', 'hanger', 'watergun', 'crib', 'roadsign', 'knife', 'necklace', 'clock', 'bearteddy', 'cake', 'nunchaku', 'keyboard', 'wineglass', 'trunk', 'patioloungechair', 'christmasstocking', 'cigarette', 'bottle', 'microscope', 'abacus', 'powerstrip', 'orifan', 'chessboard', 'candleholderwithcandle', 'pizza', 'fish hook', 'bonzai', 'goggle', 'exercise_equipment', 'bathsuit', 'rug', 'beermug', 'mp3player', 'toothpaste', 'pitcher', 'backpack', 'axe', 'ring', 'cookingpan', 'fan', 'saltpeppershake', 'compass', 'cookie', 'scissors', 'socks', 'ringbinder', 'hat', 'spoon', 'stamp', 'earings', 'flashlight', 'watch', 'broom', 'binoculars', 'bell', 'tape', 'cherubstatue', 'lei', 'phone', 'umbrella', 'train', 'suitcase', 'nailpolish', 'suit', 'rock', 'saddle', 'airplane', 'cheesegrater', 'headband', 'ball', 'magazinecovers', 'keychain', 'seashell', 'juice', 'bird', 'ceilingfan', 'bucket', 'bowl', 'button', 'sofa', 'coin', 'barbiedoll', 'dollhouse', 'mask', 'christmastreeornamantball', 'glove', 'sodacan', 'bowtie', 'pants', 'pen', 'gift', 'easteregg_redo', 'tent', 'decorativescreen', 'garbagetrash', 'baseballcards', 'doll', 'coatrack', 'snowglobe', 'vase', 'breadloaf', 'tv', 'yarn', 'frisbee', 'toiletseat', 'muffins', 'armyguy', 'key', 'frame', 'dresser', 'cheese', 'rosary', 'chair', 'guitar', 'handbag', 'motorcycle', 'babushkadolls', 'stapler', 'lock', 'lantern', 'kayak', 'trumpet', 'gamehandheld', 'shoe', 'recordplayer', 'cellphone', 'bagel', 'leaves', 'bike', 'trophy', 'stool', 'wig', 'windchime', 'camera', 'telescope', 'tennisracquet', 'pokercard', 'toyhorse', 'videoGameController', 'doorknob', 'dog', 'camcorder', 'meat', 'mushroom', 'computer_key', 'cushion', 'donut', 'rollerskates', 'tablesmall', 'boot', 'hourglass', 'turtle', 'speakers', 'bill', 'desk', 'lawnmower', 'makeupcompact', 'calculator', 'toyrabbit', 'sippycup', 'scale', 'radio', 'tongs', 'jack-o-lantern', 'bongo', 'hairbrush', 'licenseplate', 'pipe', 'bed', 'grill', 'beanbagchair', 'golfball', 'domino', 'hammer', 'balloon', 'handgun', 'cookpot', 'flag', 'basket', 'bench', 'razor', 'sandwich', 'lamp', 'collar', 'coffeemug', 'butterfly', 'tree', 'babycarriage', 'microwave', 'jacket', 'carfront', 'headphone', 'cat', 'lipstick', 'necktie', 'helmet', 'apple', 'scrunchie', 'tricycle']
unseen_class = ['golfball', 'abacus', 'mp3player', 'telescope', 'gamehandheld', 'windchime', 'easteregg_redo', 'scale', 'roadsign', 'frisbee', 'orifan', 'scrunchie', 'seashell', 'nailpolish', 'rollerskates', 'cookingpan', 'christmasstocking', 'pitcher', 'toyhorse', 'babushkadolls', 'tennisracquet', 'flag', 'watergun', 'keyboard', 'garbagetrash', 'mushroom', 'gift', 'lei', 'decorativescreen', 'microscope', 'cigarette', 'stapler', 'donut', 'compass', 'videoGameController', 'tongs', 'bearteddy', 'calculator', 'collar', 'licenseplate', 'dollhouse', 'hourglass', 'radio', 'mask', 'hammer', 'baseballcards', 'bathsuit', 'cushion', 'carfront', 'ceilingfan', 'coatrack', 'flashlight', 'tablesmall', 'axe', 'cherubstatue', 'cellphone', 'nunchaku', 'suit', 'bowtie', 'cookpot', 'binoculars', 'saddle', 'trunk', 'patioloungechair', 'christmastreeornamantball', 'lock', 'toyrabbit', 'rosary', 'bonzai', 'camcorder', 'beermug', 'headphone', 'bill', 'tent', 'lantern', 'helmet', 'ringbinder', 'keychain', 'candleholderwithcandle', 'exercise_equipment', 'glove', 'fish hook', 'headband', 'rug', 'breadloaf', 'handbag', 'dumbbell', 'trophy', 'wineglass', 'powerstrip', 'sippycup', 'chessboard', 'yarn', 'suitcase', 'speakers', 'wig', 'vase', 'pokercard', 'goggle', 'frame', 'barbiedoll', 'lamp', 'babycarriage', 'lawnmower', 'recordplayer', 'coffeemug', 'toiletseat', 'lipstick', 'doorknob', 'handgun', 'meat', 'boot', 'doll', 'hanger', 'earings', 'dresser', 'cupsaucer', 'magazinecovers', 'computer_key', 'saltpeppershake', 'necktie', 'snowglobe', 'tape', 'motorcycle', 'sodacan', 'jack-o-lantern', 'beanbagchair', 'cheesegrater', 'razor', 'armyguy', 'domino', 'muffins', 'makeupcompact', 'grill', 'bongo', 'trumpet']
seen_class = ['necklace', 'key', 'jacket', 'bottle', 'sandwich', 'microwave', 'toothpaste', 'balloon', 'basket', 'stamp', 'pants', 'fan', 'kayak', 'butterfly', 'bed', 'spoon', 'clock', 'button', 'tricycle', 'leaves', 'knife', 'backpack', 'scissors', 'bird', 'guitar', 'chair', 'apple', 'bagel', 'rock', 'cake', 'hairbrush', 'ball', 'bucket', 'tv', 'camera', 'bowl', 'phone', 'cheese', 'bench', 'coin', 'shoe', 'ring', 'juice', 'pipe', 'desk', 'cookie', 'crib', 'pizza', 'socks', 'train', 'airplane', 'watch', 'sofa', 'pen', 'stool', 'dog', 'bike', 'bell', 'hat', 'turtle', 'tree', 'umbrella', 'broom', 'cat']

trial_save_dir = 'data/trials'

def _get_map_prefix(map_file):
    """Extract meaningful prefix from map file path"""
    if map_file is None:
        print("Map_file is None, predicting without neuron concepts")
        return ''
    elif map_file.endswith('descriptions.csv'):
        # Read args.txt from the same directory
        args_path = os.path.join(os.path.dirname(map_file), 'args.txt')
        if not os.path.exists(args_path):
            raise ValueError(f"args.txt not found in the same directory as {map_file}")
            
        # Load args json
        with open(args_path) as f:
            args = json.load(f)
            
        # Extract required fields
        model_name = os.path.dirname(map_file).split('/')[-1].split('_')[0] # neuron_concepts/cvcl-clip-layer3_24_11_13_11_31/descriptions.csv -> cvcl-clip
        concept_name = args.get('concept_set', '').split('/')[-1].replace('.txt', '')
        d_probe_name = args.get('d_probe', '')
        
        if not model_name or not concept_name or not d_probe_name:
            raise ValueError(f"Missing required fields: model_name='{model_name}', concept_name='{concept_name}', d_probe_name='{d_probe_name}'")
        
        # Return formatted prefix
        # print (f"Extracted prefix: {model_name}_{d_probe_name}_{concept_name}_")
        return f'{model_name}_{d_probe_name}_{concept_name}_'
        
    else:
        return  map_file.split('/')[-1].split('.')[0] + '_'  # neuron_concepts/cvcl_objects_baby+konk+30k.csv -> cvcl_objects_baby+konk+30k
        
class TrialGenerator:
    def __init__(self, seed, num_imgs, num_trials_per_img, class_type, map_file=None, resize=False):
        self.resize = resize
        self.root_dir = DATASET_ROOTS['objects_resized'] if resize else DATASET_ROOTS['objects']
        self.seed = seed
        self.num_imgs = num_imgs
        self.num_trials_per_img = num_trials_per_img
        #! if map_file is None, will not filter appeared neuron class
        self.map_file = map_file # neuron description csv file
        
        class_type_map = {
            'unseen': unseen_class,
            'seen': seen_class,
            'full': full_class
        }
        self.class_names = class_type_map.get(class_type, []) # find seen or unseen class names
        
        self.set_seed()
        
        # Use new helper function to get map prefix
        map_prefix = _get_map_prefix(map_file)
        resize_prefix = "resize_" if resize else ""
        
        # save trials in data folder
        os.makedirs(trial_save_dir, exist_ok=True)
        self.trials_file_path = os.path.join(trial_save_dir, 
            f'{resize_prefix}{class_type}_{map_prefix}object_{self.num_trials_per_img}_{self.num_imgs}_{self.seed}.json')
    
    def set_seed(self):
        set_seed(self.seed)
        
    def get_trials_path(self): # main function
        """Check if trials file exists and generate if not, then return trials path and data."""
        root_dir_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(root_dir_path):
            os.makedirs(root_dir_path)

        if os.path.exists(self.trials_file_path):
            print(f"Trial file already exists: {self.trials_file_path}, skipping generation.")
            trials = json.load(open(self.trials_file_path))
            return self.trials_file_path

        print("Generating trials...")
        self.set_seed() 
        
        # filter appeared neuron class if map_file is provided
        neuron_class = self.filter_appeared_neuron_class() if self.map_file is not None else self.class_names
        images = self.get_all_class_images(neuron_class)
        trials = self.generate_trials(images)
        
        self.save_json(trials, self.trials_file_path)
        return self.trials_file_path
    
    def filter_appeared_neuron_class(self):
        df_neuron = read_dissect_csv(self.map_file)
        neuron_descr = df_neuron['description'].unique().tolist()
        neuron_class = [cls for cls in self.class_names if cls in neuron_descr] # find classes that appear in neuron_descr
        return neuron_class
    

    def get_all_class_images(self, class_names):
        """Collect images for each class and return a dictionary mapping classes to image paths."""
        all_images = {}
        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                all_images[class_name] = self.get_images_from_directory(class_dir)
        return all_images

    def get_images_from_directory(self, directory, extension='.jpg'):
        """Collect all images from the specified directory with the given extension, sorted by filename."""
        image_paths = []
        for root, dirs, files in os.walk(directory):
            if 'TestItems' in dirs: # remove TestItems directory TODO justify this 
                dirs.remove('TestItems')
            for file in files:
                if file.lower().endswith(extension):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def generate_trials(self, all_images):
        """Generate trials for each image by selecting foils from other classes."""
        self.set_seed() 
        trials = []
        all_classes = sorted(all_images.keys()) 

        for class_name in all_classes:
            images = all_images[class_name]
            for image in images:
                for trial_num in range(self.num_trials_per_img):
                    foil_classes = random.sample([cls for cls in all_classes if cls != class_name and len(all_images[cls]) > 0], self.num_imgs-1) # -1 for target class
                    foil_images = [random.choice(all_images[cls]) for cls in foil_classes]
                    foil_categories = [os.path.split(os.path.dirname(foil_image))[1] for foil_image in foil_images]
                    trials.append({
                        'trial_num': trial_num,
                        'target_category': class_name,
                        'target_img_filename': image,
                        'foil_categories': foil_categories,
                        'foil_img_filenames': foil_images
                    })
        return trials

    def save_json(self, data, file_path):
        """Save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
        

def get_reported_trials(object_resize):
    """
    Using reported trials in original CVCL paper (just to reproduce science reported results)
    """
    #TODO: not hard code
    if object_resize:
        trial_path = 'data/trials/published_trial.json'
    else:
        trial_path = 'data/trials/nonresize_published_trial.json'
    print(f"Using reported trial in CVCL paper: {trial_path}")
    return trial_path


def get_trials(trial_type, seed, num_img_per_trial=4, num_trials_per_image=5, class_type="full", map_file=None, object_resize=False):
    # print(f"Generating trials with parameters:")
    # print(f"- trial_type: {trial_type}")
    # print(f"- seed: {seed}")
    # print(f"- num_img_per_trial: {num_img_per_trial}")
    # print(f"- class_type: {class_type}")
    # print(f"- object_resize: {object_resize}")
    if trial_type == 'custom': 
        generator = TrialGenerator(seed, num_img_per_trial, num_trials_per_image, 
                                   class_type, map_file, object_resize)
        trial_path = generator.get_trials_path()
    else: # reported published trials in CVCL
        trial_path = get_reported_trials(object_resize)   
    
    return trial_path

