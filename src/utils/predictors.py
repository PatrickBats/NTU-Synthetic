from collections import defaultdict
import torch
from tqdm import tqdm
from src.utils.hook import register_hooks, remove_hooks
import torch.nn.functional as F
import pandas as pd

class TrialPredictor:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.model = self.feature_extractor.model
        self.device = self.feature_extractor.device
        self.model_name = self.feature_extractor.model_name

    def predict(self, dataloader):
        predictions = []

        if self.model_name in ['resnext', 'dino_s_resnext50']:
            print("ResNext and DINO models have no text encoder, class text embedding not available")
            return predictions  

        with torch.no_grad():
            for batch_idx, (imgs, label, foil_labels) in enumerate(tqdm(dataloader, desc="Trial Prediction")):
                # resize imgs from [batch_size, trial_size, c, h, w] to [batch_size * trial_size, c, h, w]
                batch_size, per_trial_img_num, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width)  # Flatten the trials into the batch dimension

                img_features = self.feature_extractor.get_img_feature(imgs)  # [batch_size*4, 512]
                img_features = img_features.view(batch_size, per_trial_img_num, -1)  
                img_features = self.feature_extractor.norm_features(img_features) # [batch_size, 4, 512]

                txt_features = self.feature_extractor.get_txt_feature(label)  # [batch_size, 512]
                txt_features = self.feature_extractor.norm_features(txt_features) 
                txt_features = txt_features.unsqueeze(1)# [batch_size, 1,  512]

                # Calculate the cosine similarity
                similarity = (100.0 * img_features @ txt_features.transpose(-2, -1)).softmax(dim=-2)  # [batch_size, 4, 1]
                similarity = similarity.squeeze(-1) # Remove the last dimension
                
                for i in range(batch_size):# loop each trial in the batch
                    simil = similarity[i]  # Get the similarity scores for the i-th item in the batch
                    predic_idx = simil.argmax().item()  # Find the index of the max similarity score for each trial
                    trial_idx = batch_idx * batch_size + i  # get global data index, same to data[trial_idx]

                    predictions.append({
                        'trial_idx': trial_idx,
                        'predic_idx': predic_idx, # inside trial
                        'gt_label': label[i], 
                        'categories': [label[i]] + foil_labels[i],
                        'similarity': simil.tolist(),
                        'is_correct': (predic_idx == 0),
                    })

        return predictions
    
    def predict_using_neurons(self, dataloader, layers, neuron_concepts_path, top_k=1):
        predictions = []
    
        with torch.no_grad():
            for batch_idx, (imgs, label, foil_labels) in enumerate(tqdm(dataloader, desc="Trial Prediction with Neuron Concepts")):
                                
                # resize imgs from [batch_size, trial_size, c, h, w] to [batch_size * trial_size, c, h, w]
                batch_size, trial_size, channels, height, width = imgs.size()
                imgs = imgs.view(-1, channels, height, width) # [batch_size * per_trial_img_num, c, h, w]
                imgs = imgs.to(self.device)

                # register hooks
                activations, hooks = register_hooks(self.model, layers, mode='avg', trial_size=trial_size) 

                # forward pass
                _ = self.feature_extractor.get_img_feature(imgs)  # [batch_size*4, 512]
               
                neurons = self.search_label_corresponding_neuron(label, neuron_concepts_path, top_k)
                top_activated_indices, neuron_activations = self.find_top_activated_img_idx(activations, neurons)# predict upon highest activated neuron

                
                for i in range(batch_size):
                    predic_idx = top_activated_indices[i]  # Find the index of the max similarity score for each trial
                    predictions.append({
                        'batch_idx': batch_idx,
                        'data_idx': i,
                        'selected_neurons': neurons[i],
                        'neuron_activations': neuron_activations[i],
                        'prediction': predic_idx, # inside trial
                        'categories': [label[i]] + foil_labels[i],
                        'gt_label': label[i], 
                        'is_correct': (predic_idx == 0),
                    })


                activations.clear() # clear activations to save memory
                remove_hooks(hooks)

        return predictions

    def search_label_corresponding_neuron(self, label, neuron_concepts_path, top_k=1):
        """
        find top similariy neurons with same label
        """
        neurons = {}
        df_neuron_concepts = pd.read_csv(neuron_concepts_path)
        for i, lbl in enumerate(label):
            matched_neurons = df_neuron_concepts[df_neuron_concepts['description'].str.contains(lbl, case=False, na=False)]
            # print(f"matched_neurons: {matched_neurons}")
            actual_top_k = min(top_k, len(matched_neurons)) # to avoid top_k >> matched_neurons
            top_neurons = matched_neurons.nlargest(actual_top_k, 'similarity') # here to select top_k neurons
            neurons[i] = [(neuron['layer'], neuron['unit']) for _, neuron in top_neurons.iterrows()]

        """
        top_k=2
        neurons = {
            0: [(layer1, unit1), (layer4, unit7), ...],  # 1st label
            1: [(layer2, unit3), (layer3, unit218), ...],  # 2nd label
            # ...
        }"""
        return neurons
    
    def find_top_activated_img_idx(self, activations, neurons):
        """
        Find top activated image among trials for each corresponding set of neurons
        Returns:
            - top_activated_indices: list of indices with highest activation count
            - neuron_activations: dict containing activation values for each neuron
        Raises:
            ValueError: If no neurons are found for any item
        """
        top_activated_indices = []
        all_img_activations = {}  # save each batch item activations
        
        for i, neuron_list in neurons.items():
            activation_counts = defaultdict(int)
            batch_activations = {}
            
            if not neuron_list:
                raise ValueError(f"No matching neurons found for item {i}")
            
            for layer, unit in neuron_list:
                neuron_acts = activations[layer][0][i,:,unit].squeeze()  # sized [trial_size]
                batch_activations[f"{layer}_unit{unit}"] = neuron_acts.tolist()
                
                top_activated_idx = neuron_acts.argmax().item()
                activation_counts[top_activated_idx] += 1
            
            # Find the index with the highest activation count
            top_activated_idx = max(activation_counts.items(), key=lambda x: x[1])[0]
            top_activated_indices.append(top_activated_idx)
            all_img_activations[i] = batch_activations
        
        return top_activated_indices, all_img_activations


