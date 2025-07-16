import torch
import clip

class FeatureExtractor:
    """
    For CVCL and CLIP -like models to extract features from text and images.
    """
    def __init__(self, model_name, model, device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def get_txt_feature(self, label):

        if "cvcl" in self.model_name and "clip" not in self.model_name:
            tokens, token_len = self.model.tokenize(label)  # Separate the tokenization from the device transfer
            tokens = tokens.to(self.device)
            if isinstance(token_len, torch.Tensor):
                token_len = token_len.to(self.device)
            txt_features = self.model.encode_text(tokens, token_len)
            
        elif "clip" in self.model_name:
            # label = label.squeeze(0)  originated from CVCL repo
            tokens = clip.tokenize(label).to(self.device).long()
            txt_features = self.model.encode_text(tokens)
            
        else:
            raise ValueError(f"Unknown text encoder: {self.model_name}")
        
        return txt_features

    
    def get_img_feature(self, imgs):
        self.model.to(self.device)
        imgs = imgs.to(self.device)
        if hasattr(self.model, 'encode_image'):
            img_features = self.model.encode_image(imgs)
        else:
            img_features = self.model(imgs)  # Use standard forward method
        return img_features
    
    def norm_features(self, features):
        # norm txt img feature on feature dim
        return features / features.norm(dim=-1, keepdim=True)  
