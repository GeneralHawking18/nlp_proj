from lavis.models import load_model_and_preprocess
import torch
import torch.nn as nn

class BLIP(nn.Module):
    def __init__(self, mode = "eval"):
        super().__init__()
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode 
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", 
            model_type="base_coco", 
            is_eval = (mode == "eval"),
            device=self.device
        )
        self.change_mode(mode)

    def change_mode(self, mode = "eval"):
        self.transforms = self.vis_processors[mode]
    
    def get_transforms(self):
        return self.transforms

    def forward(self, image):
        caption = self.model.generate({"image": image})
        return caption
    
