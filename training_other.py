import pandas as pd
import numpy as np
from fastai.vision.all import *
import torchvision
from sklearn.model_selection import StratifiedGroupKFold
import sys
import pickle
import torch 

df = pd.read_csv("../ventilzielaufnahmen_mit_pid.csv")
df["fn"] = ["../jpegs_same_contrast_cropped/" + x.split("/")[-1].split("-")[-1] for x in df.image]
df = df[~(df["choice"] == "Schrott")]
df["choice"] = [choice if choice in ["Codman Hakim", "Codman Certas Plus", "Sophysa Sophy SM8", "proGAV 2.0"] else "other" for choice in df.choice]

def create_dls(df, split, bs=32):
    dblock = DataBlock(
                        blocks=(ImageBlock, CategoryBlock(
                            vocab=["Codman Hakim", "Codman Certas Plus", "other", "Sophysa Sophy SM8", "proGAV 2.0"],
                            sort=False
                        )), 
                        get_x=ColReader("fn"),
                        get_y=ColReader("choice"),
                        splitter=IndexSplitter(split[1]),
                        item_tfms=[Resize(512, method=ResizeMethod.Squish)],
                        batch_tfms=aug_transforms()
    )
    dls = dblock.dataloaders(df, bs=bs)
    return dls

def create_model():
    resnet34 = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
    body = create_body(resnet34, cut=-2)
    head = create_head(512,5)
    pretrained_resnet34 = nn.Sequential(body, head)
    return pretrained_resnet34

class Tee:
    def __init__(self, stdout, logfile):
        self.stdout = stdout
        self.logfile = logfile
        self.open = True
        
    def write(self, obj):
        if self.open:
            self.stdout.write(obj)
            self.logfile.write(obj)
            
    def flush(self):
        if self.open:
            self.stdout.flush()
            self.logfile.flush()
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.open = False

@contextmanager
def redirect_output(filepath):
    """Context manager for handling output redirection safely"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        with open(filepath, 'w') as logfile:
            with Tee(original_stdout, logfile) as tee:
                sys.stdout = sys.stderr = tee
                yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

skf = StratifiedGroupKFold(shuffle=True)
splits = list(skf.split(df.fn, df.choice, df.pid))

splits_name = "splits.pkl"

#save splits to disk
with open(splits_name, 'wb') as f:
    pickle.dump(splits, f)

def main():

    torch.cuda.set_device(1)

    print("===========================================\n")
    print(f"splits save to {splits_name}")
    print("===========================================\n")
            
    for i, split in enumerate(splits):
        print("===========================================\n")
        print(f"now starting split {i+1}")
        print("===========================================\n")
        
        loss_weights = torch.tensor(1 / np.sqrt(df.choice.value_counts().values)).float()
        print("loss weights: ", loss_weights)
        dls = create_dls(df, split)
        model = create_model()

        learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(weight=loss_weights), metrics=[error_rate, F1Score(average="macro")], cbs=[GradientAccumulation(n_acc=64)])


        with redirect_output(f'on_patient_split_squish_{i}.txt'):
            learn.fine_tune(50, 1e-3)

        learn.save(f"resnet34_pretrained_4_ventile_on_patient_split_squish_other_{i}")

        # Explicit cleanup
        del learn
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise