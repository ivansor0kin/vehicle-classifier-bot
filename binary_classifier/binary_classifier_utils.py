import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SimpleBinaryDataset(Dataset):
    def __init__(self, pos_root, neg_root, transform):
        self.samples = []
        for p in glob(os.path.join(pos_root, '*')):
            self.samples.append((p, 1))
        for p in glob(os.path.join(neg_root, '*')):
            self.samples.append((p, 0))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0,0,0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out,y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
            pred = out.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()
        model.eval()
        v_total, v_correct = 0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_total += y.size(0); v_correct += (pred==y).sum().item()
        print(f"{ep+1}/{epochs}  TrainLoss={loss_sum/total:.4f}  TrainAcc={100*correct/total:.2f}%  ValAcc={100*v_correct/v_total:.2f}%")
    return model
