# SOURCES
# https://challenge.isic-archive.com/data/#2020
# https://www.kaggle.com/c/siim-isic-melanoma-classification/data

import os.path
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torcheval.metrics import BinaryAccuracy, BinaryAUROC
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GroupShuffleSplit

dtypes = {
    'sex': pd.CategoricalDtype(),
    'anatom_site_general_challenge': pd.CategoricalDtype(),
    'diagnosis': pd.CategoricalDtype(),
    'benign_malignant': pd.CategoricalDtype()
}

class ISIC_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = resnet50(weights='IMAGENET1K_V2')
        self.clf = nn.Linear(1000, 1)

    def forward(self, x):
        z = self.encode(x)
        return self.clf(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict_proba(self, x):
        return torch.sigmoid(self(x))


def setup_train_test():
    df = pd.read_csv('ISIC_2020_Training_GroundTruth_v2.csv', dtype=dtypes)

    df_dup = pd.read_csv('ISIC_2020_Training_Duplicates.csv')

    dup_msk = df.image_name.isin(df_dup.image_name_1)
    print(f'Dropping {dup_msk.sum()} images that have duplicates.')
    df = df[~dup_msk]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=15)

    train_idces, test_idces = next(gss.split(df.image_name, df.target, groups=df.lesion_id))

    train_df = df.iloc[train_idces, :]
    test_df = df.iloc[test_idces, :]

    assert not train_df.lesion_id.isin(test_df.lesion_id).any()

    print('========== TRAIN_DF ===========')
    print(train_df.info())
    print(train_df.describe())

    print('========== TEST_DF ===========')
    print(test_df.info())
    print(test_df.describe())

    train_df.to_csv('isic_train.csv')
    test_df.to_csv('isic_test.csv')


class ISIC_dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root_dir, csv_file, augment=True) -> None:

        transforms_lst = [
            #transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize(  # params for pretrained resnet, see https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),            
        ]

        if augment:
            transforms_lst.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])

        transform = transforms.Compose(transforms_lst)

        super().__init__(root_dir, transform)

        df = pd.read_csv(csv_file, dtype=dtypes)

        self.root_dir = root_dir
        self.fnames = df.image_name
        self.labels = df.target
        self.sex = df.sex.cat.codes
        self.sex_codes = df.sex.cat.categories
        self.age = df.age_approx
        self.site = df.anatom_site_general_challenge.cat.codes
        self.site_codes = df.anatom_site_general_challenge.cat.categories
        self.diagnosis = df.diagnosis.cat.codes
        self.diagnosis_codes = df.diagnosis.cat.categories
        self.idx = df.index
        self.metadata_names = ['image_name', 'sex', 'age', 'site', 'diagnosis']

        self.transform = transform

        self.classes = df.target.unique()

    def __getitem__(self, index: int):
        img = torchvision.io.read_image(os.path.join(self.root_dir, self.fnames[index] + '.jpg'))
        img = self.transforms(img)
        return img, self.labels[index], (self.fnames[index], self.sex[index], self.age[index], self.site[index], self.diagnosis[index])
    
    def __len__(self) -> int:
        return len(self.fnames)


def train():
    writer = SummaryWriter()

    train_data = ISIC_dataset('ISIC_2020_Training_JPEG_lowres/train/', 'isic_train.csv', augment=True)
    test_data = ISIC_dataset('ISIC_2020_Training_JPEG_lowres/train/', 'isic_test.csv', augment=False)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)

    model = ISIC_Model()

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

    assert torch.cuda.is_available()

    device = torch.device("cuda:0")
    model = model.to(device)

    # Define the number of epochs
    num_epochs = 25

    thresh = train_data.labels.mean()
    train_acc = BinaryAccuracy(threshold=thresh)
    test_acc = BinaryAccuracy(threshold=thresh)
    train_auroc = BinaryAUROC()
    test_auroc = BinaryAUROC()

    for epoch in range(num_epochs):

        print(f'======= EPOCH {epoch} =======')

        model.train()
        train_loss = 0.0
        train_acc.reset()
        train_auroc.reset()
        for i, (inputs, labels, metadata) in tqdm(enumerate(train_loader)):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc.update(torch.sigmoid(outputs), labels)
            train_auroc.update(outputs, labels)

        model.eval()
        test_loss = 0.0
        test_acc.reset()
        test_auroc.reset()
        test_results = []
        with torch.no_grad():
            for i, (inputs, labels, metadata) in tqdm(enumerate(test_loader)):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(inputs).reshape(-1)
                loss = criterion(outputs, labels.to(torch.float32))

                test_loss += loss.item() * inputs.size(0)
                test_acc.update(torch.sigmoid(outputs), labels)
                test_auroc.update(outputs, labels)

                test_results.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu())}
                                                 | {k: v.cpu() if isinstance(v, torch.Tensor) else v for (k, v) in zip(test_data.metadata_names, metadata)}))

        test_results_df = pd.concat(test_results, ignore_index=True)
        test_results_df["site"] = test_results_df.site.map({0: "head/neck", 1: "lower extremity", 2: "oral/genital", 3: "palms/soles", 4: "torso", 5: "upper extremity"})
        test_results_df.to_csv('isic_test_results.csv')

        train_loss /= len(train_data)
        test_loss /= len(test_data)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc.compute():.4f} Train AUROC: {train_auroc.compute()}\n"
              f"             Test  Loss: {test_loss:.4f} Test  Acc: {test_acc.compute():.4f} Test AUROC: {test_auroc.compute()}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("acc/train", train_acc.compute(), epoch)
        writer.add_scalar("acc/test", test_acc.compute(), epoch)
        writer.add_scalar("auroc/train", train_auroc.compute(), epoch)
        writer.add_scalar("auroc/test", test_auroc.compute(), epoch)

    writer.flush()
    writer.close()

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'isic_resnet50.chkpt')


def resize_image(args):
    img_file, input_path, output_path = args
    
    # Keep the same relative path structure
    rel_path = img_file.relative_to(input_path)
    output_file = output_path / rel_path
    
    # Create subdirectories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with Image.open(img_file) as img:
        resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        resized.save(output_file, 'JPEG', quality=95)


def make_low_res():
    os.makedirs('ISIC_2020_Training_JPEG_lowres')

    input_dir = "ISIC_2020_Training_JPEG"
    output_dir = "ISIC_2020_Training_JPEG_lowres"

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all JPG files recursively
    jpg_files = list(input_path.rglob('*.jpg')) + list(input_path.rglob('*.jpeg'))

    # Prepare arguments for parallel processing
    args_list = [(img_file, input_path, output_path) for img_file in jpg_files]

    num_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(resize_image, args_list), total=len(jpg_files)))

    print(f"Resized {len(jpg_files)} images using {num_workers} cores.")


if __name__ == '__main__':
    if not (os.path.exists('isic_train.csv') and os.path.exists('isic_test.csv')):
        setup_train_test()

    if not os.path.exists('ISIC_2020_Training_JPEG_lowres'):
        make_low_res()

    train()