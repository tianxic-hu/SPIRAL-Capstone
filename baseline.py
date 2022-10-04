import glob
import os
import time

from monai.data import CacheDataset, DataLoader, write_nifti
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.blocks import Convolution
import monai.transforms as tf
from monai.utils import set_determinism
import torch
import sys
################################################################################
### DATA SETUP
################################################################################
device = torch.device("cpu")

# get path to $SLURM_TMPDIR
base_dir = "/Users/cindyhu/PycharmProjects/SPIRAL-baseline"

out_dir = os.path.join(base_dir, 'out')
sys.stdout = open(os.path.join(out_dir, 'console_output.txt'), 'w')
print(f'Files will be saved to: {out_dir}')

set_determinism(seed=0)

# get data files
data_dir = os.path.join(base_dir, 'Capstone_Project_Data')

print('Loading files...')

# training files (remove validation subjects after)
train_cta1 = sorted(glob.glob(os.path.join(data_dir, '*', 'CTA1.nii*')))
train_cta2 = sorted(glob.glob(os.path.join(data_dir, '*', 'CTA2*.nii*')))
train_cta3 = sorted(glob.glob(os.path.join(data_dir, '*', 'CTA3*.nii*')))
train_spiral = sorted(glob.glob(os.path.join(data_dir, '*', 'Spiral*.nii')))
train_labels = sorted(glob.glob(os.path.join(data_dir, '*', '*clot.nii*')))
train_files = [
    {'train_cta1': train_cta1, 'train_cta2': train_cta2, 'train_cta3': train_cta3, 'train_spiral': train_spiral, 'label': label_name}
    for train_cta1, train_cta2, train_cta3, label_name in zip(train_cta1, train_cta2, train_cta3, train_labels)
]
print("train_file_sample: ", train_files)
# validation files
val_subj = '013'
val_files = [train_file for train_file in train_files if val_subj in train_file['label'].split('/')[-1]]
train_files = [train_file for train_file in train_files if val_subj not in train_file['label'].split('/')[-1]]

print(f'Total {len(train_files)} subjects for training.')
print(f'Total {len(val_files)} subjects for validation.')

################################################################################
### DEFINE TRANSFORMS
################################################################################

# train transforms
train_transforms = tf.Compose([
    tf.LoadImaged(keys=['train_cta2', 'label']),
    tf.AddChanneld(keys=['train_cta2', 'label']),
    tf.Spacingd(keys=['train_cta2', 'label'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    tf.ResizeWithPadOrCropd(keys=['train_cta2', 'label'], spatial_size=(250, 250, 150)),
    tf.ToTensord(keys=['train_cta2', 'label']),
    tf.DeleteItemsd(keys=['image_transforms', 'label_transforms'])
])
# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(keys=['train_cta2', 'label']),
    tf.AddChanneld(keys=['train_cta2', 'label']),
    tf.Spacingd(keys=['train_cta2', 'label'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    tf.ResizeWithPadOrCropd(keys=['train_cta2', 'label'], spatial_size=(250, 250, 150)),
    tf.ToTensord(keys=['train_cta2', 'label']),
])

################################################################################
### DATASET AND DATALOADERS
################################################################################

# train dataset
train_ds = CacheDataset(data=train_files, transform=train_transforms,
                        cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# valid dataset
val_ds = CacheDataset(data=val_files, transform=val_transforms,
                      cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

################################################################################
### MODEL AND LOSS
################################################################################

model = Convolution(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    adn_ordering="ADN",
    act=("prelu", {"init": 0.2}),
    dropout=0.1,
    norm=("layer", {"normalized_shape": (250, 250, 150)}),
).to(device)
loss_function = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

################################################################################
### TRAINING LOOP
################################################################################

# general training params
epoch_num = 10
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
val_loss_values = list()
metric_values = list()
epoch_times = list()
total_start = time.time()

# inference params for patch-based eval
roi_size = (250, 250, 150)
sw_batch_size = 2

print(f'Starting training over {epoch_num} epochs...')
for epoch in range(epoch_num):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        inputs, labels= (
            batch_data["train_cta2"].to(device),
            batch_data["label"].to(device),
        )

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
              f" step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    print(f"time consuming of epoch {epoch + 1} is: {epoch_time:.4f}")

    # validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            step = 0
            metric_sum = 0
            metric_count = 0
            for val_data in val_loader:
                step += 1
                #print("val_data is : ", val_data["image_adc"])
                val_inputs, val_labels = (
                    val_data["train_cta2"].to(device),
                    val_data["label"].to(device),
                )
                # print("val_inputs, val_labels: ",  val_inputs, val_labels)
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                # val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()

            val_loss /= step
            val_loss_values.append(val_loss)
            #TBD: compute metric for validation

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
      f" total time: {(time.time() - total_start):.4f}")
