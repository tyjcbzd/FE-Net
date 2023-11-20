import datetime
import time
import albumentations as A
from torch.utils.data import Dataset
from dataset import DATASET
from utils import *
from new_net.scconv_plus_mixpooling import *
from loss import DiceBCELoss
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, optimizer, data_loader, device, loss_fn):
    epoch_loss = 0
    model.train()

    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(image)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss

def evaluate(model, data_loader, device, loss_fn):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)

            y_pred = model(image)
            loss = loss_fn(y_pred, target)
            epoch_loss += loss.item()


    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss


if __name__ == "__main__":


    """ Directories """
    create_dir("files")

    train_log_path = "where you put log file"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("where you put log file", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 4
    num_epochs = 30
    lr = 1e-4
    # segmentation nun_classes + background
    num_classes = 1

    checkpoint_path = "where you put your pth file"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Dataset """
    path = "dataset path"

    (train_x, train_y), (valid_x, valid_y) = load_data_Kvasir(path)
    train_x, train_y = shuffling(train_x, train_y)


    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 监控训练过程
    writer = SummaryWriter('log file for tensorboard')

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)

    model = newNet(in_channels=3, num_classes=num_classes, base_c=32).to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, loss_fn)

        valid_loss, dice, jac = evaluate(model, val_loader, device, loss_fn)

        writer.add_scalar('Loss/train', train_loss, global_step=(epoch + 1))
        writer.add_scalar('mIOU/train', jac, global_step=(epoch + 1))
        writer.add_scalar('Dice/train', dice, global_step=(epoch + 1))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        data_str += f'\t Dice: {dice:.3f}\n'
        data_str += f'\t mIOU: {jac:.3f}\n'
        print_and_save(train_log_path, data_str)