import os
import time
from operator import add
from random import random

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
from new_net.scconv_plus_mixpooling import *
from model import FENet
# from train_utils.utils import init_mask, rle_decode, load_data

""" Load the dataset """
# def load_data(path):
#     def load_names(path, file_path):
#         f = open(file_path, "r")
#         data = f.read().split("\n")[:-1]
#         images = [os.path.join(path, "images", name) + ".png" for name in data]
#         masks = [os.path.join(path, "masks", name) + "_mask.png" for name in data]
#         return images, masks
#
#     train_names_path = f"{path}/train.txt"
#     valid_names_path = f"{path}/test.txt"
#
#     train_x, train_y = load_names(path, train_names_path)
#     valid_x, valid_y = load_names(path, valid_names_path)
#
#     return (train_x, train_y), (valid_x, valid_y)

def load_data_DSB2018(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) + ".png" for name in data]
        masks = [os.path.join(path, "masks", name) + "_mask.png" for name in data]
        return images, masks

    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/test.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

""" Load the CVC_clinicDB dataset """
def load_data_CVC(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "img", name) for name in data]
        masks = [os.path.join(path, "mask", name) for name in data]
        return images, masks

    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/test.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

""" Load the Kvasir-SEG dataset """
def load_data_Kvasir(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) + ".jpg" for name in data]
        masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
        return images, masks

    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/test.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def calculate_metrics(y_true, y_pred, img):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred)
    if float(confusion[0,0] + confusion[0,1]) != 0:
        score_specificity = float(confusion[0,0]) / float(confusion[0,0] + confusion[0,1])
    else:
        score_specificity = 0.0

    return [score_jaccard, score_f1, score_recall, score_precision, score_specificity, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

class CustomDataParallel(torch.nn.DataParallel):
	""" A Custom Data Parallel class that properly gathers lists of dictionaries. """
	def gather(self, outputs, output_device):
		# Note that I don't actually want to convert everything to the output_device
		return sum(outputs, [])

if __name__ == "__main__":
    """ Seeding """
    # seeding(42)

    """ Load dataset """
    path = "./Kvasir-SEG"
    (train_x, train_y), (test_x, test_y) = load_data_Kvasir(path)

    """ Hyperparameters """
    size = (256, 256)
    num_iter = 10
    checkpoint_path = "new_net/checkpoint_new_net_kvasir.pth"

    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = FENet(in_channels=3, num_classes=1, base_c=32)
    # model = model.to(device)
    model = newNet(in_channels=3, num_classes=1, base_c=32).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    """ Testing """
    # origin_masks = init_mask(test_x, size)
    file = open("new_net/new_net_test_results_bilinear_KvasirSEG.csv", "w")
    file.write("Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        img_x = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        ## GT Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS Calculation """
            start_time = time.time()
            pred_y = torch.sigmoid(model(image))
            end_time = time.time() - start_time
            time_taken.append(end_time)

            score = calculate_metrics(mask, pred_y, img_x)
            metrics_score = list(map(add, metrics_score, score))

    """ Mean Metrics Score """
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    specificity = metrics_score[4]/len(test_x)
    acc = metrics_score[5]/len(test_x)
    f2 = metrics_score[6]/len(test_x)

    """ Mean Time Calculation """
    mean_time_taken = np.mean(time_taken)
    print("Mean Time Taken: ", mean_time_taken)
    mean_fps = 1/mean_time_taken

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Specificity: {specificity:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - Mean Time: {mean_time_taken:1.7f} - Mean FPS: {mean_fps:1.7f}")

    save_str = f"{jaccard:1.4f},{f1:1.4f},{recall:1.4f},{precision:1.4f},{specificity:1.4f},{acc:1.7f},{f2:1.7f},{mean_time_taken:1.7f},{mean_fps:1.7f}\n"
    file.write(save_str)


