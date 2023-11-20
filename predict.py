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


