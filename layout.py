
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import cv2

import numpy as np

import os
import sys

import argparse
sys.path.append("spvloc")
from spvloc_train_test import main_start

def main(image_path):

    os.chdir("../")
    print(os.getcwd())


    image = cv2.imread(image_path)




    cv2.imwrite(os.path.join("dataset/scene_03457/2D_rendering/141477/perspective/full/0", "rgb_rawlight.png"),image)
    decoded_img_path = "results/results_ps1200_ch1400_ns1_rad1400_h300/03457/000/02b_dec_n.png"
    os.chdir("spvloc")

    main_start()
    os.chdir("../")




    k = encoded_image_scores(decoded_img_path)

    image = color_clustering(decoded_img_path,k)

    cv2.imshow("Image", image)
    cv2.waitKey(0)





def optimal_number_of_clusters(wcss, silhouette_scores, cluster_range):

    wcss_differences = np.diff(wcss)

    elbow_point = np.argmax(wcss_differences) + 2  # +2 çünkü np.diff uzunluğu bir eksik olur


    best_silhouette_k = cluster_range[np.argmax(silhouette_scores)]

    if abs(best_silhouette_k - elbow_point) <= 1:
        return elbow_point
    else:
        return best_silhouette_k

def encoded_image_scores(image_path):
    image = cv2.imread(image_path)
    image = image[59:202, 0:319]
    image = cv2.resize(image, (320, 320))

    image = Image.fromarray(image)

    image = image.convert('RGB')

    image_array = np.array(image)

    pixels = image_array.reshape(-1, 3)

    scaler = StandardScaler()
    pixels_normalized = scaler.fit_transform(pixels)

    pixels_normalized_sample = shuffle(pixels_normalized, random_state=42)[:1000]

    k_values = range(2, 6)


    wcss = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pixels_normalized)

        wcss.append(kmeans.inertia_)

        silhouette_sample = kmeans.predict(pixels_normalized_sample)
        silhouette_scores.append(silhouette_score(pixels_normalized_sample, silhouette_sample))
    optimal_k = optimal_number_of_clusters(wcss, silhouette_scores, list(k_values))

    return optimal_k



def color_clustering(image_path,k):
    image = cv2.imread(image_path)
    image = image[59:202, 0:319]
    image = cv2.resize(image, (320,320))



    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)



    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)


    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

    return gray_segmented

def create_key(*args):
    return "_".join(map(str, args))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Layout Project Parser')
    parser.add_argument('--image_path', type=str,help='path to image file',default="")
    args = parser.parse_args()

    main(args.image_path)

