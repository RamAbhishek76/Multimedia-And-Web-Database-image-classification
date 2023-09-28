# Functionality: Plots the images produced in the output as a matplotlib figure
import matplotlib.pyplot as plt
import numpy
import torch

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features


def output_plotter(feature_name, input_image, feature_vals, feature_val_keys, k):
    # Create a figure where the images can be plotted
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 6, 1)

    fig.suptitle(feature_name + ' query top 10 outputs for input image ID ' +
                 str(input_image["image_id"]), fontsize=16)

    # Plotting the input image in the figure
    plt.imshow((numpy.squeeze(torch.tensor(
        numpy.array(input_image["image"])).permute(1, 2, 0))))
    plt.title("Query Image ID: " + str(input_image["image_id"]))

    print("Results for " + feature_name + ":")

    # Plotting the images produced by the query result
    for i in range(k):
        print(feature_vals[feature_val_keys[i]])
        image = collection.find_one(
            {"image_id": feature_vals[feature_val_keys[i]]})
        img = torch.tensor(numpy.array(image["image"]))

        fig.add_subplot(2, 6, i + 1)
        plt.imshow((numpy.squeeze(img.permute(1, 2, 0))))
        plt.axis('off')
        plt.title("Result ID: " + str(image["image_id"] +
                  "\nDistance: " + str(round(feature_val_keys[i], 4))))

    plt.show()


def task_2_output_plotter(feature_name, label, feature_vals, feature_val_keys, k):
    # Create a figure where the images can be plotted
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 6, 1)

    fig.suptitle(feature_name + ' query top 10 outputs for input image ID ' +
                 str(label), fontsize=16)

    im = collection.find_one({"target": label})
    # Plotting the input image in the figure
    plt.imshow((numpy.squeeze(torch.tensor(
        numpy.array(im["image"])).permute(1, 2, 0))))
    plt.title("Query Image label: " + str(label))

    print("Results for " + feature_name + ":")

    # Plotting the images produced by the query result
    for i in range(k):
        print(feature_vals[feature_val_keys[i]])
        image = collection.find_one(
            {"image_id": feature_vals[feature_val_keys[i]]})
        img = torch.tensor(numpy.array(image["image"]))

        fig.add_subplot(2, 6, i + 1)
        plt.imshow((numpy.squeeze(img.permute(1, 2, 0))))
        plt.axis('off')
        plt.title("Result ID: " + str(image["image_id"] +
                  "\nDistance: " + str(round(feature_val_keys[i], 4))))

    plt.show()
