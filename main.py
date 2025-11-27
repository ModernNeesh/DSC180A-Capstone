#Importing packages
import argparse
import os

#library functions
import dataloading
import data_vis
import model_functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple calculator script.")
    parser.add_argument("--camera_data_dir", default="camera_data/",
                        help="The location to store the camera data")
    
    parser.add_argument("--labels_csv_name", default = "coronado_hills_data.csv",
                        help="The csv file with image paths and labels, imported from Label Studio")
    
    parser.add_argument("--image_dir", default = "camera_data/images/",
                        help="The directory to save images to")
    
    parser.add_argument('--download-imgs', action='store_true', help = "Download new image data")
    parser.add_argument('--keep-imgs', dest='image_download', action='store_false', help = "Use existing image data")
    parser.set_defaults(image_download=True)

    parser.add_argument("--model_path", default = "weights/", 
                        help = "The directory to store the model to")
    

#Load the data

#Train the model

#