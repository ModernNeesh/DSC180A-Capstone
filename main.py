#Importing packages
import argparse
import os
from torch.utils.data import DataLoader
import torch

#library functions
import helper_code.dataloading as dataloading
import helper_code.data_vis as data_vis
import helper_code.model_functions as model_functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple calculator script.")
    parser.add_argument("--camera-data-dir", default="camera_data/",
                        help="The location to store the camera data")
    
    parser.add_argument("--labels-csv-name", default = "coronado_hills_data.csv",
                        help="The csv file with image paths and labels, imported from Label Studio")
    
    parser.add_argument("--image-dir", default = "camera_data/images/",
                        help="The directory to save images to")
    
    parser.add_argument('--download-imgs', dest = "image_download", action='store_true', help = "Download new image data")
    parser.add_argument('--keep-imgs', dest='image_download', action='store_false', help = "Use existing image data")

    parser.add_argument("--model_path", default = "weights/", 
                        help = "The directory to store the model to, or load it from")
    
    parser.add_argument("--model_name", default = "model_weights_camera_11-17-25.pth", 
                        help = "Name of the model's weights")
    
    parser.add_argument('--train-model', dest = "model_train", action='store_true', help = "Train a new model")
    parser.add_argument('--load-model', dest='model_train', action='store_false', help = "Load an existing model's weights")
    parser.add_argument('--device', default = "cuda", choices = ["cuda", "cpu"], help = "Which device to use")

    parser.add_argument("--collection-dir", default = "embedding_data/", 
                        help = "The directory to save embeddings to")

    parser.add_argument('--get-embeddings', dest = "embedding_save", action='store_true', help = "Save the embedding data")
    parser.add_argument('--load-embeddings', dest='embedding_save', action='store_false', help = "Load existing embedding data")
    


    parser.set_defaults(image_download=False, model_train=False, embedding_save = True)
    
    args = parser.parse_args()

#Load the data
labels_csv = args.camera_data_dir + args.labels_csv_name
data = dataloading.get_data(labels_csv, args.image_dir, replace_images = args.image_download)

train, val, test = dataloading.get_train_val_test(data = data, output_csvs=True)

train_dataset, val_dataset, test_dataset = dataloading.get_datasets(train, val, test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)


#Train the model
encoder = model_functions.create_encoder()
encoder.to(args.device)


if args.model_train:
    num_epochs = 1
    loss_func = model_functions.triplet_loss(margin=0.18)
    optimizer = optim.Adam(encoder.parameters(), lr=2e-5) 

    model_functions.train_model(encoder, train_data=train_dataloader, 
                                num_epochs=num_epochs, loss_func=loss_func, 
                                optimizer=optimizer, name = args.model_name, path = args.model_path)
else:
    encoder.load_state_dict(torch.load(args.model_path + args.model_name, weights_only=True))
encoder.eval()

#Save embeddings

collection_name = 

dataloading.save_full_embeddings(encoder, train_dataloader, 
                     "train_embeddings", persist_directory = args.collection_dir, 
                     device = args.device)

dataloading.save_full_embeddings(encoder, val_dataloader, 
                     "val_embeddings", persist_directory = args.collection_dir, 
                     device = args.device)

train_embeddings, train_labels, _, _ = dataloading.load_full_embeddings(train, "train_embeddings", persist_directory = args.collection_dir)

val_embeddings, val_labels, _, _ = dataloading.load_full_embeddings(val, "val_embeddings", persist_directory = args.collection_dir)

train_embedding_dataset = dataloading.CustomEmbeddingDataset(train_embeddings, train_labels)
train_embedding_dataloader = DataLoader(train_embedding_dataset, batch_size=32, shuffle=True, pin_memory=True)

val_embedding_dataset = dataloading.CustomEmbeddingDataset(val_embeddings, val_labels)
val_embedding_dataloader = DataLoader(val_embedding_dataset, batch_size=32, shuffle=True, pin_memory=True)



