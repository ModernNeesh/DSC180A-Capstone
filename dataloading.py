import pandas as pd
import urllib.request
import os
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



#Get the label, image URL, and annotation id of the images
def get_data_urls(labels_csv):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form
    
    """
    data = pd.read_csv(labels_csv)
    data = data.get(["choice", "image", "annotation_id"]).fillna(0)

    data["choice"] = (data["choice"] != 0).astype(int)

    return data


#Download the images from their URLs
def download_images(data):
    """
    data: The dataset of image urls, usually the output of the get_data_urls function
    """
    for i in tqdm(range(len(data))):
            img_path = f"camera_data/images/img_{data.iloc[i]['annotation_id']}.jpg"
            urllib.request.urlretrieve(data.iloc[i]['image'], img_path)
            if i%9 == 0:
                time.sleep(2)


#Get a dataframe of the image paths and their corresponding annotation ids
def get_images_df(image_dir):
    """
    img_dir: The directory of the images
    """
    img_paths = image_dir + pd.Series(os.listdir(image_dir))
    
    annotations = img_paths.str.extract(r"(\d+)")[0].astype(int)

    return pd.DataFrame({"annotation_id" : annotations, 
                         "img_path" : img_paths})


#Gather the data for image labels and paths into one big DataFrame
def get_data(labels_csv, image_dir, replace_images = False):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form

    image_dir: The directory for the images to be saved to/gathered from

    replace_images: Whether to replace the images currently in the directory
    """
    url_data = get_data_urls(labels_csv)

    if(replace_images):
        for img in os.listdir(image_dir):
            os.remove(image_dir + img)
        download_images(url_data)

    image_df = get_images_df(image_dir)

    full_data = url_data.merge(image_df, left_on = "annotation_id", right_on="annotation_id")

    full_data = full_data.get(["choice", "img_path", "annotation_id"])

    return full_data

#Split a DataFrame into train, validation, and test splits, or pull those splits from existing CSV files
def get_train_val_test(data = None, df_dir = None, output_csvs = False, csv_output_dir = "camera_data/dataframes/"):
    """
    data: The DataFrame to split

    df_dir: The directory to the existing CSV files, if they exist

    output_csvs: Whether to output the train, validation, and test dataframes to a new file

    csv_output_dir: Where to output the dataframes
    """
    if df_dir is not None:
        train = pd.read_csv(df_dir + "train")
        val = pd.read_csv(df_dir + "val")
        test = pd.read_csv(df_dir + "test")
    else:
        if type(data) == type(None):
            raise ValueError("Must include dataframe to split")
        # X: features, y: target variable
        X_train_val, X_test, y_train_val, y_test = train_test_split(data['img_path'], data['choice'], test_size=0.2, random_state=147)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=147)

        train = pd.DataFrame({"img_directory" : X_train, "label" : y_train}).reset_index(drop=True)
        val = pd.DataFrame({"img_directory" : X_val, "label" : y_val}).reset_index(drop=True)
        test = pd.DataFrame({"img_directory" : X_test, "label" : y_test}).reset_index(drop=True)

        if(output_csvs):
            train.to_csv(csv_output_dir + "train")
            val.to_csv(csv_output_dir + "val")
            test.to_csv(csv_output_dir + "test")
    
    return train, val, test


#Defining a dataset class to import the images. We resize them to 224 by 224 since that's what the model expects, but make no other transformations.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts PIL to Tensor
])

class CustomImageDataset(Dataset):
    def __init__(self, data_df, transform = None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["img_directory"]
        label = int(self.data.iloc[idx]["label"])
    
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {"pixel_values": image,
                "labels": label}
    

def get_datasets(train_df, val_df, test_df):
    #Creating the dataset and loading it into batches with the DataLoader class
    train_dataset = CustomImageDataset(train_df, transform=transform)
    val_dataset = CustomImageDataset(val_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)

    return train_dataset, val_dataset, test_dataset
