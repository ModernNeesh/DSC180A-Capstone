from tqdm import tqdm
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import torch
from sklearn.decomposition import PCA




def train_model(model, num_epochs, train_data, loss_func, optimizer, device = "cuda", return_losses = True, save = True, name = "params", path = "weights/"):
    losses = []
    num_epochs = 10
    for j in range(num_epochs):
        for i, batch in enumerate(tqdm(train_data)):

            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            print("Hanging on embeddings")
            embeddings = model(images)
            # Pass embeddings, labels, and mined triplets
            loss = loss_func(embeddings, labels)
            print("Hanging on gradient calculation")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    if save == True:
        if name == "params":
            name = f"Epochs-{num_epochs}_Loss-{loss_func.__name__}_Optimizer-{optimizer.__name__}.pth"
        torch.save(torch.save(model.state_dict(), path + name))
    
    if return_losses:
        return losses

def triplet_loss(margin = 0.19):
    def compute_triplet_loss(embeddings, labels):
        loss_func = TripletMarginLoss(margin=margin)
        miner = TripletMarginMiner(margin=margin, type_of_triplets="semihard")

        mined_triplets = miner(embeddings, labels)

        # Pass embeddings, labels, and mined triplets
        loss = loss_func(embeddings, labels, mined_triplets)
        return loss

    return compute_triplet_loss

def get_batch_embeddings(model, data, device = "cuda"):
    batch = next(iter(data))
    images = batch['pixel_values'].to(device)
    labels = batch['labels']

    embedding = model(images)

    return embedding, labels

def reduce_pca(embeddings, labels, dimensions = 2):  
    pca_model = PCA(n_components=dimensions)

    reduced_embedding = pca_model.fit_transform(embeddings.to("cpu").detach().numpy())
    labels_np = labels.detach().numpy()

    return reduced_embedding, labels_np