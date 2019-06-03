from settings import NB_EPOCHS
from preprocess import get_data_loader
from Archi import Embedder
train_loader = get_data_loader()
emb = Embedder()
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        print("test : ", i_epoch, i_batch)
        gt_im_tensor, gt_im_landmarks_tensor, context_tensors = batch
        plop = emb(context_tensors)
        print(plop.size())
