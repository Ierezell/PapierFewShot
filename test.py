from preprocess import get_data_loader

train_loader, nb_pers = get_data_loader()
print(train_loader[0])
