from face_alignment import FaceAlignment, LandmarksType


def get_landmarks(image):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks(image)


def get_landmarks_folder(path_folder):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks_from_directory(path_folder)


class framLoader(Dataset):
    def __init__(self, root_dir):
        super(framLoader, self).__init__()
        self.root_dir = root_dir
        self.persons = glob.iglob(f"{self.root_dir}/*")
        self.folder_2_pic = [root
                             for root, dirs, files in os.walk(root_dir)
                             if len(files) >= 2]
        self.files_2_pic = list(itertools.chain.from_iterable([
            files for root, dirs, files in os.walk(root_dir)
            if len(files) >= 2]))

        self.transform_train = transforms.Compose([
            transforms.CenterCrop(115),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.names_to_int = {name.split('/')[-1]: torch.tensor([i])
                             for i, name in enumerate(
                                 glob.iglob(f"{self.root_dir}/*"))}

    def __getitem__(self, index):
        nom_anc = self.files_2_pic[index]

        possible_positive = os.listdir(f"{self.root_dir}/{nom_anc[:-9]}")
        possible_positive.remove(nom_anc)

        nom_pos = np.random.choice(possible_positive)

        nom_neg = self.files_2_pic[random.randint(0, len(self.files_2_pic))-1]
        while nom_neg[:-9] == nom_pos[:-9]:
            nom_neg = self.files_2_pic[random.randint(0,
                                                      len(self.files_2_pic))-1]

        label_anc = self.names_to_int[nom_anc[:-9]].squeeze(0)
        label_pos = self.names_to_int[nom_pos[:-9]].squeeze(0)
        label_neg = self.names_to_int[nom_neg[:-9]].squeeze(0)

        nom_anc = f"{self.root_dir}/{nom_anc[:-9]}/{nom_anc}"
        nom_pos = f"{self.root_dir}/{nom_pos[:-9]}/{nom_pos}"
        nom_neg = f"{self.root_dir}/{nom_neg[:-9]}/{nom_neg}"
        with open(nom_anc, 'rb') as a,\
                open(nom_pos, 'rb') as p, open(nom_neg, 'rb') as n:
            anchor = Image.open(a).convert('RGB')
            positive = Image.open(p).convert('RGB')
            negative = Image.open(n).convert('RGB')

        for trsf in [self.transform_train]:
            anchor = trsf(anchor)
            positive = trsf(positive)
            negative = trsf(negative)

        images = (anchor, positive, negative)
        labels = (label_anc, label_pos, label_neg)

        return images, labels

    def __len__(self):
        return len(self.files_2_pic)


datas = LfwTripletDataset(PATH_DATASET)
print(len(datas))
size_train = int(0.8 * len(datas))
size_valid = len(datas) - int(0.8 * len(datas))
train_datas, valid_datas = random_split(datas, (size_train, size_valid))

print(f"Nombre de données d'entrainement : {len(train_datas)}")
train_loader = DataLoader(train_datas, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_datas, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)

real_batch = next(iter(train_loader))
batch_view = torch.stack(real_batch[0], dim=1).view(-1, 3, 224, 224)
# %matplotlib inline
plt.figure(figsize=(25, 10))
plt.axis("off")
plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
plt.imshow(np.transpose(vutils.make_grid(batch_view, padding=2,
                                         nrow=6, normalize=True).cpu(),
                        (1, 2, 0)))
plt.show()
