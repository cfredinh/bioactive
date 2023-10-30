from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "BioAct":    {"dataset_location": "BioAct"},
}
    
  
class BioAct(BaseSet):
    
    img_channels = 5
    max_pixel_value =  255
    is_multiclass = False
    is_masked_multilabel = True
    task = 'classification'
    num_classes = 29
    int_to_labels = {i: x for i, x in enumerate(range(num_classes))}
    target_metric = 'roc_auc'
    knn_nhood = 200    
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.init_stats()
        self.dataset_location = DATA_INFO["BioAct"]["dataset_location"]
        self.dataset_csv_path = DATA_INFO["BioAct"]["dataset_csv_path"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode

        self.data, self.int_to_labels, self.labels_to_int = self.get_data_as_list(dataset_location=dataset_params["data_location"],
                                                                                    dataset_csv_path=dataset_params["dataset_csv_path"],
                                                                                    assays=dataset_params["assays"],
                                                                                    split_numbers=dataset_params["data_split_numbers"])
        self.n_classes = len(self.int_to_labels)
        self.transform, self.resizing = self.get_transforms(albumentations=True)
        
    def init_stats(self):    
        self.mean = (0.067, 0.137, 0.137, 0.114, 0.123) 
        self.std  = (0.151, 0.151, 0.156, 0.135, 0.149)

    def get_image_data(self, path: str) -> str:

        img = Image.open(path)

        img = np.array(img)
        img = np.transpose(np.array(np.hsplit(img,5)), axes=[1,2,0])
        
        return img 

    def get_data_as_list(self, dataset_location, dataset_csv_path, assays, split_numbers={"train":[0,1,2,3],"val":[4],"test":[5]}):
        
        
        df = pd.read_csv(dataset_csv_path, index_col=0)
        
        df = df.dropna(subset="Metadata_Path")

        df["known"] = 0
        df["known"] = ((df[assays].values != 0).sum(axis=1) != 0)

        df = df[df.known == 1]

        int_to_labels = {i: x for i,x in enumerate(assays)}
        labels_to_int = {val: key for key, val in int_to_labels.items()}

        if self.mode == 'train':
            data = df[df.split_number.isin(split_numbers["train"])]
        elif self.mode in ['val', 'eval']:
            data = df[df.split_number.isin(split_numbers["val"])]
        else:
            data = df[df.split_number.isin(split_numbers["test"])]


        labels    = data[assays].values.tolist()
        cmpds     = data["Metadata_JCP2022"].values.tolist()
        img_paths = data["Metadata_Path"].values.tolist()
        full_img_path = [dataset_location + "/" + imp for imp in img_paths]

        num_samples = len(labels)
        if not all(len(l) == num_samples for l in [img_paths, full_img_path]):
            raise ValueError('Not all of the lists contain the same number of samples, indicating possible label miss match')
        
        data_list = [{'img_path': img_path, 'label': label, 'id': uid, 'dataset': self.name, 'cmpd': cmpd}
                     for img_path, label, uid, cmpd in zip(full_img_path, labels, img_paths, cmpds)]
                    
        return data_list, int_to_labels, labels_to_int


    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label    = torch.as_tensor(self.data[idx]['label'])
        uid      = self.data[idx]['id']
        cmpd     = self.data[idx]['cmpd']
        
        if os.path.isfile(img_path):
            img = self.get_image_data(img_path)
        else:
            print("MISSING IMAGE: ", img_path)
            return

        if self.resizing is not None:
            if isinstance(self.resizing, A.Resize):
                img = self.resizing(image=img)["image"]
            else:
                img = self.resizing(img)
            
        if self.transform is not None:
            if isinstance(self.transform, list):
                img = [tr(img) for tr in self.transform]
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    if isinstance(self.transform, A.Compose):
                        img = [self.transform(image=img)["image"] for _ in range(self.num_augmentations)]
                    else:
                        img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
        
        if self.mode == 'test':
            return img, label, uid, cmpd

        return img, label, uid
    
    