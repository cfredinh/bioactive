from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "BioAct":    {"dataset_location": "BioAct"},
              "Hofmarcher": {"dataset_location": "Hofmarcher"},
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
    
    


class Hofmarcher(BaseSet):
    
    # To download the Hofmarcher data
    # Download the individual zip files from: https://ml.jku.at/software/cellpainting/dataset/
    # Label data etc. can be found at: https://github.com/ml-jku/hti-cnn/tree/master/labels

    img_channels = 5
    max_pixel_value =  255
    is_multiclass = False
    is_masked_multilabel = True
    task = 'classification'
    num_classes = 209
    int_to_labels = {i: x for i, x in enumerate(range(num_classes))}
    target_metric = 'roc_auc'
    knn_nhood = 200    
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.init_stats()
        self.dataset_location = DATA_INFO["Hofmarcher"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode

        self.data, self.int_to_labels, self.labels_to_int = self.get_data_as_list(dataset_location=dataset_params["data_location"],
                                                                                    split_number=dataset_params["data_split_number"])
        self.n_classes = len(self.int_to_labels)
        self.transform, self.resizing = self.get_transforms(albumentations=True)
        
    def init_stats(self):    
        self.mean = (0.1706, 0.1415, 0.1827, 0.1709, 0.0969)
        self.std  = (0.0880, 0.0861, 0.1033, 0.0863, 0.1110)

    def get_image_data(self, path: str) -> str:

        samp = np.load(path)
        img = np.array(samp["sample"])
        
        return img 


    def get_data_as_list(self, dataset_location, split_number=1):
        

        split_number = str(split_number) 
        if self.mode == 'train':
            df = pd.read_csv(dataset_location+f"/datasplit{split_number}-train.csv")
        elif self.mode in ['val', 'eval']:
            df = pd.read_csv(dataset_location+f"/datasplit{split_number}-val.csv")
        else:
            df = pd.read_csv(dataset_location+f"/datasplit{split_number}-test.csv")

        col_index = pd.read_csv(dataset_location+"/column-assay-index.csv", sep=",", header=0)
        row_index = pd.read_csv(dataset_location+"/row-compound-index.csv", sep=",", header=0)
        label_matrix = mmread(dataset_location+"/label-matrix.mtx").tocsr()
        # --
        self.label_matrix = label_matrix
        self.row_index = row_index
        self.col_index = col_index
        self.label_dict = dict(zip(df.SAMPLE_KEY, df.ROW_NR_LABEL_MAT))
        self.labels     = dict(zip(df.SAMPLE_KEY, df.ROW_NR_LABEL_MAT))
        self.n_classes = label_matrix.shape[1]
        
        int_to_labels = {i: x for i,x in enumerate(col_index.ASSAY_ID.values)}
        labels_to_int = {val: key for key, val in int_to_labels.items()}



        labels    = df.ROW_NR_LABEL_MAT.values.tolist()
        cmpds     = df["BROAD_ID"].values.tolist()
        img_paths = df["SAMPLE_KEY"].values.tolist()
        full_img_path = [dataset_location + "/" + imp + ".npz" for imp in img_paths]

        data_list = [{'img_path': img_path, 'label': self.label_matrix[label].toarray()[0], 'id': uid, 'dataset': self.name, 'cmpd': cmpd}
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
    
    