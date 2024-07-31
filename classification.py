import torch

from models.model_genomic import SNN
from utils.utils import *
from utils.loss_func import NLLSurvLoss, FocalLoss
from models.TransMIL import TransMIL
from models.model_porpoise import MLP_Block, SNN_Block, GRL, BilinearFusion
import os
from sklearn.preprocessing import StandardScaler
import random
import argparse
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CMAF_cls(nn.Module):
    def __init__(self,
                 omic_input_dim,
                 path_input_dim=1024,
                 fusion='bilinear',
                 dropout=0.25,
                 n_classes=4,
                 scale_dim1=8,
                 scale_dim2=8,
                 gate_path=1,
                 gate_omic=1,
                 skip=True,
                 dropinput=0.10,
                 use_mlp=False,
                 size_arg="small",
                 stage_num=2
                 ):
        super(CMAF_cls, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 384], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 384]}
        self.n_classes = n_classes

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        # fc = []
        if dropinput:
            self.fc = nn.Sequential(
                *[nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                  nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        else:
            self.fc = nn.Sequential(
                *[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(size[1], size[2]), nn.ReLU(),
                  nn.Dropout(dropout)])
        attention_net = TransMIL()
        # fc.append(attention_net)
        self.attention_net = attention_net

        ### Constructing Genomic SNN
        if self.fusion is not None:
            if use_mlp:
                Block = MLP_Block
            else:
                Block = SNN_Block

            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(
                    *[nn.Linear(384 * 2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=384, dim2=384, scale_dim1=scale_dim1, gate1=gate_path,
                                         scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=size[2])
            else:
                self.mm = None

        self.GRL_layer = GRL()

        # self.discriminator = nn.Linear(384, 2)
        self.discriminator = nn.Sequential(*[nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 2)])
        self.stage_cls = nn.Linear(size[2], stage_num)
        # self.classifier_mm = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        self.fc = self.fc.to(device)
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.GRL_layer = self.GRL_layer.to(device)
        self.discriminator = self.discriminator.to(device)
        self.stage_cls = self.stage_cls.to(device)
        # self.classifier_mm = self.classifier_mm.to(device)

    def forward(self, x_path, x_omic):
        attn_ret = False
        h_omic = self.fc_omic(x_omic)

        h_path = self.fc(x_path)

        # multi modal discriminator
        if random.randint(0, 1) == 1:
            modality_label = torch.tensor([0, 1])
            reverse_feature = self.GRL_layer(torch.concat([h_path.mean(0).unsqueeze(0), h_omic]))
            # reverse_feature = self.GRL_layer(h_path.mean(0)).unsqueeze(0)
        else:
            modality_label = torch.tensor([1, 0])
            reverse_feature = self.GRL_layer(torch.concat([h_omic, h_path.mean(0).unsqueeze(0)]))

        modality_pred = self.discriminator(reverse_feature)

        if attn_ret:
            return self.attention_net(h_path, h_omic)
        h_path = self.attention_net(h_path, h_omic)

        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))

        res = self.stage_cls(h_mm)

        return res, modality_pred, modality_label


class Generic_WSI_Grade_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv', mode='omic', apply_sig=False,
                 shuffle=False, seed=7, print_info=True, n_bins=4,
                 patient_strat=False, label_col=None, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        slide_data = pd.read_csv(csv_path, low_memory=False)        
        subtype = ['astrocytoma', 'oligoastrocytoma', 'oligodendroglioma']
        slide_data.drop(slide_data.loc[~slide_data['stage'].isin(subtype)].index, inplace=True)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        # slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #    slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']


        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                # print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.stage_list = slide_data['stage'].unique().tolist()
        self.num_stage = len(self.stage_list)
        self.stage_dict = {}
        count = 0
        for i in self.stage_list:
            self.stage_dict.update({i: count})
            count += 1

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        # new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        # print("silde_data\n",slide_data[new_cols])
        self.slide_data = slide_data
        # metadata = ['disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id', 'age', 'site', 'survival_months', 'censorship', 'is_female', 'oncotree_code', 'train']
        metadata = ['disc_label', 'case_id', 'Unnamed: 0', 'label', 'slide_id', 'survival_months', 'stage',
                    'censorship']
        # metadata = ['disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id', 'survival_months', 'censorship']
        # self.metadata = slide_data.columns[:12]
        self.metadata = slide_data.columns[:8]

        print('self.metadata:\n',self.metadata)
        for col in slide_data.drop(self.metadata, axis=1).columns:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(col)
        # pdb.set_trace()

        # print('metadata:\n',self.metadata)
        assert self.metadata.equals(pd.Index(metadata))
        self.mode = mode
        self.cls_ids_prep()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            # self.signatures = pd.read_csv('./datasets_csv/signatures.csv')#修改了，因为signature在这个目录
            self.signatures = pd.read_csv('/data/run01/scz5319/xly/PORPOISE-master/datasets_csv')

        else:
            self.signatures = None


    def cls_ids_prep(self):
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        r"""

        """
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""

        """

        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def get_split_from_df(self, data: pd.DataFrame, split_key: str = 'train'):
        split = data
        if len(split) > 0:
            split = Generic_Split(split, metadata=self.metadata, mode=self.mode, signatures=self.signatures,
                                  data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict,
                                  num_classes=self.num_classes, stage_dict=self.stage_dict)
        else:
            split = None

        return split

    def return_splits(self):
        train_data = self.slide_data.loc[self.slide_data['stage'].isin(['astrocytoma', 'oligodendroglioma'])]
        test_data = self.slide_data.loc[self.slide_data['stage'].isin(['oligoastrocytoma'])]
        train_split = self.get_split_from_df(data=train_data, split_key='train')
        val_split = self.get_split_from_df(data=test_data, split_key='val')

        ### --> Normalizing Data
        print("****** Normalizing Data ******")
        scalers = train_split.get_scaler()
        train_split.apply_scaler(scalers=scalers)
        val_split.apply_scaler(scalers=scalers)
        # test_split.apply_scaler(scalers=scalers)
        ### <--
        return train_split, val_split  # , test_split
    
    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Grade_Dataset(Generic_WSI_Grade_Dataset):
    def __init__(self, data_dir, mode: str = 'omic', **kwargs):
        super(Generic_MIL_Grade_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):

        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        stage = torch.Tensor([self.stage_dict[self.slide_data['stage'][idx]]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'pathomic':
                    om = False
                    path_features = []
                    if om:
                        if len(slide_ids) == 1:
                            slide_id = slide_ids[0]
                            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features = wsi_bag
                        else:
                            for slide_id in slide_ids[:2]:
                                wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                                wsi_bag = torch.load(wsi_path)
                                path_features.append(wsi_bag)
                            path_features = torch.cat(path_features, dim=0)
                    else:
                        for slide_id in slide_ids:
                            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                        path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])

                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c, stage)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c

class Generic_Split(Generic_MIL_Grade_Dataset):
    def __init__(self, slide_data, metadata, mode,
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2, stage_dict=None):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.stage_dict = stage_dict
        self.stage_num = len(stage_dict)
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        #print("1\self.genomic_features",self.genomic_features)
        self.signatures = signatures
        self.genomic_dim = [2854, 2924, 5246, 5419] # gbmlgg
        self.slide_data.reset_index(inplace=True)
        print(slide_data.shape)

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        print(self.genomic_features.shape)
        scaler_rnaseq = StandardScaler().fit(self.genomic_features.iloc[:, :self.genomic_dim[0]])
        scaler_mut = StandardScaler().fit(self.genomic_features.iloc[:, self.genomic_dim[0]:self.genomic_dim[1]])
        scaler_methy = StandardScaler().fit(self.genomic_features.iloc[:, self.genomic_dim[1]:self.genomic_dim[2]])
        scaler_cnv = StandardScaler().fit(self.genomic_features.iloc[:, self.genomic_dim[2]:self.genomic_dim[3]])

        return {'rnaseq': scaler_rnaseq, 'mut': scaler_mut, 'methy': scaler_methy, 'cnv': scaler_cnv}

    def apply_scaler(self, scalers: dict = None):
        scaler_rna = scalers['rnaseq'].transform(self.genomic_features.iloc[:, :self.genomic_dim[0]])
        scaler_mut = scalers['mut'].transform(self.genomic_features.iloc[:, self.genomic_dim[0]:self.genomic_dim[1]])
        scaler_methy = scalers['methy'].transform(self.genomic_features.iloc[:, self.genomic_dim[1]:self.genomic_dim[2]])
        scaler_cnv = scalers['cnv'].transform(self.genomic_features.iloc[:, self.genomic_dim[2]:self.genomic_dim[3]])
        transformed_omic = np.hstack([scaler_rna, scaler_mut, scaler_methy, scaler_cnv])
        transformed = pd.DataFrame(transformed_omic)
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed



parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir', type=str, default='/data/run01/scz5319/xly/PORPOISE-master/selec_wsi_pt',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir', type=str, default='./former_fusion', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits', type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir', type=str, default='tcga_coad',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--task_type', type=str, default='survival', choices=['multi_task', 'survival'],
                    help='Multi task or survival')
parser.add_argument('--alignment', action='store_true', default=False,
                    help='Using  adversarial training to align feature')

### Model Parameters.
parser.add_argument('--model_type', type=str, default='cmaf', help='Type of model (Default: cmaf)')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'pathomic_fast', 'cluster', 'coattn'],
                    default='pathomic', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='concat',
                    help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig', action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats', action='store_true', default=False,
                    help='Use genomic features as tabular features.')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='big', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='big', help='Network size of SNN model')

parser.add_argument('--n_classes', type=int, default=4)

### PORPOISE
parser.add_argument('--apply_mutsig', action='store_true', default=False)
# parser.add_argument('--apply_mutsig', action='store_true', default=True)
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

### CLAM-Specific Parameters
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

args = parser.parse_args()

args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)
args = get_custom_exp_code(args)

if 'survival' in args.task:
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_lung'
    else:
        combined_study = study

    study_dir = '%s_20x_features' % combined_study
    print('csvpath', './%s/%s_all_stage.zip' % (args.dataset_path, study))

dataset = Generic_MIL_Grade_Dataset(csv_path='./%s/%s_all_stage.zip' % (args.dataset_path, study),
                                    mode=args.mode,
                                    apply_sig=args.apply_sig,
                                    data_dir=os.path.join(args.data_root_dir, study_dir),
                                    shuffle=False,
                                    seed=args.seed,
                                    print_info=True,
                                    patient_strat=False,
                                    n_bins=4,
                                    label_col='survival_months')

if __name__ == '__main__':
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    train, test = dataset.return_splits()
    args.omic_input_dim = train.genomic_features.shape[1]

    model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes,
                  'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1,
                  'scale_dim2': args.scale_dim2,
                  'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim,
                  'use_mlp': args.use_mlp, 'stage_num': 2}
    model = CMAF_cls(**model_dict)
    print(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    cls_loss = torch.nn.CrossEntropyLoss()

    if args.reg_type == 'omic':
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5, last_epoch=-1)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train, training=True, testing=args.testing,
                                    weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(test, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    for epoch in range(args.max_epochs):
        ##### training ######
        model.train()
        train_loss_stage = 0.
        train_loss = 0.
        pos = 0

        for batch_id, (data_WSI, data_omic, label, event, c, stage) in enumerate(train_loader):
            T = 1 / (batch_id + 1)
            data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
            stage = stage.to(device)
            predict, modality_pred, modality_label = model(x_path=data_WSI, x_omic=data_omic)
            modality_label = modality_label.to(device)
            pred_stage = predict.argmax()
            if pred_stage.item() == stage.item():
                pos += 1
            loss_stage = loss_fn(predict, stage)
            loss_modality = cls_loss(modality_pred, modality_label)
            loss = loss_stage + T * loss_modality
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * args.lambda_reg
            train_loss_stage += loss_stage.item()
            train_loss += loss_value + loss_reg

            if (batch_id + 1) % 100 == 0:
                print('batch {}, loss: {:.4f}, label: {}'.format(
                    batch_id, loss_value + loss_reg, stage.detach().cpu().item()))

            # backward pass
            loss = loss / args.gc + loss_reg
            loss.backward()

            if (batch_id + 1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss_stage /= len(train_loader)
        train_loss /= len(train_loader)
        stage_acc = pos / len(train_loader)
        print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch,
                                                                                                 train_loss_stage,
                                                                                                 train_loss,
                                                                                                 stage_acc))

    ##### testing ######
    print('Testing')
    model.eval()
    val_loss, val_loss_stage = 0., 0.
    all_pred = []
    all_censorships = []
    all_event_times = []
    all_patient = []
    slide_ids = val_loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_id, (data_WSI, data_omic, label, event, c, stage) in enumerate(val_loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        slide_id = slide_ids.iloc[batch_id]
        with torch.no_grad():
            T = 1 / (batch_id + 1)
            data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
            stage = stage.to(device)
            predict, modality_pred, modality_label = model(x_path=data_WSI, x_omic=data_omic)
            modality_label = modality_label.to(device)

            pred_stage = predict.argmax()
        event_time = event.item()
        censor = c.item()
        all_pred.append(pred_stage.item())
        all_censorships.append(censor)
        all_event_times.append(event_time)
        all_patient.append(np.array(slide_id))
        patient_results.update({'slide_id': np.array(slide_id), 'pred': pred_stage.item(), 'survival': event_time, 'censorship': censor})

        print(patient_results)
    # df = pd.DataFrame(patient_results, index=list(range(len(val_loader))))
    # print('Saving result at {}'.format(os.path.join(args.results_dir, 'result.csv')))
    df = pd.DataFrame()
    df['patient_id'] = all_patient
    df['pred'] = all_pred
    df['censorship'] = all_censorships
    df['event_time'] = all_event_times
    df.to_csv(os.path.join(args.results_dir, 'result{}.csv'.format(args.max_epochs)))
    

