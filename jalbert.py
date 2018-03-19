from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 1e10
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import os.path as osp
import numpy as np
import glob

from tqdm import tqdm
import errno    
import os
import pickle

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and osp.isdir(path):
            pass
        else:
            raise

def change_tile(tile, new_width, new_height, memory_offset):
    tup = tile[0]
    return [(tup[0],) + ((0,0,new_width, new_height),) + (tup[-2]+memory_offset,) + (tup[-1],)]

def read_line_portion(img_path,x,y,w,h,i):
    img_pil = Image.open(img_path)
    W = img_pil.size[0]
    img_pil.size=(w,1)
    memory_offset = (x+i)*3*W+3*y
    img_pil.tile = change_tile(img_pil.tile,w,1,memory_offset)
    #print(img_pil.tile)
    #print(img_pil.size)
    return img_pil

def read_from_memory(img_path,x,y,w,h):
    result = Image.new('RGB',(w,h))
    for i in range(h):
        a = read_line_portion(img_path, x,y,w,h,i)
        result.paste(a,(0,i))
    return result

def show_thumbnail(img_pil, max_size_thumbnail = 200.):
    img_pil_thumbnail = img_pil.copy()
    size = img_pil.size
    max_size_img = float(max(size))
    new_size =  tuple((max_size_thumbnail/max_size_img*np.asarray(size)).astype(int))
    img_pil_thumbnail.thumbnail(new_size, Image.ANTIALIAS)
    return img_pil_thumbnail

def pil_resize(img_pil, new_max_size = 1000.):
    size = img_pil.size
    max_size_img = float(max(size))
    new_size =  tuple((new_max_size/max_size_img*np.asarray(size)).astype(int))
    img_pil_resized = img_pil.resize(new_size, Image.NEAREST)
    return img_pil_resized

def convert_to_ppm(img_path,verbose=False):
    
    if osp.isfile(img_path):
        if img_path[-3:]!="ppm":
            if verbose:
                print('not ppm')
            if osp.isfile(img_path+".ppm"):
                if verbose:
                    print('converted file already exists')
                pass
            else:
                if verbose:
                    print("conversion... ["+"convert" +img_path+" "+img_path+".ppm"+"]")
                os.system("convert " +img_path+" "+img_path+".ppm")  
                if verbose:
                    print("conversion done !")
        else:
            if verbose:
                print("file already converted")
    else:
        if verbose:
            print("file does not exist")

def purge_ppm(folder):
    os.system("rm "+folder+"/*.ppm")
def list_all_images(folder,fileExtensions = ['jpg','jpeg','JPG','JPEG']):
    directoryPath  = folder+"/*."
    listOfFiles    = []
    list_files = []
    for extension in fileExtensions:
        listOfFiles.extend( glob.glob( directoryPath + extension ))
    return listOfFiles



def convert_to_ppm_by_folder(folder_in, folder_out=None, fileExtensions = ['jpg','jpeg','JPG','JPEG'], verbose = False):
    if folder_out is None:
        folder_out = folder_in+"/ppms/"
    mkdir_p(folder_out)
    #in
    img_paths_in = list_all_images(folder_in, fileExtensions)
    img_names_in = [osp.splitext(osp.splitext(osp.basename(img_path))[0])[0] for img_path in img_paths_in]
    #out
    img_paths_out = list_all_images(folder_out, ['ppm'])
    img_names_out = [osp.splitext(osp.splitext(osp.basename(img_path))[0])[0] for img_path in img_paths_out]
    
    if verbose :
        print("in")
        print(img_paths_in)
        print(img_names_in)
        print("out")
        print(img_paths_out)
        print(img_names_out)
    for img_name_in_id,img_name_in in enumerate(tqdm(img_names_in,ncols=150,desc='ppm conversion')):
        if img_name_in not in img_names_out:
            if verbose:
                print("converting "+img_name_in+"...")
            os.system("convert " +img_paths_in[img_name_in_id]+" "+folder_out+img_name_in+".ppm")

## objectif : connaître k et s' en fonction de W, w'(scale,dim) et s(overlap,w') :
# overlap, c'est un overlap minimal qui peut s'aggrandir pour bien couvrir l'image

from math import ceil, floor
import csv

def make_spk(W, w=1000, overlap = 0.5, scale = 1):
    """ input :
        W : dimension de l'image
        w : dimension du crop
        overlap : recouvrement dans le décalage des crops avec strategie de sliding window
        scale : zoom 
        
        output :
        sp (=s') : recalcul du stride s pour améliorer le recouvrement de l'image
        k : nombre de crop que l'on peut faire le long de la dimension considérée
        
        Dans le principe, avec le stride de base s le crop k sort de l'image, 
        donc on diminue légèrement le stride pour que le k-ième crop arrive dans l'image.
        
        La conséquence directe est que l'overlap augmente un peu.
        """
    wp=ceil(w*scale) #pour s'assurer d'avoir des pixels 
    s=floor(wp-overlap*wp) #floor pour faciliter la redondance d'info
    #w'+k*s=W
    k=ceil((W-wp)/s) #ceil pour faciliter la redondance on s'assure de dépasser un peu de l'image

    #Exemple : k=ceil((15000-1125)/(1125-0.5*1125))
    #                      = 13875/562 = ceil(24,68)=25
    #Erreur sur W:
    #|W-w'-k*s|=-175
    #recalcul du s pour resserrer les crops
    sp=floor((W-wp)/k)  #=555
    #recalcul de l'erreur sur W
    #W-w'-k*s'=0
    #     key='D'+str(W)+'_d'+str(w)+'_o'+str(overlap)+'_s'+str(scale)
    return (sp,k)

def make_img_infos(folder,groundtruth_path,dim=1000, scales = [1., 1.25, 0.8, 0.6, 0.4],force_new_dataset=False):
    """
    Fonction qui construit le dataset, 
    elle liste les images, leurs tailles, le nombre de crops possibles et produit les masks de la VT
    """
    
    saving_path = osp.join(folder,'dataset_dict.p')
    if osp.isfile(saving_path) and not force_new_dataset:
        print("Loading from precedent scan")
        return pickle.load( open( saving_path, "rb" ) )
    img_infos={}
    img_infos['paths']=list_all_images(folder+'/ppms/',fileExtensions = ['ppm'])
    img_infos['mask_paths']=[]
    img_infos['filenames']=[]
    img_infos['size']=[]
    img_infos['spk']=[]
    img_infos['groundtruth']=[]
    img_infos['list_crops']=[]

    groundtruth_csvfile = open(groundtruth_path, 'r')
    # with open(pathToGT, 'r') as csvfile:
    groundtruth_reader = csv.DictReader(groundtruth_csvfile)#, delimiter=',')
    groundtruth_dict = {}
    for row in groundtruth_reader:
        groundtruth_dict.setdefault(row['filename'],[]).append(row)

    #scales = [1., 1.25, 0.8, 0.6, 0.4]
    scales.sort()
    for img_id,img_path in enumerate(tqdm(img_infos['paths'],desc='Image and VT loading',ncols=150)):
        
        filename = osp.splitext(osp.splitext(osp.basename(img_path))[0])[0]
        #filename = osp.splitext(osp.basename(img_path))[0]
        
        img_pil = Image.open(img_path)
        img_infos['filenames'].append(filename)
        img_infos['size'].append(img_pil.size)
        if filename not in groundtruth_dict:
            gt_dict = []
        else:
            gt_dict = groundtruth_dict[filename]
        img_infos['groundtruth'].append(gt_dict)

        # calcul du meilleur stride (sp) et du nombre de crops possibles (k) dans chaque direction (x,y)
        W = img_pil.size[0]
        H = img_pil.size[1]
        spk_img=[]
        for scale in scales:
            spk_img.append((scale,(make_spk(W, w=1000, overlap = 0.5, scale=scale),make_spk(H, w=1000, overlap = 0.5, scale=scale))))
        img_infos['spk'].append(spk_img)
        

                    
                    
        #creation du mask des étiquettes
        mask_path = folder+'masks/'+filename+".ppm"
        img_infos['mask_paths'].append(mask_path)
        # if not osp.isfile(mask_path):
        mkdir_p(folder+'masks')
        mask = Image.new('L',img_pil.size)
        for id_label,label in enumerate(gt_dict):
            label_polygon = [(int(label['x1']),int(label['y1'])),(int(label['x2']),int(label['y2'])),(int(label['x3']),int(label['y3'])),(int(label['x4']),int(label['y4']))]
            ImageDraw.Draw(mask).polygon(label_polygon, outline=id_label, fill=id_label)
        mask.save(mask_path)
            
        #creation des crops
        for spk in spk_img:
            scale = spk[0]
            spy = spk[1][0][0] # stride horizontal
            spx = spk[1][1][0] # stride vertical
            for kyi in range(0,spk[1][0][1]+1):
                yi=spy*kyi
                for kxi in range(0,spk[1][1][1]+1):
                    xi=spx*kxi
                    crop = (img_id,scale,int(xi),int(yi))
                    mask_cropped = mask.crop((crop[3],crop[2],crop[3]+ceil(dim*crop[1]),crop[2]+ceil(dim*crop[1])))
                    if np.amax(np.array(mask_cropped))==0:
                        has_vt=0
                    else:
                        has_vt=1
                    crop = (img_id,scale,int(xi),int(yi),has_vt)    
                    img_infos['list_crops'].append(crop)
        img_infos['crops_with_labels'] = [id for id,crop in enumerate(img_infos['list_crops']) if crop[4]==1]
    print("Saving this scan")
    pickle.dump( img_infos, open( saving_path, "wb" ) )
    return img_infos

def load_crop(crop,img_path,dim=1000):
    return read_from_memory(img_path,crop[2],crop[3],ceil(dim*crop[1]),ceil(dim*crop[1]))
    
def load_crop_mask(crop,img_path,dim=1000):
    img_pil = Image.open(img_path)
    b = img_pil.crop((crop[3],crop[2],crop[3]+ceil(dim*crop[1]),crop[2]+ceil(dim*crop[1])))
    return b
    


import math
import random
import numpy as np
import cv2

from config import Config
import utils


class jalbertConfig(Config):
    """Configuration for training on the jean albert dataset.
    Derives from the base Config class and overrides values specific
    to the jalbert dataset.
    """
    # Give the configuration a recognizable name
    NAME = "jalbert"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + label

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

import copy
def copydataset(dataset,split):
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy.change_split(split)
    dataset_copy.prepare()
    return dataset_copy

class jalbertDataset(utils.Dataset):
    """Generates the jalbert.
    """

    def load_jalbert(self, dim,data_folder,groundtruth_path, scales = [1., 1.25, 0.8, 0.6, 0.4], split=None,force_new_dataset=False):


        self.data_folder = data_folder
        self.groundtruth_path= groundtruth_path
        
        self.dim = dim
        self.scales = scales

        
        convert_to_ppm_by_folder(data_folder)
        #purge_ppm(folder)
        #list_images_path = list_all_images(self.data_folder)
        #for img_path in list_images_path:
        #    convert_to_ppm(img_path)
        
        #print("Loading infos...")
        self.img_infos = make_img_infos(self.data_folder,self.groundtruth_path, self.dim, self.scales, force_new_dataset)
        #print("Loading infos : ok !")
        
        
        # Add classes
        self.add_class("jalbert", 1, "label")
        #self.add_class("jalbert", 2, "caillebotis")
        #self.add_class("jalbert", 3, "chaine")

        # Add images
        self.change_split(split)
        
    def change_split(self,split):
        if split is not None:
            self.split=self.img_infos['crops_with_labels'][split]
        else:
            self.split = self.img_infos['crops_with_labels']
        
        self.image_info = []            
        
        for i in range(len(self.split)):
            self.add_image("jalbert", image_id=i, path=None)
        
    

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        
        crop = self.img_infos['list_crops'][self.split[image_id]]
        #result = np.array(load_crop(crop,self.img_infos['paths'][crop[0]]))
        result = np.array(pil_resize(load_crop(crop,self.img_infos['paths'][crop[0]]),self.dim))
        return result


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        return None

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        
        crop = self.img_infos['list_crops'][self.split[image_id]]
        #print(crop)
        #result = pil_resize(load_crop(crop,self.img_infos['paths'][crop[0]]),self.dim)
        result_mask = pil_resize(load_crop_mask(crop,self.img_infos['mask_paths'][crop[0]]),self.dim)

        result_mask_np = np.asarray(result_mask)
        instances = np.unique(result_mask_np)
        instances = [instance for instance in instances if instance>0]
        if len(instances)>0:
            mask_instance = 0*np.ndarray(shape=(*result_mask_np.shape,len(instances)),dtype=np.uint8)
            for instance_id,instance in enumerate(instances):
                mask_instance[:,:,instance_id]=(result_mask_np==instance).astype(np.uint8)
            mask=mask_instance
            class_ids = np.ones(len(instances), dtype=np.int32)
        else:
            # Call super class to return an empty mask
            return super(jalbertDataset, self).load_mask(image_id)
    
        return mask, class_ids
