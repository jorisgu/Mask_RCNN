from memory_profiler import profile



from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1e10
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

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

def show_thumbnail(img_pil, max_size_thumbnail = 200):
    img_pil_thumbnail = img_pil.copy()
    size = img_pil.size
    max_size_img = float(max(size))
    new_size =  tuple((max_size_thumbnail/max_size_img*np.asarray(result.size)).astype(int))
    img_pil_thumbnail.thumbnail(new_size, Image.ANTIALIAS)
    return img_pil_thumbnail
@profile
def my_func():
    #(960, 230)
    x = 100
    y = 800
    h = 30000
    w = 15000
    result = Image.new('RGB',(w,h))
    img_path = '/dds/workspace/data_ja/cat.ppm'
    i = Image.open(img_path)
    print(i.size)
    print(i.tile)
    for i in range(h):
        a = read_line_portion(img_path, x,y,w,h,i)
        result.paste(a,(0,i))
    
def old():
    i = Image.open(image_path)
    print(i.size)
    print(i.tile)
    
    #os.system("convert "+image_path+" "+image_path[-3:]+"ppm")
    
    #return
    
    ##save 
    #rgb_im = i.convert('RGB')
    #i.save(image_path+'.3.png',optimize=False,compress_level=0)
    
    ## crop
    #left = 100
    #top = 100
    #width = 200
    #height = 100
    #box = (left, top, left+width, top+height)
    #i.crop(box)
    
    #return
    
    
    
    #i = Image.open(image_path)
    w=1000
    h=1000
    i.size = (w, h)
    #i.tile = [('jpeg', (0, 0, w, h), 0, ('RGB', ''))]
    i.tile = [('raw', (0, 0, w, h), 19+2000, ('RGB', 0, 1))]
    print("Changing tile")
    print(i.size)
    print(i.tile)
    i.load()
    print(i.getextrema())

if __name__ == '__main__':
    my_func()
    
    
