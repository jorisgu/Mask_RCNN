from memory_profiler import profile



from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
image_path='/dds/workspace/data_ja/CAT_1_9013.jpg'
image_path='/dds/workspace/data_ja/CAT_1_9013.bmp'
#image_path='/dds/workspace/data_ja/test.jpg'
#image_path='/dds/workspace/data_ja/test.jpg.png'

@profile
def my_func():
    i = Image.open(image_path)
    print(i.size)
    print(i.tile)
    
    
    
    ##save 
    #rgb_im = i.convert('RGB')
    #i.save(image_path+'.3.png',optimize=False,compress_level=0)
    
    ## crop
    #left = 100
    #top = 100
    #width = 200
    #height = 100
    #box = (left, top, left+width, top+height)
    #area = i.crop(box)
    
    
    
    
    i = Image.open(image_path)
    w=10000
    h=10000
    i.size = (w, h)
    i.tile = [('raw', (0, 0, w, h), 138, ('BGR', 98376, -1))]
    print("Changing tile")
    print(i.size)
    print(i.tile)
    i.load()

if __name__ == '__main__':
    my_func()
    
    
