
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import random
from urllib.request import urlopen
from nltk.tokenize import RegexpTokenizer
import os
from inference_utils import save_img_tensor
from inference_models import get_init_noise, get_model,from_noise_to_image
import pilgram
import glob
def text2img_get_init_image(args):
    
    if args.model_type in ["sd","sd_unet"]:
        if args.sd_prompt:
            prompt = args.sd_prompt
        else:
            #prompt = "A silver mech horse running in a dark valley, in the night, high-definition picture, unreal engine, cyberpunk"
            prompt = "A cute shiba on the grass"
        #prompt = "Dinning furniture for sale"
        #prompt = "Incoming: New Products from YSL, Maybelline, and more!"
        #prompt = "A cute shiba"
        image,latents = args.cur_model(prompt, num_inference_steps=50, guidance_scale=7.5,get_latents=True)
        image = image.images[0]
        image = transforms.PILToTensor()(image).cuda()/255
        print("sd init image max:",image.max())
    
    return image,latents

def get_image_from_folder(folder_path):
    files = os.listdir(folder_path)
    image_file = random.choice(files)
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    return image
def load_images(folder_path, image_size=(512, 512)):
    image_files = glob.glob(folder_path + "/*.png")  # 修改文件夹路径和文件类型以匹配你的实际情况
    images = []

    for image_file in image_files:
        shiba_img = cv2.imread(image_file)
        b, g, r = cv2.split(shiba_img)
        shiba_img = cv2.merge([r, g, b])
        shiba_img = cv2.resize(shiba_img, image_size, interpolation=cv2.INTER_AREA)
        shiba_img = shiba_img / 255
        images.append(shiba_img)
    return images

def get_image0(args,times):
    
    gt_noise = None
    if args.input_selection == "use_stl10_image0":
        stl10_np = np.load("./data/stl10/train.npz")['x']
        print(stl10_np.shape)
        args.dataset_index = random.randint(0, 4999)
        if args.dataset_index:
            stl10_img = stl10_np[args.dataset_index]
            stl10_img_show = Image.fromarray(stl10_img)
        else:
            stl10_img = stl10_np[5]
            stl10_img_show = Image.fromarray(stl10_img)
            stl10_img_show.save("stl10_img_show.jpg")
        image0 = stl10_img

    if args.input_selection == "use_cifar10_image0":
        cifar10_np = np.load("./data/cifar10/train.npz")['x']
        if args.model_type == "styleganv2ada_cifar10":
            cifar10_np_y = np.load("./data/cifar10/train.npz")['y']
            cifar10_class_index_list = [[] for j in range(10)]
            for index in range(cifar10_np.shape[0]):
                cifar10_class_index_list[cifar10_np_y[index]].append(index)
            rnd_idx = random.randint(0, len(cifar10_class_index_list[args.stylegan_class_idx]))
            args.dataset_index = cifar10_class_index_list[args.stylegan_class_idx][rnd_idx]
        else:
            args.dataset_index = random.randint(0, 49999)

        if args.dataset_index:
            cifar10_img = cifar10_np[args.dataset_index]
            #y_lable = cifar10_np_y[args.dataset_index]
            cifar10_img=cifar10_img.astype('uint8')
            cifar10_img_show = Image.fromarray(cifar10_img)
        else:
            cifar10_img = cifar10_np[5]
            cifar10_img_show = Image.fromarray(cifar10_img)
            cifar10_img_show.save("cifar10_img_show.jpg")
        cifar10_img = cifar10_img/255
        cifar10_img = torch.from_numpy(cifar10_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        
        image0 = cifar10_img
        

    if args.input_selection == "use_imagenet_image0":
        imagenet_dir = "/hy-tmp/train/"
        class_dir_list = os.listdir(imagenet_dir)
        rnd_index = random.randint(0,len(class_dir_list)-1)
        class_dir = imagenet_dir+class_dir_list[rnd_index]+"/"
        print(class_dir)
        png_list = os.listdir(class_dir)
        rnd_index = random.randint(0,len(png_list)-1)
        png_file = class_dir+png_list[rnd_index]
        imagenet_img = cv2.imread(png_file)
        b,g,r = cv2.split(imagenet_img)
        imagenet_img = cv2.merge([r, g, b])
        imagenet_img = imagenet_img/255
        imagenet_img = torch.from_numpy(imagenet_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = imagenet_img
        save_img_tensor(image0,"image0_imagenet.png")

    if args.input_selection == "use_shiba_image0":
        shiba_img = cv2.imread("shiba_images.jpg")
        b,g,r = cv2.split(shiba_img)
        shiba_img = cv2.merge([r, g, b])
        shiba_img = cv2.resize(shiba_img, (512,512), interpolation=cv2.INTER_AREA)
        #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
        print(shiba_img.shape)
        shiba_img_show = Image.fromarray(shiba_img)
        shiba_img_show.save("shiba_img_show.jpg")
        shiba_img = shiba_img/255
        shiba_img = torch.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = shiba_img
    if args.input_selection == "restore_image0":
        folder_path="./caliimg/"
        img_size=(64,64)
        image0=load_images(folder_path,img_size)
        #print(image0.shape)
        #exit('here')
    if args.input_selection == "use_generated_image0":
        with torch.no_grad():
            if args.model_type in ["sd","sd_unet"]:
                image0,gt_noise = text2img_get_init_image(args)
                #gt_noise = None
            elif "cm" in args.model_type:
                gt_noise = get_init_noise(args,args.model_type,bs=args.bs)[0].unsqueeze(0).repeat(args.bs,1,1,1)
                #print(gt_noise)
                #exit("here")
                image0 = from_noise_to_image(args,args.cur_model,gt_noise,args.model_type)
                print(image0)
                #exit("here")
      #          for ip in range(image0.shape[0]):
    # 获取单个样本的图像张量
       #             sample_image = image0[ip]

    # 将 PyTorch 张量保存为图像文件
        #            save_img_tensor(sample_image, f'./conss_model/sample_{ip+1}_times_{times}.png')
                #save_img_tensor(image0,"image0_cm_samenoise.png")
            else:
                gt_noise = get_init_noise(args,args.model_type)[0].unsqueeze(0)
                image0 = from_noise_to_image(args,args.cur_model,gt_noise,args.model_type)
                #print(image0.shape)
                #exit()
                
            # 保存图像
            save_img_tensor(image0, "./original_image0_0204/image0_{}.png".format(times))

    if args.input_selection_model_type != None:
        another_model = get_model(args.input_selection_model_type,args.input_selection_model_path,args)
        with torch.no_grad():
            if "cm" in args.input_selection_model_type:
                another_model_noise = get_init_noise(args,args.input_selection_model_type,bs=args.bs)
                image0 = from_noise_to_image(args,another_model,another_model_noise,args.input_selection_model_type)[0]
            else:
                args.cur_model = another_model
                another_model_noise = get_init_noise(args,args.input_selection_model_type,bs=1)
                image0 = from_noise_to_image(args,another_model,another_model_noise,args.input_selection_model_type)
                #image0_tmp=get_image_from_folder("/hy-tmp/imnet1")
                #下三行是启用consistency_model的
                #image0_tmp=another_model(num_inference_steps=50).images[0]
                #transform=transforms.ToTensor()
                #image0=transform(image0_tmp).to("cuda:0")
                #save_img_tensor(image0_tmp, "./original_image0_ddpm/image0_{}.png".format(times))
            gt_noise = another_model_noise
            #print("gt_noise:",gt_noise.size())
            #exit("see here")

    if args.input_selection_url != None:
        #url_img = cv2.imread(args.input_selection_url)
        readFlag = cv2.IMREAD_COLOR
        resp = urlopen(args.input_selection_url)
        url_img = np.asarray(bytearray(resp.read()), dtype="uint8")
        url_img = cv2.imdecode(url_img, readFlag)

        b,g,r = cv2.split(url_img)
        url_img = cv2.merge([r, g, b])
        url_img = cv2.resize(url_img, (512,512), interpolation=cv2.INTER_AREA)
        #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
        print(url_img.shape)
        url_img_show = Image.fromarray(url_img)
        url_img_show.save("url_img_show.jpg")
        url_img = url_img/255
        url_img = torch.from_numpy(url_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = url_img

    if args.input_selection_name != None:
        shiba_img = cv2.imread(args.input_selection_name)
        b,g,r = cv2.split(shiba_img)
        shiba_img = cv2.merge([r, g, b])
        shiba_img = cv2.resize(shiba_img, (512,512), interpolation=cv2.INTER_AREA)
        #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
        print(shiba_img.shape)
        shiba_img_show = Image.fromarray(shiba_img)
        shiba_img_show.save("input_selection_name_img_show3.jpg")
        shiba_img = shiba_img/255
        shiba_img = torch.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = shiba_img

    if args.input_selection != "use_generated_image0" and args.model_type in ["sd","sd_unet"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        image0 = transforms.Resize(height)(image0)
        save_img_tensor(image0,"image0_sd_not_generated.png")

    '''image0 = transforms.ToPILImage()(image0.squeeze(0).detach().cpu())
    image0.save("image0_beforefiler.jpg")
    image0 = pilgram._1977(image0)
    image0.save("image0_afterfiler.jpg")
    image0 = transforms.PILToTensor()(image0).unsqueeze(0).cuda().float()/255'''
    
    if args.model_type == "ddpm_cifar10":
        imsize = 32
    elif args.model_type == "dcgan_cifar10":
        imsize = 32
    elif args.model_type == "styleganv2ada_cifar10":
        imsize = 32
    elif args.model_type == "vae_cifar10":
        imsize = 32
    elif "cm" in args.model_type:
        imsize = 64
    elif args.model_type in ["sd","sd_unet"]:
        imsize = 512
    
    #image0 = transforms.Resize((imsize,imsize))(image0)
    #for iu in range(image0.shape[0]):
    # 获取单个样本的图像张量
     #   sample_image = image0[iu]

    # 将 PyTorch 张量保存为图像文件
    #save_img_tensor(sample_image, f'./conss_final/sample_{iu+1}_times_{times}.png')
    #save_img_tensor(image0,"./final_image_0204/image0_final_{}.png".format(times))

    return image0, gt_noise




