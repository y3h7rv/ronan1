"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import cv2
import argparse
import os
import sys
sys.path.append('/root/guided-diffusion')#到guided_diffusion包的路径
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
    #logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    init_noise=th.randn(*(args.batch_size, 3, args.image_size, args.image_size), device="cuda")#(1,3,64,64)
    
    #get image0
    shiba_img = cv2.imread("shiba_images.jpg")
    b,g,r = cv2.split(shiba_img)
    shiba_img = cv2.merge([r, g, b])
    shiba_img = cv2.resize(shiba_img, (64,64), interpolation=cv2.INTER_AREA)
    #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
    print(shiba_img.shape)
    shiba_img_show = Image.fromarray(shiba_img)
    shiba_img_show.save("shiba_img_show.jpg")
    shiba_img = shiba_img/255
    shiba_img = th.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
    image0 = shiba_img
    image0 = image0.detach()
    
    cur_noise = th.nn.Parameter(th.tensor(init_noise)).cuda()
    optimizer = th.optim.Adam([cur_noise], lr=0.1)
    criterion = th.nn.MSELoss(reduction='none')
    
    
    
    
    for i in range(10):
    #重建
    
        logger.log("sampling...")
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                noise=cur_noise,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                
                device=dist_util.dev(),
            )
            
            
            loss = criterion(sample,image0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
            
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            
            
            
        with torch.no_grad():
            folder_name="/hy-tmp/test"
            for iu in range(image.shape[0]):
                # 获取单个样本的图像张量
                sample_image = image[iu]

                # 将 PyTorch 张量保存为图像文件
                #save_image(sample_image, f'sample_{i+1}.png')
                torchvision.utils.save_image(sample_image, "{}/image_{}.png".format(folder_name,iu))
            #gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            #dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            #all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            # logger.log(f"created {len(all_images) * args.batch_size} samples")

        # arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples]
        # label_arr = np.concatenate(all_labels, axis=0)
        # label_arr = label_arr[: args.num_samples]
#         if dist.get_rank() == 0:
#             shape_str = "x".join([str(x) for x in arr.shape])
#             out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
#             logger.log(f"saving to {out_path}")
#             np.savez(out_path, arr, label_arr)

#         dist.barrier()
        logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
