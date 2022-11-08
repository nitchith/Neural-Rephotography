import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import argparse
import os
from tqdm import tqdm
from dominate import document
from dominate.tags import *

def create_html(abs_path, rel_path, output_html):
    photos = sorted(glob.glob(abs_path + '/*.png'))
    with document(title='Photos') as doc:
        h1('Photos')
        for path in photos:
            path = path.replace(abs_path, rel_path)
            div(img(src=path), _class='photo')
    with open(output_html, 'w') as f:
        f.write(doc.render())
    return

def PSNR(pred_rgb, gt_rgba, count=None):

    assert pred_rgb.shape[-1] == 3
    assert gt_rgba.shape[-1] == 4

    pred = pred_rgb
    alpha = gt_rgba[:,:,3:]
    gt = alpha *  gt_rgba[:, :, :3] + (1 - alpha) * 1.0

    if not count:
        count = gt.size

    psnr = -10. * np.log10(np.sum(np.square(pred - gt))/ count)

    return psnr


# python compare_outputs.py --json /home/srinitca/capstone/nerf-pytorch/data/nerf_smallfstop/transforms_test.json --logs /home/srinitca/capstone/NeReFocus/tmp_defocus_results/train_4stacks/lego/test_smallfstop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="/home/srinitca/capstone/nerf-pytorch/data/nerf_render/transforms_test.json")
    parser.add_argument("--logs", type=str, default="/home/srinitca/capstone/NeReFocus/tmp_defocus_results/train_4stacks/lego/test_preds_all")
    parser.add_argument("--out_dir", type=str, default="/home/srinitca/capstone/NeReFocus/results/")
    parser.add_argument("--num_cols", type=int, default=6)
    args = parser.parse_args()

    suffix = "./" + args.logs.split("/")[-3] + "/" + args.logs.split("/")[-1] + "/"
    output_dir = args.out_dir + "/" + suffix

    output_html = args.out_dir + "/" + args.logs.split("/")[-3] + "_" + args.logs.split("/")[-1] + ".html"
    output_psnr = output_html.replace("html", "csv")

    os.makedirs(output_dir, exist_ok=True)

    # Get GT paths
    gt_img_path = []
    with open(args.json, 'r') as f:
        lego_data = json.load(f)
        for frame in lego_data["frames"]:
            gt_img_path.append(frame["file_path"] + ".png")
        
    # Get rendered paths    
    render_img_path = sorted(glob.glob(args.logs + "/color_*.png"))
    
    assert len(render_img_path) == len(gt_img_path)

    psnr_data = []

    for gt_path, render_path in tqdm(zip(gt_img_path, render_img_path)):
        gt_img_name = gt_path.split("/")[-1].split(".")[0]
        gt_img = plt.imread(gt_path)
        fig, axarr = plt.subplots(1, 2, figsize=(2 * 6, 1 * 6))
        axarr[0].imshow(gt_img)
        axarr[0].title.set_text("Ground Truth")
        axarr[0].get_xaxis().set_ticks([])
        axarr[0].get_yaxis().set_ticks([])

        render_img_name = render_path.split("/")[-1].split(".")[0]
        render_img = plt.imread(render_path)
        axarr[1].imshow(render_img)
        axarr[1].title.set_text("Rendered Image")
        axarr[1].get_xaxis().set_ticks([])
        axarr[1].get_yaxis().set_ticks([])

        psnr = PSNR(render_img, gt_img)
        psnr_data.append(psnr)

        fig.suptitle(f"Image id = {gt_img_name}, PSNR = {psnr:.2f}")
        fig.tight_layout()

        output_img_name = output_dir + render_img_name + "_" + gt_img_name + ".png"
        
        plt.savefig(output_img_name)
        plt.close()
        # break

    create_html(output_dir, suffix, output_html)

    psnr_data = np.array(psnr_data)
    pad = -len(psnr_data) % args.num_cols
    zero_pad = np.concatenate((psnr_data,np.zeros(pad)))
    psnr_data = zero_pad.reshape((-1, args.num_cols))
    np.savetxt(output_psnr, psnr_data, delimiter=",", fmt='%.2f')



if __name__ == "__main__":
    main()