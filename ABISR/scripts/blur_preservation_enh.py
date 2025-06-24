import cv2
import numpy as np
import os
import multiprocessing as mp
from multiprocessing import set_start_method
from tqdm import tqdm
import argparse

def enhance_image(image_path):
    img_color = cv2.imread(image_path)

    kernel_size = (5, 5)
    sigma = 1.5 

    img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(img_lab) 

    blurred_l = cv2.GaussianBlur(l_channel, kernel_size, sigma)  
    high_pass_l = cv2.subtract(l_channel, blurred_l) 

    alpha = 1.0
    sharpened_l = cv2.addWeighted(l_channel, 1 + alpha, blurred_l, -alpha, 0)
    sharpened_l = cv2.addWeighted(sharpened_l, 1, high_pass_l, 1, 0)

    sharpened_lab = cv2.merge((sharpened_l, a_channel, b_channel))
    sharpened_img = cv2.cvtColor(sharpened_lab, cv2.COLOR_Lab2BGR)

    sharpened_img_denoised = cv2.fastNlMeansDenoisingColored(sharpened_img, None, 2, 2, 7, 21)

    return sharpened_img_denoised

def process_single_img(image_paths, output_dir):
    for image_path in image_paths:
        try:
            enhanced_img = enhance_image(image_path)

            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(input_dir, output_dir, num_workers):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    chunk_size = len(image_paths) // num_workers
    chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    set_start_method('spawn', force=True)
    processes = []
    for idx in range(num_workers):
        chunk = chunks[idx] if idx < len(chunks) else []
        p = mp.Process(target=process_single_img, args=(chunk, output_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime Image Enhancement")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="输入文件夹路径")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="输出文件夹路径")
    parser.add_argument('--num_workers', type=int, default=6, help="并行处理的进程数量")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.num_workers)
