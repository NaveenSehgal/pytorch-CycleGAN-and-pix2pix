import os
import shutil
from tqdm import tqdm
from options.test_options import TestOptions
from models import create_model
from PIL import Image
from data.base_dataset import get_transform
from util.util import tensor2im
import matplotlib.pyplot as plt

dirs = ["SYN_RR_amir_180329_0624_G20190212_1843_P2000_A00",
        "SYN_RR_behnaz_180118_2009_G20190212_1854_P2000_A00",
        "SYN_RR_boya_G20190216_1337_P2000_A00",
        "SYN_RR_chen_G20190216_1438_P2000_A00",
        "SYN_RR_dan_jacket_180521_1312_G20190605_0832_P2000_A00",
        "SYN_RR_eddy_no_coat_180517_G20190212_1950_P2000_A00",
        "SYN_RR_jianchao_G20190216_1507_P2000_A00",
        "SYN_RR_jinpeng_G20190216_1929_P2000_A00",
        "SYN_RR_kefei_G20190217_1000_P2000_A00",
        "SYN_RR_kian_180517_1605_G20190212_1953_P2000_A00",
        "SYN_RR_kian_jacket_180517_1617_G20190605_0832_P2000_A00",
        "SYN_RR_naveen_180403_1612_G20190212_1953_P2000_A00",
        "SYN_RR_naveen_180403_1635_G20190605_1348_P2000_A00",
        "SYN_RR_ray_G20190217_1000_P2000_A00",
        "SYN_RR_sarah_171201_1045_G20190213_0752_P2000_A00",
        "SYN_RR_sarah_180423_1211_G20190213_0752_P2000_A00",
        "SYN_RR_sarah_180423_1220_G20190213_0753_P2000_A00",
        "SYN_RR_sarah_180423_1317_G20190213_0753_P2000_A00",
        "SYN_RR_sharyu_G20190217_1238_P2000_A00",
        "SYN_RR_shiva_G20190217_1239_P2000_A00",
        "SYN_RR_shuangjun_180403_1734_G20190213_0753_P2000_A00",
        "SYN_RR_shuangjun_180403_1748_G20190213_0810_P2000_A00",
        "SYN_RR_shuangjun_180502_1536_G20190213_0810_P2000_A00",
        "SYN_RR_shuangjun-2_G20190217_1239_P2000_A00",
        "SYN_RR_shuangjun_blackT_180522_1542_G20190604_2240_P2000_A00",
        "SYN_RR_shuangjun_blueSnow_180521_1531_G20190604_2241_P2000_A00",
        "SYN_RR_shuangjun_G20190217_1239_P2000_A00",
        "SYN_RR_shuangjun_grayDown_180521_1516_G20190604_2241_P2000_A00",
        "SYN_RR_shuangjun_grayT_180521_1658_G20190605_1031_P2000_A00",
        "SYN_RR_shuangjun_gridDshirt_180521_1548_G20190604_2243_P2000_A00",
        "SYN_RR_shuangjun_jacketgood_180522_1628_G20190605_0831_P2000_A00",
        "SYN_RR_shuangjun_nikeT_180522_1602_G20190605_1030_P2000_A00",
        "SYN_RR_shuangjun_whiteDshirt_180521_1600_G20190605_0834_P2000_A00",
        "SYN_RR_steve_2_good_color_G20190605_1348_P2000_A00",
        "SYN_RR_william_180502_1449_G20190213_0810_P2000_A00",
        "SYN_RR_william_180502_1509_G20190213_0810_P2000_A00",
        "SYN_RR_william_180503_1704_G20190213_0810_P2000_A00",
        "SYN_RR_yu_170723_1000_G20190213_0810_P2000_A00",
        "SYN_RR_zishen_G20190217_1239_P2000_A00",]
in_path = "/scratch/sehgal.n/datasets/synthetic/{}"
out_path = "/scratch/sehgal.n/cyclegan_seg_synthetic/{}"

opt = TestOptions().parse()


def get_model():
    # Hard code options required for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    model = create_model(opt)
    model.setup(opt)

    return model


def get_image_paths(input_dir, output_dir):
    """ Returns dictionary matching image input paths to their relative output paths """
    # Find all images in input directory
    image_paths = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                img_input_path = os.path.join(root, file)
                img_output_path = output_dir + root.replace(
                    input_dir, '') + "/{}".format(file)

                image_paths[img_input_path] = img_output_path

    return image_paths


def configure_output_dirs(input_dir, output_dir):
    """ We want to copy input dir structure to output dir """
    def ig_f(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    shutil.copytree(input_dir, output_dir, ignore=ig_f)


def run_directory(in_dir, out_dir):
    image_paths = get_image_paths(in_dir, out_dir)
    configure_output_dirs(in_dir, out_dir)
    model = get_model()
    transform_A = get_transform(opt, grayscale=0)

    for input_file, output_file in tqdm(image_paths.items()):
        # Load input data
        data = Image.open(input_file).convert('RGB')
        data = {"A": transform_A(data).unsqueeze(0), "A_paths": input_file}

        # Inference on input data
        model.set_input(data)
        model.test()
        faked = model.get_current_visuals()['fake']
        faked_img = tensor2im(faked)

        # Save output data
        plt.imsave(output_file, faked_img)


if __name__ == '__main__':
    for directory in dirs:
        run_directory(in_path.format(directory), out_path.format(directory))
