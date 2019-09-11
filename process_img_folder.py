from util.util import tensor2im
import os
from options.test_options import TestOptions
from data import create_dataset
import matplotlib.pyplot as plt
from models import create_model


def fileList(source):
    matches = []
    for root, dirname, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith('.png'):
                matches.append(os.path.join(root, filename))

opt = TestOptions().parse()

# Hard code options required for test
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1

in_dir = opt.indir
out_dir = opt.outdir

assert in_dir and out_dir, "Must specify an input and output directory"
assert os.path.isdir(in_dir) and os.path.isdir(out_dir), "In and out paths must be directories"

image_files = fileList(in_dir)
import pdb; pdb.set_trace()

# Create dataset and model
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)

model.eval()
for i, data in enumerate(dataset):
    import pdb; pdb.set_trace()
    model.set_input(data)
    model.test()
    faked = model.get_current_visuals()['fake']
    faked_img = tensor2im(faked)
    img_path = model.image_paths

    print("Writing {}".format(img_path))
    plt.imsave('test.png', faked_img)



'''
model.set_input(data)
model.test()
visuals = model.get_current_visuals()
fake = visuals['fake']


'''

