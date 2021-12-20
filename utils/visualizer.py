import os
from . import html
from . import misc
from .img_utils import tensor2img, imwrite
from .dist_utils import master_only


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.name = opt['name']
        self.win_size = opt.get('display_winsize', 256)
        self.web_dir = opt['path']['web']
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.init_visualizer()

    def save_image(self, image_tensor, img_path):
        image_tensor = image_tensor[0]  # save one image
        image = tensor2img(image_tensor)
        imwrite(image, img_path)

    @master_only
    def init_visualizer(self):
        print('create web directory %s...' % self.web_dir)
        misc.mkdir(self.img_dir)

    @master_only
    def __call__(self, visuals, epoch, image_format='jpg'):
        for label, image_tensor in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.%s' % (epoch, label, image_format))
            self.save_image(image_tensor, img_path)
        # update website
        webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
        for n in range(epoch, -1, -1):
            webpage.add_header('epoch [%d]' % n)
            ims = []
            txts = []
            links = []
            for label, _ in visuals.items():
                img_path = 'epoch%.3d_%s.%s' % (n, label, image_format)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.win_size)
        webpage.save()
