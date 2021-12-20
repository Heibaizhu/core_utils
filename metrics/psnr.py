import torch

class PSNR:

    def __init__(self, data_range=1):
        self.data_range = data_range

    def __call__(self, X, Y, data_range=None):
        if data_range is None:
            data_range = self.data_range

        if len(X.shape) != 4:
            raise ValueError('Input images must 4-d tensor.')

        if not X.type() == Y.type():
            raise ValueError('Input images must have the same dtype.')

        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')



        return self.calculate_psnr(X, Y, data_range).item()

    def calculate_psnr(self, X, Y, data_range):
        X, Y = X.type(torch.float), Y.type(torch.float)
        mse = torch.mean((X-Y)**2, dim=[1,2,3])
        psnr = 10 * torch.log10(data_range**2 / mse)
        psnr = torch.mean(psnr)
        return psnr


if __name__ == '__main__':
    """
    from metric import PSNR_NP
    from torchvision import transforms
    img1 = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\clear\clear\1.png')
    img1_haze = io.imread(r'F:\datasets\RESIDE-standard\ITS_v2\hazy\hazy\1_1_0.90179.png')
    from skimage import io

    psnr_np = PSNR_NP()
    psnr_np.update(img1, img1_haze)
    print(psnr_np.val)

    totensor = transforms.ToTensor()
    img1 = totensor(img1).unsqueeze(0)
    img1_haze = totensor(img1_haze).unsqueeze(0)

    psnr = PSNR(data_range=1)
    print(psnr(img1, img1_haze))
    """