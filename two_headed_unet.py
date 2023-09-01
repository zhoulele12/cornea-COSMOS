""" Full assembly of the parts to form the complete network """

from unet_parts import *
from PIL import Image
from torchvision import transforms
class twoHeadedUNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        # bilinear not fully implemented
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.encoder = Encoder(self.n_channels)
        self.translation_decoder = Decoder(1,bilinear)
        self.segmentor = Decoder(1,bilinear)

    def forward(self, x):
        encoded_feats = self.encoder(x)
        translated_out = self.translation_decoder(encoded_feats)
        seg_out = self.segmentor(encoded_feats)

        return (translated_out,seg_out)

class Encoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels,64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512,1024)


    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        return [x1,x2,x3,x4]


class Decoder(nn.Module):
    def __init__(self,n_classes,bilinear):
        super().__init__()
        # self.up1 = Up(1024, 512,bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64,n_classes)

    def forward(self,x_list):
        # y1 = self.up1(x_list[4], x_list[3])
        y2 = self.up2(x_list[3], x_list[2])
        y3 = self.up3(y2, x_list[1])
        y4 = self.up4(y3, x_list[0])
        logits = self.outc(y4)

        return logits

# mac = torch.device("mps")
# device = mac
# model = twoHeadedUNet(1)
# model.to(device)
# img1 = Image.open("Cornea_New/HE/HE/AZARN_002.tif")
# convert = transforms.Compose([transforms.ToTensor()])
# tensor1 = convert(img1)[0,:,:].resize(1,1,868,768)
# tensor1 = tensor1.to(device)
#
# out = model(tensor1)