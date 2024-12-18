'''Network modules for pytorch models. 

Edited by: Kirsten Mayer
Written by: Elizabeth A. Barnes

Functions
---------
dense_couplet(in_features, out_features, act_fun, *args, **kwargs)
dense_block(out_features, act_fun)

Classes
---------
NeuralNetwork()

'''

import numpy as np
import torch
from base.base_model import BaseModel


def init_weights(m):
    """
    Apply Kaiming or Xavier initialization to Conv2d and ConvTranspose2d layers.
    """
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        # Use Kaiming for ReLU-based activations
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # Alternatively, use Xavier initialization for Sigmoid/Tanh activations
        # torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
def conv_couplet(in_channels, out_channels, act_fun=False, *args, **kwargs):
    if not act_fun:
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels)
        )
    else:
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels),
            getattr(torch.nn, act_fun)()
        )
    block.apply(init_weights)
    return block


def upconv_couplet(in_channels, out_channels, act_fun=False, *args, **kwargs):
    if not act_fun:
        block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels)
        )
    else:
        block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels),
            getattr(torch.nn, act_fun)(),
        )
    block.apply(init_weights)
    return block


def upconv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        upconv_couplet(in_channels, out_channels, act_fun, kernel_size,
                       padding=2, stride=2, output_padding=(1,0)) 
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ] # padding and output_padding are used to get the correct upscale from input size
    return torch.nn.Sequential(*block)
    
######################################### UNET #########################################
class UNet(BaseModel):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True)
        
        self.conv1 = conv_couplet(in_channels=config["down_filters"][0],
                                  out_channels=config["down_filters"][1],
                                  kernel_size=config["down_kernel_size"][0],
                                  act_fun=config["down_act"][0],
                                  padding=config["down_padding"][0],
                                  stride=1
                                  )
        self.conv11 = conv_couplet(in_channels=config["down_filters"][1],
                                  out_channels=config["down_filters"][1],
                                  kernel_size=config["down_kernel_size"][0],
                                  act_fun=config["down_act"][0],
                                  padding=config["down_padding"][0],
                                  stride=1
                                  )
        
        self.conv2 = conv_couplet(in_channels=config["down_filters"][1],
                                  out_channels=config["down_filters"][2],
                                  kernel_size=config["down_kernel_size"][1],
                                  act_fun=config["down_act"][1],
                                  padding=config["down_padding"][1],
                                  stride=1
                                  )
        self.conv22 = conv_couplet(in_channels=config["down_filters"][2],
                                  out_channels=config["down_filters"][2],
                                  kernel_size=config["down_kernel_size"][1],
                                  act_fun=config["down_act"][1],
                                  padding=config["down_padding"][1],
                                  stride=1
                                  )

        self.convbottle = conv_couplet(in_channels=config["down_filters"][2],
                                  out_channels=config["down_filters"][3],
                                  kernel_size=config["down_kernel_size"][2],
                                  act_fun=config["down_act"][3],
                                  padding=config["down_padding"][3],
                                  stride=1
                                  )
        self.convbottle1 = conv_couplet(in_channels=config["down_filters"][3],
                                  out_channels=config["down_filters"][3],
                                  kernel_size=config["down_kernel_size"][2],
                                  act_fun=config["down_act"][3],
                                  padding=config["down_padding"][3],
                                  stride=1
                                  )

        self.upconv1 = upconv_couplet(
            in_channels = config["down_filters"][-1],
            out_channels = config["up_filters"][0],
            kernel_size = config["up_kernel_size"][0],
            act_fun = config["up_act"][0],
            padding=config["up_padding"][0],
            output_padding=config["up_output_padding"][0],
            stride=2
        )
        self.upconv11 = conv_couplet(in_channels=config["up_filters"][0]+ config["down_filters"][2],
                                  out_channels=config["up_filters"][0],
                                  kernel_size=config["up_kernel_size"][0],
                                  act_fun=config["up_act"][0],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )
        self.upconv111 = conv_couplet(in_channels=config["up_filters"][0],
                                  out_channels=config["up_filters"][0],
                                  kernel_size=config["up_kernel_size"][0],
                                  act_fun=config["up_act"][0],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )
        
        self.upconv2 = upconv_couplet(
            in_channels = config["up_filters"][0],
            out_channels = config["up_filters"][1],
            kernel_size = config["up_kernel_size"][1],
            act_fun = config["up_act"][1],
            padding = config["up_padding"][1],
            output_padding = config["up_output_padding"][1], 
            stride = 2
        )
        
        self.upconv22 = conv_couplet(in_channels=config["up_filters"][1]+ config["down_filters"][1],
                                  out_channels=config["up_filters"][1],
                                  kernel_size=config["up_kernel_size"][1],
                                  act_fun=config["up_act"][1],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )

        self.upconv222 = conv_couplet(in_channels=config["up_filters"][1],
                                  out_channels=config["up_filters"][1],
                                  kernel_size=config["up_kernel_size"][1],
                                  act_fun=config["up_act"][1],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )

        self.out = conv_couplet(
            in_channels = config["up_filters"][1] ,
            out_channels = config["up_filters"][-1],
            kernel_size = config["up_kernel_size"][2],
            act_fun=False,
            padding = config["down_padding"][0],
            stride = 1,)
                
    def forward(self,x,device="cuda"):
        
        # x = self.pad_lons(x) #96,154
        # residual1 = x
        conv1 = self.conv1(x) 
        conv11 = self.conv11(conv1) #96,154
        # if residual1.shape != conv11.shape:
        #     #adjust channels to match
        #     convadjust = torch.nn.Conv2d(residual1.shape[1], conv11.shape[1], kernel_size=1).to(device)
        #     residual1 = convadjust(residual1)
        # conv11 = conv11 + residual1
        down1 = self.max_pool2d(conv11) #48,77

        # residual2 = down1
        conv2 = self.conv2(down1)
        conv22 = self.conv22(conv2) #48,77
        # if residual2.shape != conv22.shape:
        #     #adjust channels to match
        #     convadjust = torch.nn.Conv2d(residual2.shape[1], conv22.shape[1], kernel_size=1).to(device)
        #     residual2 = convadjust(residual2)
        # conv22 = conv22 + residual2
        down2 = self.max_pool2d(conv22) #24,39

        # residual_bottle = down2
        convbottle = self.convbottle(down2) #24,39
        # if residual_bottle.shape != convbottle.shape:
        #     #adjust channels to match
        #     convadjust = torch.nn.Conv2d(residual_bottle.shape[1], convbottle.shape[1], kernel_size=1).to(device)
        #     residual_bottle = convadjust(residual_bottle)
        # convbottle = convbottle + residual_bottle
        convbottle1 = self.convbottle1(convbottle) #24,39
        
        # residual_up1 = convbottle
        up1 = self.upconv1(convbottle1) #48,77
        cat1 = torch.cat([conv22, up1], 1) #48,77
        up11 = self.upconv11(cat1) 
        up111 = self.upconv111(up11)
        # if residual_up1.shape != upconv11.shape:
        #     #adjust channels to match
        #     convadjust = torch.nn.Conv2d(residual_up1.shape[1], upconv11.shape[1], kernel_size=1).to(device)
        #     residual_up1 = convadjust(residual_up1)
        #     #adjust image size to match
        #     residual_up1 = torch.nn.functional.interpolate(residual_up1,
        #                                                    size=upconv11.shape[2:],
        #                                                    mode='bilinear', align_corners=False)
        # upconv11 = upconv11 + residual_up1
        

        # residual_up2 = cat1
        up2 = self.upconv2(up111) #96,154
        cat2 = torch.cat([conv11, up2],1) #96,154
        up22 = self.upconv22(cat2)
        up222 = self.upconv222(up22)
        # if residual_up2.shape != upconv22.shape:
        #     #adjust channels to match
        #     convadjust = torch.nn.Conv2d(residual_up2.shape[1], upconv22.shape[1], kernel_size=1).to(device)
        #     residual_up2 = convadjust(residual_up2)
        #     #adjust image size to match
        #     residual_up2 = torch.nn.functional.interpolate(residual_up2,
        #                                                    size=upconv22.shape[2:],
        #                                                    mode='bilinear', align_corners=False)
        # upconv22 = upconv22 + residual_up2
        

        x = self.out(up222)

        if x.size()[2:] != torch.Size([96, 144]):
            raise ValueError("output of UNet needs to be [batch, filter, 96, 154]. It is currently "+str(x.size()))
        else:
            return x #[:,:,:,5:-5]
        
    
    def predict(self, dataset=None, dataloader=None, batch_size=32, device="gpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be none.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError("dataset and dataloader cannot both be defined. choose one.")

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                )

        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, target = (
                    data.to(device),
                    target.to(device),
                    )

                out = self(input).to("cpu").numpy() # this has to be "cpu" to convert to a numpy
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output


######################################### CONVOLUTIONAL NEURAL NETWORK #########################################
def dense_couplet(in_features, out_features, act_fun=False, *args, **kwargs):
    if not act_fun:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True))
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            getattr(torch.nn, act_fun)(),
        )


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def dense_block(out_features, act_fun, in_features=None):
    if in_features is None:
        block = [
            dense_lazy_couplet(out_channels, act_fun)
            for out_channels, act_fun in zip([*out_features], [*act_fun])
        ]
        return torch.nn.Sequential(*block)
    else:
        block = [
            dense_couplet(in_features, out_features, act_fun)
            for in_features, out_features, act_fun in zip(
                [*in_features], [*out_features], [*act_fun]
            )
        ]
        return torch.nn.Sequential(*block)


def conv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        conv_couplet(in_channels, out_channels, act_fun, kernel_size,
                     padding="same", stride = 1)
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ]
    return torch.nn.Sequential(*block)
    
    
class ConvNeuralNetwork(BaseModel):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])

        # CNN block
        self.conv_block = conv_block(
            [config["n_inputchannel"], *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_act"]],
            [*config["kernel_size"]],
        )
        
        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Input Dense blocks
        self.denseblock = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )

        # Final dense layer
        self.output = dense_couplet(
            out_features=config["hiddens_final"],
            # act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

    def forward(self,x):
        
        x = self.pad_lons(x)
        x = self.conv_block(x)
        x = self.flat(x)
        x = self.denseblock(x)
        x = self.output(x)
        
        return x
        
    
    def predict(self, dataset=None, dataloader=None, batch_size=32, device="gpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be none.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError("dataset and dataloader cannot both be defined. choose one.")

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                )

        self.to(device)
        self.eval()
        with torch.inference_mode():

            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, target = (
                    data.to(device),
                    target.to(device),
                    )

                out = self(input).to("cpu").numpy() # this has to be "cpu" to convert to a numpy
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
