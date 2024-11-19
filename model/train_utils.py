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

def compute_padding(input_shape, output_shape, kernel_size, stride, dilation=1):
    
    # Effective kernel size considering dilation
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    # Compute total padding needed
    total_padding = stride * (input_shape - 1) + effective_kernel_size - output_shape
    # Padding and output padding
    padding = max(0, total_padding // 2)
    output_padding = total_padding % 2
    return padding, output_padding


def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)()
    )


def upconv_couplet(in_channels, out_channels, act_fun=False, *args, **kwargs):
    if not act_fun:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs))
    else:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            getattr(torch.nn, act_fun)(),
        )


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

        image_shape = [96,154] # [H,W]
        shape1 = [48,77]
        shape2 = [24,39]
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

        # padding1_h, output_padding1_h = compute_padding(input_shape=shape2[0],
        #                                             output_shape=shape1[0],
        #                                             kernel_size=config["up_kernel_size"][1],
        #                                             stride=2)
        # padding1_w, output_padding1_w = compute_padding(input_shape=shape2[1],
        #                                             output_shape=shape1[1],
        #                                             kernel_size=config["up_kernel_size"][1],
        #                                             stride=2)
        

        # print(padding1_w, padding1_h)
        # print(output_padding1_w, output_padding1_h)
        
        self.upconv1 = upconv_couplet(
            in_channels = config["down_filters"][-1],
            out_channels = config["up_filters"][0],
            kernel_size = config["up_kernel_size"][0],
            act_fun = config["up_act"][0],
            padding=config["up_padding"][0],#[padding1_h,padding1_w], #
            output_padding=config["up_output_padding"][0],#[output_padding1_h,output_padding1_w], #
            stride=2
        )

        self.upconv11 = conv_couplet(in_channels=config["up_filters"][0],
                                  out_channels=config["up_filters"][0],
                                  kernel_size=config["up_kernel_size"][0],
                                  act_fun=config["up_act"][0],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )
        
        # padding2_h, output_padding2_h = compute_padding(input_shape=shape1[0],
        #                                             output_shape=image_shape[0],
        #                                             kernel_size=config["up_kernel_size"][1],
        #                                             stride=2)
        # padding2_w, output_padding2_w = compute_padding(input_shape=shape1[1],
        #                                             output_shape=image_shape[1],
        #                                             kernel_size=config["up_kernel_size"][1],
        #                                             stride=2)

        # print(padding2_w, padding2_h)
        # print(output_padding2_w, output_padding2_h)
        self.upconv2 = upconv_couplet(
            in_channels = config["up_filters"][0] + config["down_filters"][2],
            out_channels = config["up_filters"][1],
            kernel_size = config["up_kernel_size"][1],
            act_fun = config["up_act"][1],
            padding = config["up_padding"][1],#[padding2_h,padding2_w],#
            output_padding = config["up_output_padding"][1], #[output_padding2_h,output_padding2_w],#
            stride = 2,
        )

        self.upconv22 = conv_couplet(in_channels=config["up_filters"][1],
                                  out_channels=config["up_filters"][1],
                                  kernel_size=config["up_kernel_size"][1],
                                  act_fun=config["up_act"][1],
                                  padding=config["down_padding"][0], # "same"
                                  stride=1
                                  )

        # padding3_h, output_padding3_h = compute_padding(input_shape=image_shape[0],
        #                                             output_shape=image_shape[0],
        #                                             kernel_size=config["up_kernel_size"][2],
        #                                             stride=1)
        # padding3_w, output_padding3_w = compute_padding(input_shape=image_shape[1],
        #                                             output_shape=image_shape[1],
        #                                             kernel_size=config["up_kernel_size"][2],
        #                                             stride=1)

        # print(padding3_w, padding3_h)
        # print(output_padding3_w, output_padding3_h)
        self.out = upconv_couplet(
            in_channels = config["up_filters"][1] + config["down_filters"][1],
            out_channels = config["up_filters"][-1],
            kernel_size = config["up_kernel_size"][2],
            padding = config["up_padding"][2],#[padding3_h,padding3_w],#config["up_padding"][2],
            output_padding = config["up_output_padding"][2],#[output_padding3_h,output_padding3_w],#
            stride = 1,
        )

    def forward(self,x):
        
        x = self.pad_lons(x) #96,154
        
        conv1 = self.conv1(x) 
        conv11 = self.conv11(conv1) #96,154
        down1 = self.max_pool2d(conv11) #48,77
        
        conv2 = self.conv2(down1)
        conv22 = self.conv22(conv2) #48,77
        down2 = self.max_pool2d(conv22) #24,39

        convbottle = self.convbottle(down2) #24,39
        
        up1 = self.upconv1(convbottle) #48,77
        upconv11 = self.upconv11(up1) 
        cat1 = torch.cat([conv22, upconv11], 1) #48,77
        
        up2 = self.upconv2(cat1) #96,154
        upconv22 = self.upconv22(up2)
        cat2 = torch.cat([conv11, upconv22],1) #96,154

        x = self.out(cat2)

        if x.size()[2:] != torch.Size([96, 154]):
            raise ValueError("output of UNet needs to be [batch, filter, 96, 154]. It is currently "+str(x.size()))
        else:
            return x[:,:,:,5:-5]
        
    
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
            [config["n_inputchannel"], *config["down_filters"][:-1]],
            [*config["down_filters"]],
            [*config["down_act"]],
            [*config["down_kernel_size"]],
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
