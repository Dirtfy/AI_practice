import torch.nn as nn

from Block import *

class CnnClassifier(nn.Module):
    def __init__(self,
                 input_shape,
                 num_class,
                 depth=4):
        super().__init__()

        channel_schedule = [input_shape[0], *[64*(2**i) for i in range(depth)]]
        print(channel_schedule)
        group_schedule = [1, *[8 for _ in range(depth)]]

        module_list = []
        for i in range(depth):
            module_list.append(
                GSC(num_group=group_schedule[i],
                    in_channels=channel_schedule[i],
                    out_channels=channel_schedule[1+1]),
            )
            for _ in range(2):
                module_list.append(
                    GSC(num_group=group_schedule[i],
                        in_channels=channel_schedule[i+1],
                        out_channels=channel_schedule[1+1]),
                )
                
            module_list.append(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        self.feature_extracter = nn.Sequential(*module_list)

        self.classifier = nn.Sequential(
            SF(in_features=(channel_schedule[4])*(input_shape[1]*input_shape[2])/(depth),out_features=4096),
            nn.Dropout1d(),
            SF(in_features=4096,out_features=1000),
            nn.Dropout1d(),
            SF(in_features=1000,out_features=num_class)
        )
        

    def forward(self, x):
        x = self.feature_extracter(x)
        x = x.flatten(2)
        x = self.classifier(x)
        return x

