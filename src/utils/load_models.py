import src.networks as network

def get_model(classifier, pretrained=True, num_classes=10, state_dict_path=None, device=None):
    if classifier == "resnet11_fedml":
        net = network.resnet_fedml.resnet11_fedml(class_num=num_classes)
        if pretrained:
            try:
              state_dict = torch.load(state_dict_path, map_location=device)['net']
              from collections import OrderedDict
              new_state_dict = OrderedDict()
              for k, v in state_dict.items():
                  name = k.replace('module.', '')
                  new_state_dict[name]=v
            except:
              new_state_dict = torch.load(state_dict_path, map_location=device)
            net.load_state_dict(new_state_dict)
        return net
    elif classifier == "resnet56_fedml":
        net = network.resnet_fedml.resnet56_fedml(class_num=num_classes)
        if pretrained:
            state_dict = torch.load(state_dict_path, map_location=device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name]=v
            net.load_state_dict(new_state_dict)
        return net
    elif classifier == "resnet18":
        net = network.resnet.resnet18(num_classes=num_classes)
        if pretrained:
            try:
              state_dict = torch.load(state_dict_path, map_location=device)['net']
              from collections import OrderedDict
              new_state_dict = OrderedDict()
              for k, v in state_dict.items():
                  name = k.replace('module.', '')
                  new_state_dict[name]=v
            except:
              new_state_dict = torch.load(state_dict_path, map_location=device)
            net.load_state_dict(new_state_dict)
        return net
    elif classifier == "resnet10":
        net = network.resnet.resnet10(num_classes=num_classes)
        return net
    elif classifier == "resnet34":
        net = network.resnet.resnet34(num_classes=num_classes)
        if pretrained:
            state_dict = torch.load("./network/resnet34-333f7ec4.pth", map_location=device)
            init_state_dict = net.state_dict()
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'fc' not in k:
                    new_state_dict[k]=v
                elif 'fc.weight' in k:
                    new_state_dict[k]=init_state_dict[k]
            net.load_state_dict(new_state_dict)
        
        return net
    elif classifier == "resnet50":
        net = network.resnet.resnet50(num_classes=num_classes)
        if pretrained:
            state_dict = torch.load("./network/resnet34-333f7ec4.pth", map_location=device)
            init_state_dict = net.state_dict()
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'fc' not in k:
                    new_state_dict[k] = v
                elif 'fc.weight' in k:
                    new_state_dict[k] = init_state_dict[k]
            net.load_state_dict(new_state_dict)

        return net
    elif classifier == "vgg19":
        net = network.vgg.vgg19(num_classes=num_classes)
        return net
    else:
        raise NameError('Please enter a valid classifier')