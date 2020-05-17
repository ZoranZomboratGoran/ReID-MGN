import torch
from opt import opt

def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, _) in loader:

        input_img = inputs
        if opt.usecpu == False and torch.cuda.is_available():
            input_img = inputs.cuda()
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs
        if opt.usecpu == False and torch.cuda.is_available():
            input_img = inputs.cuda()
        outputs = model(input_img)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features
