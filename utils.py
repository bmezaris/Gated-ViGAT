import torch


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    # return torch.softmax(y/temperature, dim=-1)
    return torch.sigmoid(y/temperature)


def gumbel_sigmoid(logits, temperature, thresh, hard=False):
    """
    ST-gumple-sigmoid
    input: [*, n_class]
    return: flatten --> [*, n_class] a multi-hot vector
    """
    y = gumbel_sigmoid_sample(logits, temperature)

    if not hard:
        return y

    y_hard = y.ge(thresh).to(torch.float32)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
