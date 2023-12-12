import torch


def accuracy_food(output, target):
    mask_tensor1 = output > 0.5
    mask_tensor2 = target == 1

    # 找出两个张量中哪些位置同时大于0.5
    common_mask = mask_tensor1 & mask_tensor2

    # 计算同时大于0.5的值的数量
    count = common_mask.sum()

    train_acc = count / output.numel()
    return train_acc

a = torch.tensor([0.51,2,0.1])
b = torch.tensor([1,1,0])
print(accuracy_food(a,b))

