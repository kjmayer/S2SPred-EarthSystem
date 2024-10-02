import torch


# def custom_accuracy(output, target):
#     """Compute the prediction accuracy
#     """
#     with torch.no_grad():

#         assert len(output) == len(target)
        
#         num_correct = (output.argmax(1) == target.argmax(1)).type(torch.float).sum()
        
#         return ((num_correct/torch.tensor(len(output), dtype=torch.float32))*100).item()


def MAE(output,target):
    with torch.no_grad():
        assert len(output) == len(target)
        MAE = torch.nn.functional.l1_loss(output, target)
    
    return MAE.item()