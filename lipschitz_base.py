import torch

def lipschitz_cal(model, dataset):
    for i, cluster in enumerate(dataset):
        datapoint = []
        for i in range(150):
            datapoint.append(cluster['centroid'] + cluster['diameter'] / 2 * torch.rand(*cluster['centroid'].shape))

        datapoint = torch.stack(datapoint)
        datapoint.to('cuda')
        model.to('cuda')
        model.eval()
        with torch.no_grad():
            output = model(datapoint)

        norm_max_output = torch.nn.functional.pdist(output, p=torch.inf)
        norm_max_input = torch.nn.functional.pdist(datapoint, p=torch.inf)

        lipschitz = norm_max_output / norm_max_input
        dataset[i]['lipschitz_const'] = torch.max(lipschitz).item()
