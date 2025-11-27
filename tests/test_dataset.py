from spectrum.dataset import YahooDataset, KPIDataset

dataset = YahooDataset(_id=1)
print(dataset)
for data in dataset:
    print(data)
    break
dataset = YahooDataset(_id=2)
print(dataset)

dataset = KPIDataset("0efb375b-b902-3661-ab23-9a0bb799f4e3")
print(dataset)
for data in dataset:
    print(data)
    break