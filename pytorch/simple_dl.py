import torch 
import torch.nn as nn

device="cuda"

X=torch.tensor([[1],[2],[3],[4]],device=device,dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]],device=device,dtype=torch.float32)

#selecting the linear regression model
model=nn.Linear(1,1).to(device)

#declaring the loss function
l=nn.MSELoss()

#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

n_iter=1000

for epoch in range(n_iter):
    y_pred=model(X)

    loss=l(y_pred,Y)
    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch%100==0:
        w=model.weight.item()
        print(f'epoch {epoch+1},weight={w:.3f},loss={loss:.3f}')
with torch.no_grad():
    print(f'Prediction after training: f(5)={model(torch.tensor([[5.0]],device=device)).item():.3f}')