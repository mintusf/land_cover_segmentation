import torch


def training_loop(
    epochs,
    train_loader,
    val_loader,
    optimizer,
    model,
    criterion,
    device,
    verbose_step=100,
    val_step=1000,
):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["input"]
            labels = data["target"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % verbose_step == 0:  # print every N mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / verbose_step)
                )
                running_loss = 0.0

    print("Finished Training")
